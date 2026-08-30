from gstatsMCMC.MCMC import *
from gstatsMCMC import Topography
import torch
import cupy as cp
from copy import deepcopy
from . import gstatsim_custom_gpu as gsim
import numbers 
from . import SGS_GPU as sgs_cxt

from . import MC_res

"""
Cupy
- Invoke kernel (e.g. Elementwise, ReductionKernel, RawKernel)

Goals:
1. Minimize CPU to GPU memory transfer
2. Keep RNG on GPU
"""


""""""

def get_mass_conservation_residual_GPU(bed, surf, velx, vely, dhdt, smb, resolution):
    thick = surf - bed
    
    dx = cp.gradient(velx*thick, resolution, axis=1)
    dy = cp.gradient(vely*thick, resolution, axis=0)
    
    res = dx + dy + dhdt - smb
    
    return res


def move_cursor_to_line(line_number):
    """Move cursor to specific line for updating in place"""
    sys.stdout.write(f'\033[{line_number};0H')  # Move to line N, column 0
    sys.stdout.flush()

def clear_line():
    """Clear the current line"""
    sys.stdout.write('\033[2K')
    sys.stdout.flush()


# code adopted from gstatsim_custom by Michael
def _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.

    Returns:
        (out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil)
    """
    
    # get masks and gaussian transform data
    cond_msk = ~cp.isnan(grid)

    #out_grid, nst_trans = gaussian_transformation(grid, cond_msk)
    out_grid = grid.copy()

    if sim_mask is None:
        sim_mask = cp.full(xx.shape, True)

    # get index coordinates and filter with sim_mask
    xx_rows, xx_cols = cp.arange(xx.shape[0]) , cp.arange(xx.shape[1])
    ii, jj = cp.meshgrid(xx_rows, xx_cols, indexing = 'ij')

    filtered_ind_flat =  [ii[sim_mask].flatten(), jj[sim_mask].flatten()]
    inds = cp.array(filtered_ind_flat).T

    vario = deepcopy(variogram)

    # turn scalar variogram parameters into grid
    for key in vario:
        if isinstance(vario[key], numbers.Number):
            vario[key] = cp.full(grid.shape, vario[key])

    # mean of conditioning data for simple kriging
    global_mean = cp.mean(out_grid[cond_msk])

    # make stencil for faster nearest neighbor search
    if stencil is None:
        stencil, _, _ = gsim.neighbors_gpu.make_circle_stencil_gpu_safe(xx[0, :], radius) # Stencil, [xx_st, yy_st]
    return out_grid, cond_msk, inds, vario, global_mean, stencil

def _preprocess_gpu_safe(xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb, dtype=cp.float32):
    """Common setup: Gaussian transform, index generation, and stencil creation."""
    cond_msk = ~cp.isnan(grid)

    # 1. Normal Score Transform
    out_grid, nst_trans = gsim.utilities_gpu.gaussian_transformation_gpu(grid, cond_msk, dtype=dtype)

    # 2. Determine simulation path (mask)
    if sim_mask is None:
        sim_mask = cp.full(xx.shape, True)

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')
    inds = cp.stack([ii[sim_mask].flatten(), jj[sim_mask].flatten()], axis=1)

    # 3. Clean Variogram dict
    vario = deepcopy(variogram)
    for k in vario:
        if isinstance(vario[k], numbers.Number):
            vario[k] = float(vario[k])
            
    global_mean = float(cp.mean(out_grid[cond_msk]))

    # 4. Prepare Stencil for neighbor search
    if stencil is None:
        stencil_res = gsim.neighbors_gpu.make_circle_stencil_gpu_safe(xx[0, :], radius, max_memory_gb)
        stencil = stencil_res if stencil_res[0] is not None else None
        
    return out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil

# Referenced from Matthew's code
# def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, rcond=None, seed=None):

def sgs_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok', 
            sim_mask=None, quiet=False, stencil=None, seed=None, max_memory_gb=150.0,
            batch_size=None, use_sector_balance=True, n_sectors=8, dtype=cp.float64):
    """
    Sequential Gaussian Simulation (SGS) on GPU.
    OPTIMIZED VERSION.
    
    The key difference from Kriging is that simulated values are added to the 
    conditioning data ('cond_msk') instantly, affecting subsequent points.
    """

    assert isinstance(xx, cp.ndarray), 'Error: xx is not a Cupy Array'
    assert isinstance(yy, cp.ndarray), 'Error: yy is not a Cupy Array'
    assert isinstance(grid, cp.ndarray), 'Error: grid is not a Cupy Array'
    gsim.interpolate_gpu._sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)
    #out_grid, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)

    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess_gpu_safe(
        xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb, dtype=dtype
    )
    
    rng = cp.random.default_rng(seed)

    shuffled = cp.copy(inds)
    cp.random.shuffle(shuffled)

    xx_rows, xx_cols = cp.arange(xx.shape[0]) , cp.arange(xx.shape[1])
    ii, jj = cp.meshgrid(xx_rows, xx_cols, indexing = 'ij')

    ## Batch size optimization | GPU limited VRAM
    if batch_size is None:
        # Approximate VRAM usage per point (bytes):
        # 1. Neighbor Search: Stencil size * bytes (distances)
        # 2. Kriging Matrix: (K+1)^2 * bytes
        # 3. Intermediate tensors (coords, etc): K * 4*bytes
        
        bytes_per = 4 if dtype == cp.float64 else 8 # bytes
        
        # Estimate stencil size (area of circle in pixels)
        dx = float(xx[0,1] - xx[0,0])
        stencil_pixels = int(3.14159 * (radius/dx)**2)
        
        mem_per_point = (stencil_pixels * bytes_per) + (num_points**2 * bytes_per) + (num_points * 128)
        
        # Use 80% of available memory safe margin
        avail_mem = max_memory_gb * (1024**3) * 0.8
        calc_batch = int(avail_mem // mem_per_point)
        
        # Clamp between reasonable limits (e.g., 4k to 200k)
        batch_size = max(4096, min(calc_batch, 200000))
        
        if not quiet:
            print(f"Calculated optimal batch size: {batch_size} (Stencil ~{stencil_pixels} px)")
    
    n_points = shuffled.shape[0]
    pbar = tqdm(total = n_points, desc = "Sequential Gaussian Simulation", unit = "pts")

    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, bstart + batch_size)
        batch_inds = shuffled[bstart:bend].astype(cp.int32)

        # 1. Find neighbors
        neighbors, nb_counts = gsim.interpolate_gpu.batch_neighbors_distance_based(
            batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb, dtype=dtype
        )

        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], 
                               yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)

        valid_mask = nb_counts > 0
        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue
        
        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        # 2. Kriging
        if ktype == 'ok':
            ests, vars_ = gsim.interpolate_gpu.batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario, dtype=dtype)
        else:
            ests, vars_ = gsim.interpolate_gpu.batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean, dtype=dtype)

        # 3. Sample from Local Condition Distribution
        vars_safe = cp.abs(vars_)
        std_norm = rng.standard_normal(size=ests.shape, dtype=dtype)
        samp = std_norm * cp.sqrt(vars_safe) + ests

        # 4. Update Grid and Mask

        valid_rows, valid_cols = batch_inds[valid_mask, 0], batch_inds[valid_mask, 1]
        out_grid[valid_rows, valid_cols] = samp

        # Update cond mask
        cond_msk[valid_rows, valid_cols] = True
        pbar.update(bend - bstart)
    
    pbar.close()
    
    out_grid = cp.nan_to_num(out_grid, nan=0.0, posinf=5.0, neginf=-5.0)
    out_grid = cp.clip(out_grid, -5.0, 5.0)

    #sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)

    return out_grid


class chain_sgs_gpu(chain):
    """
    Inherit the chain class. Used for creating sequential gaussian simulation blocks-based MCMC chains.
    
    Parameters in addition to chain's parameters:
        do_transform (bool): If true, normalize radar measurements with normal score transformation to generate subglacial topography. If false, directly use Sequential Gaussian Simulation on un-normalized subglacial topography.
        nst_trans (scikit-learn.preprocessing.QuantileTransformer): The normal score transformation for the (detrended/not detrended) subglacial topography.
        trend: (numpy.ndarray): A 2D array representing the trend of the subglacial topography
        detrend_map (bool): If 'True', the subglacial topography will be de-trended using parameter 'trend'. If 'False', the topography will not be de-trended.
        vario_type (string): The type of variogram model used for SGS ('Gaussian', 'Exponential', 'Spherical', or 'Matern').
        vario_param (list): A list of parameters defining the variogram model, set by the `set_variogram` method. [azimuth, nugget, major range, minor range, sill, variogram type, smoothness].
        sgs_param (list): A list of parameters controlling the SGS behavior, set by the `set_sgs_param` method. [number of nearest neighbors, searching radius, randomly drop out conditioning data (True or False), dropout rate]
        block_min_x, block_max_x, block_min_y, block_max_y (int): the minimum and maximum width and height of the update block
    """ 
    def __init__(self, xx, yy, initial_bed, surf, velx, vely, dhdt, smb, 
                 cond_bed, data_mask, grounded_ice_mask, resolution):
        # Let the parent do validation and attribute assignment
        super().__init__(xx, yy, initial_bed, surf, velx, vely, dhdt, smb,
                         cond_bed, data_mask, grounded_ice_mask, resolution)
        
        # Convert all 2D grid arrays to CuPy in place | no copy
        velx[velx < -9000] = np.nan 
        vely[vely < -9000] = np.nan

        self.xx                = cp.asarray(self.xx)
        self.yy                = cp.asarray(self.yy)
        self.initial_bed       = cp.asarray(self.initial_bed)
        self.surf              = cp.asarray(self.surf)
        self.velx              = cp.asarray(self.velx)
        self.vely              = cp.asarray(self.vely)
        self.dhdt              = cp.asarray(self.dhdt)
        self.smb               = cp.asarray(self.smb)
        self.cond_bed          = cp.asarray(self.cond_bed)
        self.data_mask         = cp.asarray(self.data_mask)
        self.grounded_ice_mask = cp.asarray(self.grounded_ice_mask)

        # resolution stays the same (float)
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_in_region(region_mask) and set_update_region(update_in_region)')
        print('please also set up the sgs parameters using set_sgs_param(self, block_size, sgs_param)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')

    def loss(self, massConvResidual, dataDiff):
        loss_mc = cp.nansum(cp.square(massConvResidual[self.mc_region_mask == 1], dtype=cp.float64)) / (2 * self.sigma_mc ** 2)        
        loss_data = 0
        return loss_mc + loss_data, loss_mc, loss_data
        
    def set_normal_transformation(self, nst_trans, do_transform = True):
        """
        Set the normal score transformation object (from scikit-learn package) used to normalize the bed elevation.
        The function has no returns. Its effect can be checked in the object's parameter 'nst_trans'
        
        Args:
            nst_trans (QuantileTransformer): A fitted scikit-learn transformer used to normalize input data.
        
        Note:
            This transformation must be fit beforehand (e.g., via `MCMC.fit_variogram`).
        """
        self.do_transform = do_transform
        if do_transform:
            self.nst_trans = nst_trans
        else:
            self.nst_trans = None
      
    def set_trend(self, trend = None, detrend_map = True):
        """
        Set the long-wavelength trend component of the bed topography.
        Notice that detrend topography means that the SGS simulation will only simulate the short-wavelength topography residuals that is not a part of the trend
        The function has no returns. Its effect can be checked in the object's parameter 'trend' and 'detrend_map'
        
        Args:
            trend (np.ndarray): A 2D array, representing the topographic trend.
            detrend_map (bool): If True, remove trend before transforming the bed elevation and add it back after inverse transform.
        
        Raises:
            ValueError: If detrend_map is True but trend has invalid shape.
        """
        trend = cp.asarray(trend)
        assert isinstance(trend, cp.ndarray), 'Error: trend is not a Cupy Array'

        if detrend_map == True:
            if len(trend)!=len(self.xx) or trend.shape != self.xx.shape:
                raise ValueError('if detrend_map is set to True, then the trend of the topography, which is a 2D numpy array, must be provided')
            else:
                self.trend = trend
        else:
            self.trend = None
        self.detrend_map = detrend_map
    
    def set_variogram(self, vario_type, vario_range, vario_sill, vario_nugget, isotropic = True, vario_smoothness = None, vario_azimuth = None):
        """
        Specify variogram model and its parameters for SGS interpolation.
        The function has no returns. Its effect can be checked in the object's parameter 'vario_type' and 'vario_param'
        
        Args:
            vario_type (str): Variogram model type. One of 'Gaussian', 'Exponential', 'Spherical', 'Matern'.
            vario_range (float or list): Correlation range(s). One value for isotropic; list of two for anisotropic.
            vario_sill (float): Variogram sill (variance).
            vario_nugget (float): Nugget effect.
            isotropic (bool): Whether the variogram is isotropic. Default is True.
            vario_smoothness (float): Smoothness parameter for Matern model (required if `vario_type` is 'Matern').
            vario_azimuth (float): Azimuth angle for anisotropic variograms in degrees. Units is degrees (360 maximum)
        
        Raises:
            ValueError: If required parameters are missing or in the wrong format.
        """
        # Cast variogram parameters to standard floats
        vario_sill = float(vario_sill)
        vario_nugget = float(vario_nugget)
        vario_range = float(vario_range)
  #      vario_azimuth = float(vario_azimuth) 
 #       vario_smoothness = float(vario_smoothness) if not None
        
        if vario_smoothness is not None:
            vario_smoothness = float(vario_smoothness)
        
        if (vario_type == 'Gaussian') or (vario_type == 'Exponential') or (vario_type == 'Spherical'):
            print('the variogram is set to type', vario_type)
        elif vario_type == 'Matern':
            if (vario_smoothness == None) or (vario_smoothness <= 0):
                raise ValueError('vario_smoothness argument should be a positive float when the vario_type is Matern')
            else:
                print('the variogram is set to type', vario_type)
        else:
            raise ValueError('vario_type argument should be one of the following: Gaussian, Exponential, Spherical, or Matern')
        
        
        self.vario_type = vario_type
        
        if isotropic:
            vario_azimuth = 0.0
            vario_range = float(vario_range)
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type, vario_smoothness]
        else:
            if (isinstance(vario_range, (list, tuple, np.ndarray)) and len(vario_range) == 2):
                # Ensure both ranges are native floats
                r_major = float(vario_range[0])
                r_minor = float(vario_range[1])
                vario_azimuth = float(vario_azimuth) if vario_azimuth is not None else 0.0
                
                print('set to anisotropic variogram with major range and minor range to be ', [r_major, r_minor])
                self.vario_param = [vario_azimuth, vario_nugget, r_major, r_minor, vario_sill, vario_type, vario_smoothness]
            else:
                raise ValueError("vario_range needs to be a list with two floats specifying major and minor range when isotropic is False")
    
    def set_sgs_param(self, sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on = False, dropout_rate = 0):
        """
        Set parameters for Sequential Gaussian Simulation (SGS). Details please see implementation of SGS in GStatSim
        The function has no returns. Its effect can be checked in the object's parameter 'sgs_param
        
        Args:
            sgs_num_nearest_neighbors (int): Number of nearest neighbors used in simulation.
            sgs_searching_radius (float): Radius (in meters) to search for neighbors.
            sgs_rand_dropout_on (bool): Whether to randomly drop conditioning points in simulation block.
            dropout_rate (float): Proportion of conditioning data to drop if dropout is enabled (between 0 and 1).
        """
        
        if sgs_rand_dropout_on == False:
            dropout_rate = 0
            print('because the sgs_rand_dropout_on is set to False, the dropout_rate is automatically set to 0')
            
        self.sgs_param = [sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on, dropout_rate]
    
    def set_block_sizes(self, block_min_x, block_max_x, block_min_y, block_max_y):
        """
        Set minimum and maximum block sizes (in grid cells) for SGS updates.
        The function has no returns. Its effect can be checked in the object's parameter 'block_min_x', 'block_max_x', 'block_min_y', 'block_max_y'
        
        Args:
            block_min_x (int): Minimum width of block in x-direction. Unit in grid cells
            block_max_x (int): Maximum width of block in x-direction.
            block_min_y (int): Minimum height of block in y-direction.
            block_max_y (int): Maximum height of block in y-direction.
        """
        self.block_min_x = block_min_x
        self.block_min_y = block_min_y
        self.block_max_x = block_max_x
        self.block_max_y = block_max_y
    
    def set_random_generator(self, rng_seed = None):
        """
        Set the random generator for the chain to maintain replicability
        Notice that once set_random_generator is called, the random generator for all following call of "run()" will use the continuous sequence of randomness defined by the random generator here. In other words, no other object of random generator will be created unless the set_random_generator() is called again.
        Default: to run a complete chain, call set_random_generator only once before the chain started.

        Args:
            rng_seed (None, int, or numpy.random._generator.Generator): set the random number generator for the chain, either by assigning a random default rng (rng_seed = None), a random generator with seed of rng_seed (rng_seed has type int), or a random generator defined by rng_seed (rng_seed is a generator itself).
        """
        if rng_seed is None:
            rng = cp.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = cp.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, cp.random.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a CuPy random Generator, or None')
            
        self.rng = rng

    def ensure_cupy_arrays_and_validate(self):
        """Ensures all array data is on the GPU and validates scalar parameters."""
    
        # 1. attribute arrays
        # cp.asarray does nothing (zero overhead).
        array_attrs = [
            'xx', 'yy', 'initial_bed', 'surf', 'velx', 'vely', 'dhdt',
            'smb', 'cond_bed', 'data_mask', 'grounded_ice_mask', 
            'region_mask', 'mc_region_mask'
        ]

        for attr in array_attrs:
            val = getattr(self, attr, None)
            if val is None:
                raise ValueError(f"Error: {attr} is missing entirely.")
            # Safely convert to CuPy array (handles NumPy arrays, lists, etc.)
            setattr(self, attr, cp.asarray(val))

        # 2. Conditional arrays
        if self.detrend_map:
            assert hasattr(self, 'trend'), 'Error: trend is not set. Did you call set_trend()?'
            self.trend = cp.asarray(self.trend)

        if self.sample_loc is not None:
            self.sample_loc = cp.asarray(self.sample_loc)

        # 3. scalar / non-scalar params check
        assert isinstance(self.resolution, (int, float)),      'Error: resolution is not a scalar'

        assert hasattr(self, 'sigma_mc'),                      'Error: sigma_mc is not set. Did you call set_loss_type()?'
        assert isinstance(self.sigma_mc, (int, float)),        'Error: sigma_mc is not a scalar'
        assert self.sigma_mc > 0,                              'Error: sigma_mc must be positive'

        assert hasattr(self, 'vario_param'),                   'Error: vario_param is not set. Did you call set_variogram()?'
        assert hasattr(self, 'sgs_param'),                     'Error: sgs_param is not set. Did you call set_sgs_param()?'
        assert hasattr(self, 'block_min_x'),                   'Error: block sizes not set. Did you call set_block_sizes()?'

        # 4. rng generator check
        assert hasattr(self, 'rng'),                           'Error: rng is not set. Did you call set_random_generator()?'

    def run(self, n_iter, only_save_last_bed=False, info_per_iter=100, plot=True, progress_bar=True):
        """
        Run the MCMC chain using block-based SGS updates, with the new gstatsim_custom code
        
        Args:
            n_iter (int): Number of iterations in the MCMC chain.
            only_save_last_bed: If true, the function will only return one subglacial topography at the end of iterations. If false, the function will return all subglacial topography in every iteration.
            info_per_iter (int): for every this number of iterations, the information regarding current loss values and acceptance rate will be printed out.
        
        Returns:
            bed_cache (np.ndarray): A 3D array showing subglacial topography at each iteration, or only the last topography.
            loss_mc_cache (np.ndarray): A 1D array of mass conservation loss at each iteration. If the mass conservation loss is not used, return array of 0
            loss_data_cache (np.ndarray): A 1D array of data misfit loss at each iteration. If the data misfit loss is not used, return array of 0
            loss_cache (np.ndarray): A 1D array of total loss at each iteration.
            step_cache (np.ndarray): A 1D array of boolean indicating if the step was accepted.
            resampled_times (np.ndarray): A 2D array of number of times each pixel was updated.
            blocks_cache (np.ndarray): A 1D array of info on block proposals at each iteration, (x coordinate for the center of the block, y coordinate for the center of the block, block size in x-direction, block size in y-direction).
        """
            
        rng = self.rng
        self.ensure_cupy_arrays_and_validate()
        
        rows = self.xx.shape[0]
        cols = self.xx.shape[1]
        
        loss_cache      = cp.zeros(n_iter)
        loss_mc_cache   = cp.zeros(n_iter)
        loss_data_cache = cp.zeros(n_iter)
        step_cache      = cp.zeros(n_iter)

        if not only_save_last_bed:
            bed_cache   = cp.zeros((n_iter, rows, cols))

        blocks_cache    = cp.full((n_iter, 4), cp.nan)
        resampled_times = cp.zeros(self.xx.shape)
        
        # if the user request to return bed elevation in some sampling locations
        if not (self.sample_loc is None):
            sample_values = cp.zeros((self.sample_loc.shape[0], n_iter))
            
            # convert sample_loc from x and y locations to i and j indexes
            sample_loc_ij = cp.zeros(self.sample_loc.shape, dtype=cp.int16)
            for k in range(self.sample_loc.shape[0]):
                sample_i,sample_j = cp.where((self.xx == self.sample_loc[k,0]) & (self.yy == self.sample_loc[k,1]))
                sample_loc_ij[k,:] = [int(sample_i[0]), int(sample_j[0])]
                
            sample_values[:,0] = self.initial_bed[sample_loc_ij[:,0],sample_loc_ij[:,1]]

        if self.detrend_map:
            bed_c = (self.initial_bed - self.trend).copy()
            cond_bed_c = (self.cond_bed - self.trend).copy()
        else:
            bed_c = self.initial_bed.copy()
            cond_bed_c = self.cond_bed.copy()
       
        
        # NORMAL SPACE INITIAL
        if self.do_transform:
            nst_trans  = self.nst_trans
            # Create an array of the bed entirely in Normal Space (z-scores)
            z_bed_c    = nst_trans.transform(bed_c.reshape(-1,1)).reshape(self.xx.shape)
            z_cond_bed = nst_trans.transform(cond_bed_c.reshape(-1,1)).reshape(self.xx.shape)
        else:
            # If no transform is used, normal space is just physical space
            z_bed_c    = bed_c.copy()
            z_cond_bed = cond_bed_c.copy()
    
        z_cond_bed = z_cond_bed.reshape(self.xx.shape)

        resolution = self.resolution
        
        # INIT LOSS
        if self.detrend_map == True:
            mc_res = MC_res.get_mass_conservation_residual_fused(bed_c + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        else:
            mc_res = MC_res.get_mass_conservation_residual_fused(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        """
        print(f"mc_res stats: min={float(cp.nanmin(mc_res)):.3e}, "
          f"max={float(cp.nanmax(mc_res)):.3e}, "
          f"has_inf={bool(cp.any(cp.isinf(mc_res)))}, "
          f"has_nan={bool(cp.any(cp.isnan(mc_res)))}")
        print(f"mc_res^2 sum = {float(cp.nansum(cp.square(mc_res[self.mc_region_mask == 1]))):.3e}")
        print(f"sigma_mc = {self.sigma_mc}")
        """
        
        data_diff = bed_c - cond_bed_c
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)
        # sync to floats instead of CuPy floats
        loss_prev = float(loss_prev)       
        loss_prev_mc = float(loss_prev_mc)
        loss_prev_data = float(loss_prev_data)

        loss_cache[0] = loss_prev
        loss_mc_cache[0] = loss_prev_mc
        loss_data_cache[0] = loss_prev_data
        step_cache[0] = False
        sim_mask = cp.full(self.xx.shape, False)

        
        if not only_save_last_bed:
            bed_cache[0] = bed_c
    
        rad = self.sgs_param[1]
        neighbors = self.sgs_param[0]
        
        if self.vario_param[5] == 'Matern':
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
                's' : self.vario_param[6]
            }
        else:
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
            }
            
        # SGS MCMC
        sgs_ctx = sgs_cxt.SGS_MCMC(
            self.xx, 
            self.yy, 
            vario,
            radius = rad, 
            num_points = neighbors,
            ktype=  'ok', 
            seed = rng,
            max_memory_gb = 1500.0,
            dtype = cp.float64, 
            quiet = False,
            sigma = self.sigma_mc
        )

        # plotting for real-time result update
        if plot:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12,5))
            (line_loss,) = ax_loss.plot([], [], color='tab:blue', label='Loss')
            (line_acc,)  = ax_acc.plot([], [], color='tab:green', label='Acceptance Rate')
            #NOTE use get_mass_conservation_residual on BedMachine data
            # bm_loss = 
            # ax_loss.axhline(bm_loss, ls='--', label='BedMachine loss') 
            
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("MCMC Loss")

            ax_acc.set_xlabel("Iteration")
            ax_acc.set_ylabel("Acceptance Rate (%)")
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title("MCMC Acceptance Rate")

            ax_loss.legend()
            ax_acc.legend()
            
            display_handle = display.display(fig, display_id=True)
            plt.tight_layout()

        # Track acceptance rate
        accepted_count = 0
        acceptance_rates = []

        if progress_bar == True:
            chain_id = getattr(self, 'chain_id', 0)
            seed = getattr(self, 'seed', 'Unknown')
            tqdm_position = getattr(self, 'tqdm_position', 0)

            iterator = tqdm(range(1, n_iter),
                            desc=f'Chain {chain_id} | Seed {seed}',
                            position=tqdm_position,
                            leave=True)
        else:
            iterator = range(1, n_iter)
            chain_id = getattr(self, 'chain_id', 'Unknown')
            output_line = getattr(self, 'tqdm_position', 0) + 2 
            seed = getattr(self, 'seed', 'Unknown')

        iter_start_time = time.time()   
        
        # Transform once
        #bed_tosim = nst_trans.transform(bed_c.reshape(-1,1)).reshape(self.xx.shape)
        
        valid_indices = cp.argwhere(self.region_mask == 1)  # (N, 2) on GPU
        n_valid = valid_indices.shape[0]

        # Pre-generate enough random indices for all iterations
        """
        print("HERE", n_valid)
        print(type(n_valid))
        """
        rand_idx = rng.integers(0, int(n_valid), size=n_iter)
        rand_block_x = rng.integers(self.block_min_x, self.block_max_x, size=n_iter)
        rand_block_y = rng.integers(self.block_min_y, self.block_max_y, size=n_iter)

        # Pull to CPU once
        rand_idx_cpu = rand_idx.get()
        valid_indices_cpu = valid_indices.get()
        rand_bx_cpu = rand_block_x.get()
        rand_by_cpu = rand_block_y.get()

        for i in iterator:
            
            idx = valid_indices_cpu[rand_idx_cpu[i]]
            indexx, indexy = int(idx[0]), int(idx[1])
            block_size_x = int(rand_bx_cpu[i])
            block_size_y = int(rand_by_cpu[i])
            
            blocks_cache[i,:] = cp.array([indexx,indexy,block_size_x,block_size_y])
    
            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = max( (0,int(indexx-block_size_x/2)) )
            bxmax = min((bed_c.shape[0],int(indexx+block_size_x/2)))
            bymin = max((0,int(indexy-block_size_y/2)))
            bymax = min((bed_c.shape[1],int(indexy+block_size_y/2)))
            
            # Backup normal space block (if rejected we revert to this)
            z_bed_backup = z_bed_c[bxmin:bxmax, bymin:bymax].copy()
            bed_block_backup = bed_c[bxmin:bxmax, bymin:bymax].copy()

            
            # For proposal, overlay the fixed conditioning data onto the normal space grid
            z_bed_c[bxmin:bxmax,bymin:bymax] = z_cond_bed[bxmin:bxmax,bymin:bymax]
            
            # Boolean mask for blocks to change
            sim_mask[:] = False                              
            sim_mask[bxmin:bxmax, bymin:bymax] = True

            ######
            # Pass normal space to SGS
        
            newsim_z = sgs_ctx.simulate(z_bed_c, sim_mask)
            if self.do_transform:
                bed_c[bxmin:bxmax, bymin:bymax] = nst_trans.inverse_transform(
                    newsim_z[bxmin:bxmax, bymin:bymax].reshape(-1, 1)
                ).reshape(bxmax - bxmin, bymax - bymin)
            else:
                bed_c[bxmin:bxmax, bymin:bymax] = newsim_z[bxmin:bxmax, bymin:bymax]
                
            # GET MCR on the pertubated blocks only
            
            pxmin = max(0, bxmin - 1)
            pxmax = min(bed_c.shape[0], bxmax + 1)
            pymin = max(0, bymin - 1)
            pymax = min(bed_c.shape[1], bymax + 1)

            local_surf = self.surf[pxmin:pxmax, pymin:pymax]
            local_velx = self.velx[pxmin:pxmax, pymin:pymax]
            local_vely = self.vely[pxmin:pxmax, pymin:pymax]
            local_dhdt = self.dhdt[pxmin:pxmax, pymin:pymax]
            local_smb  = self.smb[pxmin:pxmax, pymin:pymax]

            
            if self.detrend_map == True:
                new_local_bed = bed_c[pxmin:pxmax, pymin:pymax] + self.trend[pxmin:pxmax, pymin:pymax]
            else:
                new_local_bed = bed_c[pxmin:pxmax, pymin:pymax]
            local_mc_res_pad = MC_res.get_mass_conservation_residual_fused_local(
                new_local_bed, local_surf, local_velx, local_vely, local_dhdt, local_smb, resolution
            )
            dx_min = bxmin - pxmin
            dx_max = dx_min + (bxmax - bxmin)
            dy_min = bymin - pymin
            dy_max = dy_min + (bymax - bymin)

            local_mc_res = local_mc_res_pad[dx_min:dx_max, dy_min:dy_max]

            # LOCAL LOSSES
            mc_res_block_backup = mc_res[bxmin:bxmax, bymin:bymax].copy()
            
            mc_res[bxmin:bxmax, bymin:bymax] = local_mc_res

                
            #data_diff = bed_next - cond_bed_c
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res, None)


            if self.detrend_map == True:
                block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - (bed_c[bxmin:bxmax,bymin:bymax] + self.trend[bxmin:bxmax,bymin:bymax])
            else:
                block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - bed_c[bxmin:bxmax,bymin:bymax]
                
            block_region_mask = self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]
                
            if cp.sum((block_thickness<=0)&(block_region_mask==1)) > 0:
                loss_next = cp.inf
                
            loss_next_val      = float(loss_next)
            loss_next_mc_val   = float(loss_next_mc)
            loss_next_data_val = float(loss_next_data)

            if not math.isfinite(loss_next_val):
                acceptance_rate = 0.0
            elif loss_prev > loss_next_val:
                acceptance_rate = 1.0
            else:
                acceptance_rate = min(1.0, math.exp(loss_prev - loss_next_val))
    
            u = rng.random()
            # ACCEPTED
            if (u <= acceptance_rate):
                z_bed_c[bxmin:bxmax, bymin:bymax] = newsim_z[bxmin:bxmax, bymin:bymax]
    
                loss_cache[i]       = loss_next_val
                loss_mc_cache[i]    = loss_next_mc_val
                loss_data_cache[i]  = loss_next_data_val
                step_cache[i]       = True
                
                loss_prev       = loss_next_val
                loss_prev_mc    = loss_next_mc_val
                loss_prev_data  = loss_next_data_val
                # Update global mass residual
                
                resampled_times[bxmin:bxmax,bymin:bymax] += 1
                accepted_count += 1
            # REJECTED
            else:
                # Restore
                z_bed_c[bxmin:bxmax, bymin:bymax] = z_bed_backup
                bed_c[bxmin:bxmax, bymin:bymax] = bed_block_backup

                loss_cache[i]       = loss_prev
                loss_mc_cache[i]    = loss_prev_mc
                loss_data_cache[i]  = loss_prev_data
                step_cache[i]       = False
                mc_res[bxmin:bxmax, bymin:bymax] = mc_res_block_backup
    
            if not only_save_last_bed:
                if self.detrend_map == True:
                    bed_cache[i,:,:] = bed_c + self.trend
                else:
                    bed_cache[i,:,:] = bed_c
                    
            if not (self.sample_loc is None):
                sample_values[:,i] = bed_c[sample_loc_ij[:,0],sample_loc_ij[:,1]]

            if progress_bar:
                # Update tqdm progress bar
                # Use float() to safely pull CuPy 0-dim arrays to CPU without formatting errors
                # Use accepted_count instead of cp.sum() to prevent GPU syncing
                iterator.set_postfix({
                    'chain_id'  :   chain_id,
                    'seed'      :   seed,
                    'mc loss'   :   f'{float(loss_mc_cache[i]):.3e}',
                    'data loss' :   f'{float(loss_data_cache[i]):.3e}',
                    'loss'      :   f'{float(loss_cache[i]):.3e}',
                    'acc rate'  :   f'{accepted_count / i:.4f}'  # i is the total steps taken so far
                })
            else:
                if i % info_per_iter == 0 or i == 1 or i == n_iter - 1:
                    move_cursor_to_line(output_line)
                    clear_line()
                    
                    progress = i / (n_iter - 1)
                    progress_pct = progress * 100
                    elapsed = time.time() - iter_start_time
                    iter_per_sec = i / elapsed if elapsed > 0 else 0
                    
                    if iter_per_sec > 0:
                        remaining_iters = n_iter - i
                        eta_seconds = remaining_iters / iter_per_sec
                        eta_hours = int(eta_seconds // 3600)
                        eta_minutes = int((eta_seconds % 3600) // 60)
                        eta_secs = int(eta_seconds % 60)
                        eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                    else:
                        eta_str = "--:--:--"
                    
                    bar_length = 10
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '▍' * (1 if filled_length < bar_length and progress > 0 else 0)
                    bar = bar.ljust(bar_length)
                    
                    print(f'Chain {chain_id} ({str(seed)[:6]}): {progress_pct:3.0f}%|{bar}| ETA: {eta_str} | it/s: {iter_per_sec:6.2f} | n: {n_iter:{len(str(n_iter))}d} | loss: {float(loss_cache[i]):.3e} | acc: {accepted_count / i:.4f}', end='')
                    sys.stdout.flush()

            # Calculate acceptance rate for plot
            total_acceptance = (accepted_count / i) * 100
            acceptance_rates.append(total_acceptance)

            if plot:
                if i < 5000:
                    update_interval = 100
                else:
                    update_interval = info_per_iter

                if i % update_interval == 0:
                    # Update loss line
                    line_loss.set_data(range(i + 1), loss_cache[:i + 1].get())
                    ax_loss.relim()
                    ax_loss.autoscale_view()

                    # Update acceptance rate line
                    line_acc.set_data(range(len(acceptance_rates)), acceptance_rates)
                    ax_acc.set_ylim(0, 100)
                    ax_acc.relim()
                    ax_acc.autoscale_view()

                    display_handle.update(fig)
    
        if self.detrend_map == True:
            last_bed = bed_c + self.trend
        else:
            last_bed = bed_c
    
        if not only_save_last_bed:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return last_bed, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache

