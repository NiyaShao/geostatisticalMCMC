"""
Created on Mond Mar 4

@author: tylerrleee
"""

import cupy as cp
import numbers
from copy import deepcopy
from . import gstatsim_custom_gpu as gsim

class SGS_MCMC:
    """
    Precomputes assets for Small Scale Chain , and seperate SGS work.
    
    Only load and save in memory once to preseve GPU memory, and reduce CPU to GPU sync. 

    - Stencil (circle mask for neighbor search)
    - meshgrids (ii, jj)
    - variogram dict (in _preprocess() we make a deepcopy everytime)
    - batch size
    - RNG
    
    SGS_iter: (neighbor search, kriging, sampling)
    """
    
    def __init__(self, 
                 xx, 
                 yy, 
                 variogram, 
                 radius = 100e3, 
                 num_points = 32,
                 ktype = 'ok',  # Ordinary krigging
                 seed = None, 
                 max_memory_gb = 150.0,
                 batch_size=None, 
                 dtype = cp.float32, 
                 quiet = False, 
                 sigma = 1.5):
        """
        One-time setup.  Call this ONCE before the MCMC loop.
 
        Args:
            xx, yy        : 2D CuPy coordinate grids (unchanged across iterations)
            variogram     : dict with keys azimuth, nugget, major_range, minor_range,
                            sill, vtype, (optionally 's' for Matern)
            radius        : search radius for neighbor lookup
            num_points    : max neighbors per simulation point
            ktype         : 'ok' (ordinary kriging) or 'sk' (simple kriging)
            seed          : CuPy RNG or integer seed — usually pass the chain's rng
            max_memory_gb : memory budget for batch-size auto-tuning  # this really depends on the user's RAM, but we assume 150GB for HPC
            batch_size    : override auto batch size (None = auto)
            dtype         : cp.float32 to mirror sklearn scipy
            quiet         : suppress print output
        """
        
        self.xx = xx
        self.yy = yy
        self.radius = radius
        self.num_points = num_points
        self.ktype = ktype
        self.dtype = dtype
        self.max_memory_gb = max_memory_gb
        self.quiet = quiet
        self.sigma = sigma
        
        # Index meshgrids
        xx_rows = cp.arange(xx.shape[0])
        xx_cols = cp.arange(xx.shape[1])
        self.ii, self.jj = cp.meshgrid(xx_rows, xx_cols, indexing='ij')
        
        # Clean Variogram
        self.vario = deepcopy(variogram)
        for k in self.vario:
            if isinstance(self.vario[k], numbers.Number):
                self.vario[k] = float(self.vario[k])
 
        # Stencil — depends only on xx spacing and radius 
        self.stencil = gsim.neighbors_gpu.make_circle_stencil_gpu_safe(
            xx[0, :], radius, max_memory_gb
        )
        
        # Batch size tuning 
        if batch_size is None:
            bytes_per = 4 if dtype == cp.float32 else 8
            dx = float(xx[0, 1] - xx[0, 0])
            stencil_pixels = int(3.14159 * (radius / dx) ** 2)
            mem_per_point = (
                stencil_pixels * bytes_per
                + num_points ** 2 * bytes_per
                + num_points * 128
            )
            avail_mem = max_memory_gb * (1024 ** 3) * 0.8
            calc_batch = int(avail_mem // mem_per_point)
            self.batch_size = max(4096, min(calc_batch, 200_000))
            if not quiet:
                print(f"[SGSContext] batch_size={self.batch_size}  "
                      f"(stencil ~{stencil_pixels} px)")
        else:
            self.batch_size = batch_size
            
        # Use one RNG 
        # In the past, we create a new RNG
        if isinstance(seed, cp.random.Generator):
            self.rng = seed
            print("CuPy Generator")
        elif seed is not None:
            self.rng = cp.random.default_rng(seed)
            print("Default RNG")
        else:
            self.rng = cp.random.default_rng()
            print("Default RNG")
            
    def simulate(self, grid, sim_mask):
        """
            
        Run SGS only on cells where sim_mask is True
        - we skip making new stencil, meshgrids, variogram 
        - - in the past, this step would sync CPU/GPU, causing overhead for n-iterations
        - Passed a transformed grid
        - Return result in normal space so MCMC can inverse-transform only the block that changed
            
        Args:
        grid     : 2D CuPy array, normal-score-transformed bed.
                       NaN where data is absent, real values at conditioning
                       points.  The block to simulate should already be set
                       to the conditioning values (or NaN for unknown).
        sim_mask : 2D bool CuPy array, True where SGS should simulate.
                       Typically a small rectangular block.
 
        Returns:
            out_grid : 2D CuPy array, same shape as grid.  Simulated values
                       are filled in where sim_mask was True; everywhere else
                       is identical to the input grid.
        """
           
        xx, yy = self.xx, self.yy
        ii, jj = self.ii, self.jj
        #rng = self.rng
        dtype = self.dtype
        vario = self.vario
        num_points = self.num_points
        radius = self.radius
        max_memory_gb = self.max_memory_gb
        batch_size = self.batch_size
        
        out_grid = grid.copy()
        cond_mask = ~cp.isnan(out_grid)
            
        # Simulation path
        inds = cp.stack([
            ii[sim_mask].flatten(),
            jj[sim_mask].flatten()
        ] , axis = 1)
            
        # Global mean (simple kriging)
        cond_vals = out_grid[cond_mask]
        global_mean = float(cp.mean(cond_vals)) if cond_vals.size > 0 else 0.0
            
        # Shuffle path
        if isinstance(self.rng, cp.random.Generator):
            # CuPy Generator lacks .permutation, so we use an argsort shuffle
            weights = self.rng.random(inds.shape[0])
            shuffled = inds[cp.argsort(weights)]
            n_points = shuffled.shape[0] 
        else:
            # Legacy RandomState fallback (this object DOES have permutation)
            shuffled = self.rng.permutation(inds)
            n_points = shuffled.shape[0]
        
        # Batched SGS 
        for bstart in range(0, n_points, batch_size):
            bend = min(n_points, bstart + batch_size)
            batch_inds = shuffled[bstart:bend].astype(cp.int32)
                
            # 1. Find neighbors
            neighbors, nb_counts = gsim.interpolate_gpu.batch_neighbors_distance_based(
                    batch_inds, ii, jj, xx, yy,
                    out_grid, cond_mask, radius, num_points,
                    max_memory_gb, dtype=dtype
            )
                
            sim_points = cp.stack([
                    xx[batch_inds[:, 0], batch_inds[:, 1]],
                    yy[batch_inds[:, 0], batch_inds[:, 1]]
            ], axis=1)
                
            valid_mask = nb_counts > 0
            if not cp.any(valid_mask):
                continue
                
            sim_points_valid = sim_points[valid_mask]
            neighbors_valid = neighbors[valid_mask]
                
            # Krige
            if self.ktype == 'ok':
                ests, vars_ = gsim.interpolate_gpu.batch_ok_solve_gpu(
                        sim_points_valid, neighbors_valid, vario, dtype=dtype
                    )
            else:
                ests, vars_ = gsim.interpolate_gpu.batch_sk_solve_gpu(
                        sim_points_valid, neighbors_valid, vario,
                        global_mean, dtype=dtype
                )
                
            # Draw local cond dist
            vars_safe = cp.abs(vars_)
            std_norm = self.rng.standard_normal(size=ests.shape, dtype=dtype)
            samp = std_norm * cp.sqrt(vars_safe) + ests
                
            # WRite simulated values to grid & update mask
            valid_rows = batch_inds[valid_mask, 0]
            valid_cols = batch_inds[valid_mask, 1]
            out_grid[valid_rows, valid_cols] = samp
            cond_mask[valid_rows, valid_cols] = True
                
        # Clamp extremes

        # Replace NaN with zero and posinf for large finite numbers
        out_grid = cp.nan_to_num(x = out_grid, 
                                 nan = 0.0, 
                                 posinf = self.sigma, 
                                 neginf= -1 * self.sigma)
        
        #Clips the values of an array to a given interval
        out_grid = cp.clip(a = out_grid, 
                           a_min = -1*self.sigma, 
                           a_max = self.sigma)
                
        return out_grid