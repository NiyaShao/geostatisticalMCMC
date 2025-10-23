#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:57:24 2025

@author: niyashao
"""

###import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import gstools as gstools
import skgstat as skg
from skgstat import models

from . import Topography

def fit_variogram(data, coords, roughness_region_mask, maxlag, n_lags=50, samples=0.6, subsample=100000, data_for_trans = []):
    """
    This function computes an experimental variogram from the input data and fits
    several theoretical models (Gaussian, Exponential, Spherical, Matern) to it.
    By default, the nugget is fixed at 0.

    Args:
        data (np.ndarray): A column vector of shape (N, 1) containing the input
            data for variogram computation. Use `data.reshape((-1, 1))` if the
            input is a 1D array.
        coords (np.ndarray): An array of shape (N, 2) with spatial coordinates
            corresponding to the data. Each row represents a location, with the
            first column as the x-coordinate and the second as the y-coordinate.
        roughness_region_mask (np.ndarray): A binary mask of shape (M, N)
            identifying the region for roughness evaluation (1 for included,
            0 for excluded).
        maxlag (float): The maximum lag distance to consider for the variogram.
        n_lags (int, optional): The number of lag bins for the experimental
            variogram. Defaults to 50.
        samples (float, optional): The proportion of data pairs to use when
            computing the experimental variogram. Defaults to 0.6.
        subsample (int, optional): The number of data points to use for the
            quantile transformation. Defaults to 100,000.
        data_for_trans (np.ndarray, optional): A column vector of data used to
            compute the quantile transformation, if different from `data`.

    Returns:
        tuple: A tuple containing the following:
            nst_trans (sklearn.preprocessing.QuantileTransformer): The fitted
                quantile transformer used for data normalization.
            transformed_data (np.ndarray): The quantile-transformed data used in
                the variogram calculation.
            params (list[dict]): A list of dictionaries, one for each fitted
                model (Gaussian, Exponential, Spherical, Matern). Each dict
                contains the 'range', 'sill', and 'nugget' parameters.
            fig (matplotlib.figure.Figure): A figure object that plots the
                experimental variogram against the fitted theoretical models.
    """
    if len(data_for_trans)==0:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=0,subsample=subsample).fit(data)
    else:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=0,subsample=subsample).fit(data_for_trans)
        
    transformed_data = nst_trans.transform(data)
    
    coords = coords[roughness_region_mask==1]
    values = transformed_data[roughness_region_mask==1].flatten()
    
    test1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                    maxlag=maxlag, normalize=False, model='gaussian',samples=samples)
    test2 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='exponential',samples=samples)
    test3 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='spherical',samples=samples)
    test4 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='matern',samples=samples)

    tp1 = test1.parameters
    tp2 = test2.parameters
    tp3 = test3.parameters
    tp4 = test4.parameters

    print('range, sill, and nugget for gaussian variogram is ', tp1)
    print('for exponential variogram is ', tp2)
    print('for spherical variogram is ', tp3)
    print('range, sill, smoothness, and nugget for for matern variogram is ', tp4)
    
    # extract experimental variogram values
    xdata1 = test1.bins
    ydata1 = test1.experimental
    xdata2 = test2.bins
    ydata2 = test2.experimental
    xdata3 = test3.bins
    ydata3 = test3.experimental
    xdata4 = test4.bins
    ydata4 = test4.experimental

    # evaluate models
    xi = np.linspace(0, xdata2[-1], n_lags) 
    y_gauss = [models.gaussian(h, tp1[0], tp1[1], tp1[2]) for h in xi]
    y_exp = [models.exponential(h, tp2[0], tp2[1], tp2[2]) for h in xi]
    y_sph = [models.spherical(h, tp3[0], tp3[1], tp3[2]) for h in xi]
    y_mtn = [models.matern(h, tp4[0], tp4[1], tp4[2], tp4[3]) for h in xi] # variogram parameter is [range, sill, shape, nugget] for matern model.

    # plot variogram model
    fig = plt.figure(figsize=(6,4))
    plt.plot(xi, y_gauss,'b--', label='Gaussian variogram model')
    plt.plot(xi, y_exp,'b-', label='Exponential variogram model')
    plt.plot(xi, y_sph,'b*-', label='Spherical variogram model')
    plt.plot(xi, y_mtn,'b-.', label='Matern variogram model')
    plt.plot(xdata1, ydata1,'o', markersize=4, color='green', label='Experimental variogram gaussian',alpha=0.4)
    plt.plot(xdata2, ydata2,'o', markersize=4, color='orange', label='Experimental variogram exponential',alpha = 0.4)
    plt.plot(xdata3, ydata3,'o', markersize=4, color='pink', label='Experimental variogram spherical', alpha = 0.4)
    plt.plot(xdata4, ydata4,'o', markersize=4, color='burlywood', label='Experimental variogram matern', alpha = 0.4)
    plt.title('Variogram for synthetic data')
    plt.xlabel('Lag [m]'); plt.ylabel('Semivariance')  
    plt.legend(loc='lower right', fontsize=8)
    
    return nst_trans, transformed_data, [tp1, tp2, tp3, tp4], fig

class RandField:
    """Generates 2D random fields based on specified variogram models.

    This class creates random fields with defined spatial statistics. It can produce both unconditional fields and conditional fields, where conditioning is achieved using a weighting scheme based on the distance to known data points.

    Before generating fields, the `set_block_sizes` and `set_weight_param` methods must be called to initialize the simulation and conditioning parameters.

    Attributes:
        range_min_x (float): Minimum spatial correlation range in x-direction. The range in x-direction for each realization is randomly sampled between `range_min_x` and `range_max_x`. Similarly, the range in y-direction is randomly sampled between `range_min_y` and `range_max_y`.
        range_max_x (float): Maximum spatial correlation range in x-direction.
        range_min_y (float): Minimum spatial correlation range in y-direction.
        range_max_y (float): Maximum spatial correlation range in y-direction.
        scale_min (float): Minimum vertical scaling (standard deviation).
        scale_max (float): Maximum vertical scaling (standard deviation).
        nugget_max (float): Maximum nugget effect.
        model_name (str): The variogram model type ('Gaussian', 'Exponential', or 'Matern').
        isotropic (bool): Flag for isotropic (True) or anisotropic (False) fields.
        smoothness (float): Smoothness parameter for the Matern model.
        rng_seed (int): Seed for the random number generator.
        rng (np.random.Generator): The random number generator instance.
        pairs (list[tuple]): List of (block_width, block_height) tuples generated by `set_block_sizes`.
        logistic_param (list[float]): Parameters [L, x0, k, offset] for the logistic weighting function.
        max_dist (float): Maximum distance used for scaling the logistic weighting mask.
        resolution (float): Spatial resolution of the grid for conditioning.
        edge_masks (np.ndarray): Precomputed masks for applying conditioning weights at edges.
    """
    
    def __init_func(self):
        print("Before using the `RandField` object in an MCMC chain or for field generation, call method `set_block_sizes` and method`set_weight_param` to initialize block size ranges and conditional weighting parameters.")

    def __init__(self,range_min_x,range_max_x,range_min_y,range_max_y,scale_min,scale_max,nugget_max,model_name,isotropic,smoothness = None, rng_seed=None):
        """Initializes the RandField object.

        This method sets up the parameters for generating random fields based on specified spatial correlation ranges, variogram model type, and scaling. Anisotropy and nugget effects can be included. 
        A NumPy random number generator is initialized using `np.random.default_rng()`, and seeds to this generator is optional. 
        The maximum range values must be greater than or equal to the corresponding minimums, and a `smoothness` value is required if `model_name` is 'Matern'.

        Args:
            range_min_x (float): Minimum spatial correlation range in the x-direction. The actual range is randomly sampled between `range_min_x` and `range_max_x`.
            range_max_x (float): Maximum spatial correlation range in the x-direction.
            range_min_y (float): Minimum spatial correlation range in the y-direction. The actual range is randomly sampled between `range_min_y` and `range_max_y`.
            range_max_y (float): Maximum spatial correlation range in the y-direction.
            scale_min (float): Minimum vertical scaling (standard deviation) of the random field.
            scale_max (float): Maximum vertical scaling (standard deviation) of the random field.
            nugget_max (float): Maximum nugget effect in the variogram.
            model_name (str): The variogram model type used for defining spatial covariance ('Gaussian', 'Exponential', or 'Matern').
            isotropic (bool): If True, enforces isotropic spatial correlation. If False, generates fields with random anisotropy, where the strength depends on the ratio between x- and y-direction ranges.
            smoothness (float, optional): The smoothness parameter for the Matern variogram model. Required if `model_name='Matern'`.
            rng_seed (int, optional): The seed for the NumPy random number generator. If None, a random seed is used. If a seed is used, the RandField object will produce a fixed sequence of conditional / unconditional random fields. Pass the RandField object to two different MCMC chain will not "re-initiate" the random generator.
        """
            
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
 
        self.rng = rng
            
        if (range_max_x < range_min_x) or (range_max_y < range_min_y):
            print('the maximum range must be greater to equal to the minimum range')
        
        self.range_max_x = range_max_x
        self.range_max_y = range_max_y
        self.range_min_x = range_min_x
        self.range_min_y = range_min_y
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.nugget_max = nugget_max
        if (model_name != 'Gaussian') and (model_name != 'Exponential') and (model_name != 'Matern'):
            raise Exception('please put in a valid model_name, including Gaussian, Exponential, and Matern')
        if (model_name == 'Matern') and (smoothness == None):
            raise Exception('a smoothness value must be defined if model name is Matern')
        self.smoothness = smoothness
        self.model_name = model_name
        self.isotropic = isotropic
        
        self.__init_func()
           
    def set_block_sizes(self,min_block_x,max_block_x,min_block_y,max_block_y,steps=5):
        """
        Defines the minimum and maximum block dimensions in x and y directions and determines the number of discrete block sizes between them. This method does not return a value; the resulting block size pairs are stored in the ``pairs`` attribute.
        
        Args:
            min_block_x (int): The minimum block width.
            max_block_x (int): The maximum block width.
            min_block_y (int): The minimum block height.
            max_block_y (int): The maximum block height.
            steps (int, optional): The number of size intervals between the minimum and maximum values. Defaults to 5.
        """
        
        self.min_block_x = min_block_x
        self.min_block_y = min_block_y
        self.max_block_x = max_block_x
        self.max_block_y = max_block_y
        self.steps = steps

        self.pairs = self.get_block_sizes()
    
    def set_weight_param(self, logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution):
        """
        Set logistic function parameters for computing conditioning weights.
    
        The logistic function is defined as ``f(x) = (L / (1 + exp(-k(x - x0)))) - offset``, where ``x`` is the distance to the nearest conditioning data point. The resulting weight equals 1 at conditioning data locations and decays logistically with distance 'x'.
    
        Args:
            logis_func_L (float): The upper limit (L) of the logistic function.
            logis_func_x0 (float): The distance value (x0) where the function reaches its midpoint.
            logis_func_k (float): The growth rate (k) of the logistic curve.
            logis_func_offset (float): A constant offset subtracted from the logistic function's output.
            max_dist (float): The maximum distance used for scaling the weighting mask.
            resolution (float): The spatial resolution of the grid.
        """
        
        if not hasattr(self, 'pairs'):
            raise Exception('It seems like the set_block_sizes has not been called yet before calling set_weight_param')
        
        self.logistic_param = [logis_func_L, logis_func_x0, logis_func_k, logis_func_offset]
        self.max_dist = max_dist
        self.resolution = resolution
        self.edge_masks = self.get_edge_masks()
        
    
    def get_block_sizes(self):
        """
        Internal function. Calculate the discrete block sizes from set_block_size()

        Returns:
            pairs (np.ndarray) : An array of shape (2, N) containing the discrete block sizes. The first row holds the widths and the second row holds the heights.
        """
        
        width = np.linspace(self.min_block_x,self.max_block_x,self.steps,dtype=int)
        height = np.linspace(self.min_block_y,self.max_block_y,self.steps,dtype=int)
        w,h = np.meshgrid(width,height)
        pairs = np.array([(w//2*2).flatten(),(h//2*2).flatten()])
        
        return pairs
    
    def get_edge_masks(self):
        """
        Generate block-edge conditioning masks for each block size.
        For each block defined in ``pairs``, this method computes a distance-based logistic weighting mask along the block edges. The resulting masks are used for conditioning random fields near block boundaries.
        
        Returns:
            edge_masks (list : List of 2D arrays representing edge conditioning masks for each block size.
        """
        
        if not hasattr(self, 'pairs'):
            raise Exception('It seems like the set_block_sizes has not been called yet before calling get_edge_mask')
        
        edge_masks = []
        pairs = self.pairs
        res = self.resolution

        for i in range(pairs.shape[1]):
            bwidth = pairs[:,i][0]
            bheight = pairs[:,i][1]
            
            xx,yy=np.meshgrid(range(bwidth),range(bheight))
            xx = xx*res 
            yy = yy*res
            
            # make a mask of the block boundaries
            cond_msk_edge = np.zeros((bheight,bwidth))
            cond_msk_edge[0,:]=1
            cond_msk_edge[bheight-1,:]=1
            cond_msk_edge[:,0]=1
            cond_msk_edge[:,bwidth-1]=1
            
            # calculate the distance to block boundaries
            dist_edge = RandField.min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
            # re-scale the distance by the maximum correlation distance
            dist_rescale_edge = RandField.rescale(dist_edge, self.max_dist)
            # calculate the logistic function
            dist_logi_edge = RandField.logistic(dist_rescale_edge, self.logistic_param[0], self.logistic_param[1], self.logistic_param[2]) - self.logistic_param[3]
            edge_masks.append(dist_logi_edge)

        return edge_masks
    
    def get_random_field(self,X,Y,n=1):
        """Generates random field realizations.

        This method samples variogram parameters (e.g., range, sill, nugget) from the ranges specified in the object's attributes.
        It then creates one or more spatial random fields on the provided grid using the selected covariance model.

        Args:
            X (np.ndarray): A 1D array of the grid's x-coordinates.
            Y (np.ndarray): A 1D array of the grid's y-coordinates.
            n (int, optional): The number of random field realizations to generate. Defaults to 1.

        Returns:
            np.ndarray: A 3D array of shape (n, len(Y), len(X)) containing the generated random fields.
        """
        
        rng = self.rng
        
        _mean=0
        _var=1
        
        scale  = rng.uniform(low=self.scale_min, high=self.scale_max, size=1)[0]/3
        nug = rng.uniform(low=0.0, high=self.nugget_max, size=1)[0]
        
        if not self.isotropic:
            range1 = rng.uniform(low=self.range_min_x, high=self.range_max_x, size=1)[0]
            range2 = rng.uniform(low=self.range_min_y, high=self.range_max_y, size=1)[0]
            angle = rng.uniform(low=0, high=180, size=1)[0]
        else:
            range1 = rng.uniform(low=self.range_min_x, high=self.range_max_x, size=1)[0]
            range2 = range1
            angle = 0.0
            
        if self.model_name == 'Gaussian':
            model = gstools.Gaussian(dim=2, var = _var,
                                len_scale = [range1/np.sqrt(3),range2/np.sqrt(3)],
                                angles = angle*np.pi/180,
                                nugget = nug)
        elif self.model_name == 'Exponential':
            model = gstools.Exponential(dim=2, var = _var,
                        len_scale = [range1/3,range2/3],
                        angles = angle*np.pi/180,
                        nugget = nug)
        elif self.model_name == 'Matern':
            smoothness = self.smoothness
            model = gstools.Matern(dim=2, var=_var,
                        len_scale = [range1/2,range2/2],
                        angles = angle*np.pi/180,
                        nugget = nug,
                        nu = smoothness)
        else:
            print('error model name')
            return

        fields = np.zeros((n,len(Y),len(X)))
        for i in range(n):
            srf = gstools.SRF(model)
            fields[i,:,:] = srf.structured([X, Y]).T*scale + _mean

        return fields
    
    def min_dist(hard_mat, xx, yy):
        """
        Compute the minimum Euclidean distance to non-NaN points.
        
        Notes: This is an internal helper function and not intended for direct use.
        """
        dist = np.zeros(xx.shape)
        xx_hard = np.where(np.isnan(hard_mat), np.nan, xx)
        yy_hard = np.where(np.isnan(hard_mat), np.nan, yy)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                dist[i,j] = np.nanmin(np.sqrt(np.square(yy[i,j]-yy_hard)+np.square(xx[i,j]-xx_hard)))
        return dist

    def rescale(x, maxdist):
        """
        Rescale distance values by the specified maximum distance.
        
        Notes: This is an internal helper function and not intended for direct use.
        """
        return np.where(x>maxdist,1,(x/maxdist))

    def logistic(x, L, x0, k):
        """
        Evaluate the logistic function with given parameters.
        
        Notes: is an internal helper function and not intended for direct use.
        """
        return L/(1+np.exp(-k*(x-x0)))
    
    def get_crf_weight(self,xx,yy,cond_data_mask):
        """
        This method generates weights for a conditional random field using a mask showing locations of the conditioning data. 

        Args:
            xx (np.ndarray): A 2D array of the grid's x-coordinates.
            yy (np.ndarray): A 2D array of the grid's y-coordinates.
            cond_data_mask (np.ndarray): A 2D array showing locations of conditioning data. (1 = have data, 0 = do not have data)

        Returns:
            weight (np.ndarray): 2D array. The final conditioning weights.
            dist (np.ndarray): 2D array. The minimal distance to the closest conditioning data.
            dist_rescale (np.ndarray): 2D array. The distance array scaled such that `self.max_dist` maps to 1.
            dist_logi (np.ndarray): 2D array. The raw output of the logistic function applied to the rescaled distances.
        """
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        # calculate the distance to block boundaries
        dist = RandField.min_dist(np.where(cond_data_mask==0, np.nan, 1), xx, yy)
        # re-scale the distance by the maximum correlation distance
        dist_rescale = RandField.rescale(dist, max_dist)
        # calculate the logistic function
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi

    def get_crf_weight_from_dist(self,xx,yy,dist):
        """
        This method generates weights for a conditional random field using a distance array that has already been computed. 
        It is useful for large domains where calculating distances on-the-fly is computationally expensive, allowing the distance array to be loaded from a file instead.

        Args:
            xx (np.ndarray): A 2D array of the grid's x-coordinates.
            yy (np.ndarray): A 2D array of the grid's y-coordinates.
            dist (np.ndarray): A 2D array containing the pre-calculated distance to the nearest conditioning point for each grid cell.

        Returns:
            weight (np.ndarray): 2D array. The final conditioning weights.
            dist (np.ndarray): 2D array. The original, unmodified distance array passed to the function.
            dist_rescale (np.ndarray): 2D array. The distance array scaled such that `self.max_dist` maps to 1.
            dist_logi (np.ndarray): 2D array. The raw output of the logistic function applied to the rescaled distances.
        """
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi
    
    def get_rfblock(self):
        """
        Generate a random field block based on the information stored in the RandField object.
        The block size is randomly selected from values defined in set_block_sizes() function. 
        The random field has a logistic decay to the block's edges
        
        Returns:
            f (2D numpy array): a random field sample shaped by logistic mask.
        """
        
        res = self.resolution
        
        # randomly choose a size from the list
        block_size_i = self.rng.integers(low=0, high=self.pairs.shape[1], size=1)[0]
        block_size = self.pairs[:,block_size_i]
        
        #generate field
        x_uniq = np.arange(0,block_size[0]*res,res)
        y_uniq = np.arange(0,block_size[1]*res,res)

        #in-case of a weird bug
        while True:
            ## TODO: have to modify this for n>1
            f = self.get_random_field(x_uniq, y_uniq)
            f = f[0,:,:]
            if (np.sum(np.isnan(f))) != 0:
                print('f have nan')
                continue
            else:
                break
            
        return f*self.edge_masks[block_size_i]
    
class chain:
    """
    A base class for setting up a Markov chain for MCMC topography sampling.

    This class handles the initialization of data and the configuration of the loss function for its child classes (e.g., crf_chain, sgs_chain).
    
    Parameters:
        xx, yy, initial_bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution: As defined in the initialization function
        update_in_region (bool): If 'True', the block update will only happen within the region. If 'False', the block update will happen for the entire map
        region_mask (numpy.ndarray): The 2D mask defining the region where block update occur (1 = inside the region, 0 = outside the region).
        mc_region_mask (numpy.ndarray): The 2D mask defining the region where the mass conservation loss is calculated (1 = inside the region, 0 = outside the region).
        data_region_mask (numpy.ndarray): The 2D mask defining the region where the data misfit loss is calculated (1 = inside the region, 0 or np.nan = outside the region).
        sigma_mc (float): standard deviation for the mass conservation loss, when the 'map_func' is 'sumsquare' (a.k.a. the mass flux residuals have gaussian distributions)
        sigma_mc (float): standard deviation for the data misfit loss, when the 'diff_func' is 'sumsquare' (a.k.a. the data misfits have gaussian distributions)
        map_func (string): A string for choosing the function to calculate for mass conservation loss.
        diff_func (string): A string for choosing the function to calculate for data misfit loss.
        loss_function_list (list): A list of two functions used for mass conservation loss and data misfit loss respective.
    """

    def __init_func__(self):
        """
        A placeholder function for printing out notes and instruction on setting-up the chain

        No returns.
        """
        
        return
    
    def __init__(self, xx, yy, initial_bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution):
        """Initializes the MCMC chain with input data and model geometry.

        Args:
            xx (np.ndarray): A 2D array of grid x-coordinates.
            yy (np.ndarray): A 2D array of grid y-coordinates.
            initial_bed (np.ndarray): The initial bed topography guess.
            surf (np.ndarray): The ice surface elevation.
            velx (np.ndarray): The ice surface velocity in the x-direction.
            vely (np.ndarray): The ice surface velocity in the y-direction.
            dhdt (np.ndarray): The rate of surface height change.
            smb (np.ndarray): The surface mass balance.
            cond_bed (np.ndarray): Conditioning radar bed measurements. Grid cells without data should be marked with NaN.
            data_mask (np.ndarray): A binary mask where 1 indicates the presence of conditioning data.
            grounded_ice_mask (np.ndarray): A binary mask where 1 indicates grounded ice.
            resolution (float): The spatial resolution of the grid in meters.

        Raises:
            Exception: If input arrays do not have matching shapes.
        """
        
        self.xx = xx
        self.yy = yy
        self.initial_bed = initial_bed
        self.surf = surf
        self.velx = velx
        self.vely = vely
        self.dhdt = dhdt
        self.smb = smb
        self.cond_bed = cond_bed
        self.data_mask = data_mask
        self.grounded_ice_mask = grounded_ice_mask
        self.resolution = resolution
        self.loss_function_list = []
        
        if (initial_bed.shape!=surf.shape) or (initial_bed.shape!=velx.shape) or (initial_bed.shape!=vely.shape) or (initial_bed.shape!=dhdt.shape) or (initial_bed.shape!=smb.shape) or (initial_bed.shape!=cond_bed.shape) or (initial_bed.shape!=data_mask.shape):
            raise Exception('the shape of bed, surf, velx, vely, dhdt, smb, radar_bed, data_mask need to be same')
        
        self.__init_func__()
        
    def set_update_region(self, update_in_region, region_mask = []):
        """
        Defines a spatial mask to constrain where topography updates can occur.

        Args:
            update_in_region (bool): If True, updates are restricted to the area defined by `region_mask`.
            region_mask (np.ndarray): A binary mask specifying the update region.

        Raises:
            ValueError: If `region_mask` does not match the shape of the grid in the initialization.
        """
        
        self.update_in_region = update_in_region
        
        if update_in_region == False:
            print('the update blocks is set to be randomly generated for any locations inside the entire map')
            self.region_mask = np.full(self.xx.shape, 1)
        else:
            if region_mask.shape != self.xx.shape:
                raise ValueError('the region_mask input is invalid. It has to be a 2D numpy array with the shape of the map')
                self.region_mask = None
            else:
                print('the update blocks is set to be randomly generated for any locations inside the given region')
                self.region_mask = region_mask

    
    def __meanabs(data,mask):
        """
        A loss function, return the mean of absolute values of data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the mean of absolute data inside the mask
        """
        return np.nanmean(np.abs(data[mask==1]))
    
    def __meansq(data,mask):
        """
        A loss function, return the mean of squared data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the mean of squared data inside the mask
        """
        return np.nanmean(np.square(data[mask==1]))
    
    def __sumabs(data,mask):
        """
        A loss function, return the sum of absolute values of data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the sum of the absolute data value inside the mask
        """
        return np.nansum(np.abs(data[mask==1]))
    
    def __sumsq(data,mask):
        """
        A loss function, return the sum of squared data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the sum of the squared data inside the mask
        """
        return np.nansum(np.square(data[mask==1]))
    
    def __return0(data,mask):
        return 0

    def set_loss_type(self, map_func = None, diff_func = None, sigma_mc = -1, sigma_data = -1, massConvInRegion = True, dataDiffInRegion = False):
        """
        Configure loss function used in MCMC chain
        Currently enable either of losses (or both) to be used: mass flux residuals and misfit to radar measurements.
        The function has no return. Its effect can be checked in chain object's 'loss_function_list' attribute
        
        Args:
            map_func (str, optional): The aggregation function for the mass flux residuals ('meanabs', 'meansquare', 'sumabs', 'sumsquare'). 'sumsquare' corresponds to a Gaussian likelihood. If None, this loss component is ignored.
            diff_func (str, optional): The aggregation function for the radar data misfit ('meanabs', 'meansquare', 'sumabs', 'sumsquare'). 'sumsquare' corresponds to a Gaussian likelihood. If None, this loss component is ignored.
            sigma_mc (float): The standard deviation for the mass flux residuals likelihood. This is required if `map_func` is not None.
            sigma_data (float): The standard deviation for the data misfit likelihood. This is required if `diff_func` is not None.
            massConvInRegion (bool): If True, calculates the mass conservation loss only within the specified `region_mask`.
            dataDiffInRegion (bool): If True, calculates the data misfit loss only within the specified `region_mask`.

        Raises:
            ValueError: If function names are invalid or required `sigma` values are not provided.
        """
    
        function_list = []
        
        if (map_func == None) and (diff_func == None):
            raise ValueError('please specify either one of or both of map_func and diff_func. The chain need at least one loss function. If plan to set up custom function later, please also put in valid values in map_func and/or data_func as a filler')
        if ((map_func != None) and (sigma_mc <= 0)) or ((diff_func != None) and (sigma_data <= 0)):
            raise ValueError('please make sure sigma is correctly set for either sigma_mc and/or sigma_data (sigma >= 0)')
    
        if massConvInRegion:
            self.mc_region_mask = self.region_mask
        else:
            self.mc_region_mask = np.full(self.xx.shape,1)
        
        if dataDiffInRegion:
            self.data_region_mask = self.region_mask
        else:
            self.data_region_mask = np.full(self.xx.shape,1)
            
    
        if map_func == 'meanabs':
            function_list.append(chain.__meanabs)
        elif map_func == 'meansquare':
            function_list.append(chain.__meansq)
        elif map_func == 'sumabs':
            function_list.append(chain.__sumabs)
        elif map_func == 'sumsquare':
            function_list.append(chain.__sumsq)
        elif map_func == None:
            function_list.append(chain.__return0)
        else:
            raise Exception("the map_func argument is not set to correct value.")
            
            
        if diff_func == 'meanabs':
            function_list.append(chain_crf.__meanabs)
        elif diff_func == 'meansquare':
            function_list.append(chain_crf.__meansq)
        elif diff_func == 'sumabs':
            function_list.append(chain_crf.__sumabs)
        elif diff_func == 'sumsquare':
            function_list.append(chain_crf.__sumsq)
        elif diff_func == None:
            function_list.append(chain_crf.__return0)
        else:
            raise Exception("the diff_func argument is not set to correct value.")
      
        self.sigma_mc = sigma_mc
        self.sigma_data = sigma_data
        self.map_func = map_func
        self.diff_func = diff_func
        self.loss_function_list = function_list
    
    def loss(self, massConvResidual, dataDiff):
        """Computes the value of the loss function for a candidate topography.

        Args:
            massConvResidual (np.ndarray): A 2D array of the mass flux residuals field.
            dataDiff (np.ndarray): A 2D array of the difference between the candidate topography and observed bed elevation.

        Returns:
            total_loss (float): The combined, weighted loss value.
            loss_mc (float): The loss component from mass conservation.
            loss_data (float): The loss component from data misfit.
        """
        
        f1 = self.loss_function_list[0]
        f2 = self.loss_function_list[1]
            
        # TODO: is it inappropriate to use sum when the two loss have unequal number of grid cells?
        loss_mc = f1(massConvResidual, self.mc_region_mask) / (2*self.sigma_mc**2)
        loss_data = f2(dataDiff, (self.data_mask==1)&(self.data_region_mask==1)) / (2*self.sigma_data**2)
        
        return loss_mc + loss_data, loss_mc, loss_data
      
class chain_crf(chain):
    """
    Inherit the chain class. Used for creating random field-based MCMC chains. Can choose between conditional or unconditional random fields.
    
    Parameters in addition to chain's parameters:
        block_type (String): Specifying uncondtional or conditional random field, 'CRF_weight' or 'RF'.
        crf_data_weight (numpy.ndarray): A 2D array of the data weight to ensure that the updates will not perturb the conditioning data.
    """ 
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_region(update_in_region, region_mask)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')
        print('an RandField object also need to be created correctly and passed in set_crf_data_weight(RF) and in run(n_iter, RF)')
        return
    
    def set_update_type(self, block_type):
        """
        Set types of the perturbation blocks. 
        For the current version of the algorithm, can choose from unconditional random field ('RF') or conditional random field ('CRF') created by logistic weighting (CRF_weight)
        The function has no returns. Its effect can be checked in the object's parameter 'block_type'
        
        Args:
            block_type (str): 'CRF_weight' or 'RF'.
        
        Raises:
            ValueError: If block_type is invalid.
        """
            
        if block_type == 'CRF_rbf':
            print('The update block is set to conditional random field generated by rbf method (not implemented yet)')
        elif block_type == 'CRF_weight':
            print('The update block is set to conditional random field generated by calculating weights with logistic function')
        elif block_type == 'RF':
            print('The update block is set to Random Field')
        else:
            raise ValueError('The block_type argument should be one of the following: CRF_weight, CRF_rbf, RF')
        
        self.block_type = block_type
            
        return
    
    def set_crf_data_weight(self, RF):
        """
        Calculate and store conditioning random field weights from conditioning data.
        The function has no returns. Its effect can be checked in the object's parameter 'crf_data_weight'
        
        Args:
            RF (RandField): A RandField object with the logistic function parameters configured.
        """
        
        crf_weight, dist, dist_rescale, dist_logi = RF.get_crf_weight(self.xx,self.yy,self.data_mask)
        self.crf_data_weight = crf_weight

    def run(self, n_iter, RF, rng_seed=None, only_save_last_bed=False, info_per_iter = 1000):
        """Runs the MCMC sampling chain to generate topography realizations.

        Args:
            n_iter (int): The total number of MCMC iterations to perform.
            RF (RandField): An initialized `RandField` object used to generate the topography perturbations.
            rng_seed (int, optional): A seed for the random number generator to ensure reproducibility. Defaults to None.
            only_save_last_bed (bool): If True, only the final topography is returned. If False, the topography from every iteration is saved, which requires more memory.
            info_per_iter (int): The iteration interval for printing progress updates, such as loss and acceptance rate, to the console.

        Returns:
            bed_cache (np.ndarray): A 4D array of saved topographies if `only_save_last_bed` is False, or a 2D array of the final topography if True.
            loss_mc_cache (np.ndarray): A 1D array of the mass conservation loss at each iteration. If no mass conservation loss is set, will return zeros.
            loss_data_cache (np.ndarray): A 1D array of the data misfit loss at each iteration. If no data misfit loss is set, will return zeros.
            loss_cache (np.ndarray): A 1D array of the total loss (mass conservation loss + data misfit loss) at each iteration.
            step_cache (np.ndarray): A 1D boolean array indicating whether the proposal was accepted at each iteration.
            resampled_times (np.ndarray): A 2D array counting how many times each grid cell was part of an accepted proposal.
            blocks_cache (np.ndarray): A 2D array logging the location and size `[row, col, height, width]` of the proposed update block at each iteration.
        """
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
            
        if not isinstance(RF, RandField):
            raise TypeError('The arugment "RF" has to be an object of the class RandField')
        
       # initialize storage
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        loss_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        if ~only_save_last_bed:
            bed_cache = np.zeros((n_iter, self.xx.shape[0], self.xx.shape[1]))
        blocks_cache = np.full((n_iter, 4), np.nan)
        resampled_times = np.zeros(self.xx.shape)
        
        bed_c = self.initial_bed

        resolution = self.resolution
        
        # initialize loss
        mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        data_diff = bed_c - self.cond_bed
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_data_cache[0] = loss_prev_data
        loss_mc_cache[0] = loss_prev_mc
        step_cache[0] = False
        if ~only_save_last_bed:
            bed_cache[0] = bed_c
        
        #crf_weight = self.crf_data_weight

        for i in range(1,n_iter):
                        
            #not done yet
            f = RF.get_rfblock()
            block_size = f.shape
            
            # determine the location of the block
            if self.update_in_region:
                while True:
                    indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                    indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                    if self.region_mask[indexx,indexy] == 1:
                        break
            else:
                indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                
            #record block
            blocks_cache[i,:]=[indexx,indexy,block_size[0],block_size[1]]

            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = np.max((0,int(indexx-block_size[0]/2)))
            bxmax = np.min((bed_c.shape[0],int(indexx+block_size[0]/2)))
            bymin = np.max((0,int(indexy-block_size[1]/2)))
            bymax = np.min((bed_c.shape[1],int(indexy+block_size[1]/2)))
            
            #find the index of the block side in the coordinate of the block
            mxmin = np.max([block_size[0]-bxmax,0])
            mxmax = np.min([bed_c.shape[0]-bxmin,block_size[0]])
            mymin = np.max([block_size[1]-bymax,0])
            mymax = np.min([bed_c.shape[1]-bymin,block_size[1]])
            
            #perturb
            if self.block_type == 'CRF_weight':
                perturb = f[mxmin:mxmax,mymin:mymax]*self.crf_data_weight[bxmin:bxmax,bymin:bymax]
            else:
                perturb = f[mxmin:mxmax,mymin:mymax]

            bed_next = bed_c.copy()
            bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb
            
            if self.update_in_region:
                bed_next = np.where(self.region_mask, bed_next, bed_c)
            else:
                bed_next = np.where(self.grounded_ice_mask, bed_next, bed_c)
                
            mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            data_diff = bed_next - self.cond_bed
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
           
            #make sure no bed elevation is greater than surface elevation
            block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - bed_next[bxmin:bxmax,bymin:bymax]
            
            if self.update_in_region:
                block_region_mask = self.region_mask[bxmin:bxmax,bymin:bymax]
            else:
                block_region_mask = self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]
            
            if np.sum((block_thickness<=0)[block_region_mask==1]) > 0:
                loss_next = np.inf

            if loss_prev > loss_next:
                acceptance_rate = 1
            else:
                acceptance_rate = min(1,np.exp(loss_prev-loss_next))
            
            u = rng.random()
            if (u <= acceptance_rate):
                bed_c = bed_next.copy()
                
                loss_prev = loss_next
                loss_prev_mc = loss_next_mc
                loss_cache[i] = loss_next
                loss_mc_cache[i] = loss_next_mc
                loss_prev_data = loss_next_data
                loss_data_cache[i] = loss_next_data
                
                step_cache[i] = True
                if self.update_in_region:
                    resampled_times[bxmin:bxmax,bymin:bymax] += self.region_mask[bxmin:bxmax,bymin:bymax]
                else:
                    resampled_times[bxmin:bxmax,bymin:bymax] += self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]
                
            else:
                loss_mc_cache[i] = loss_prev_mc
                loss_cache[i] = loss_prev
                loss_data_cache[i] = loss_prev_data
                step_cache[i] = False
            
            if ~only_save_last_bed:
                bed_cache[i,:,:] = bed_c

            if i%info_per_iter == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} data loss: {loss_data_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache)/(i+1)}')
                
        if ~only_save_last_bed:
            return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            return bed_c, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache

class chain_sgs(chain):
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
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_in_region(region_mask) and set_update_region(update_in_region)')
        print('please also set up the sgs parameters using set_sgs_param(self, block_size, sgs_param)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')
        
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
        
        if detrend_map == True:
            if len(trend)!=len(self.xx) or trend.shape!=self.xx.shape:
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
            vario_azimuth = 0
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type, vario_smoothness]
        else:
            if (len(vario_range) == 2):
                print('set to anistropic variogram with major range and minor range to be ', vario_range)
                self.vario_param = [vario_azimuth, vario_nugget, vario_range[0], vario_range[1], vario_sill, vario_type, vario_smoothness]
            else:
                raise ValueError ("vario_range need to be a list with two floats to specifying for major range and minor range of the variogram when isotropic is set to False")
    
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

    def run(self, n_iter, rng_seed=None, only_save_last_bed=False, info_per_iter=1000):
        """
        Run the MCMC chain using block-based SGS updates
        
        Args:
            n_iter (int): Number of iterations in the MCMC chain.
            rng_seed (None or string): The seed for the NumPy random number generator. If None, a random seed is used.
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
            
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
        
        xmin = np.min(self.xx)
        xmax = np.max(self.xx)
        ymin = np.min(self.yy)
        ymax = np.max(self.yy)
        
        rows = self.xx.shape[0]
        cols = self.xx.shape[1]
        
        loss_cache = np.zeros(n_iter)
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        if ~only_save_last_bed:
            bed_cache = np.zeros((n_iter, rows, cols))
        blocks_cache = np.full((n_iter, 4), np.nan)
        
        if self.detrend_map:
            bed_c = self.initial_bed - self.trend
            cond_bed_c = self.cond_bed - self.trend
        else:
            bed_c = self.initial_bed
            cond_bed_c = self.cond_bed

        
        if self.do_transform:
            nst_trans = self.nst_trans
            z = nst_trans.transform(bed_c.reshape(-1,1))
            z_cond_bed = nst_trans.transform(cond_bed_c.reshape(-1,1))
        else:
            z = bed_c.reshape(-1,1)
            z_cond_bed = cond_bed_c.reshape(-1,1)
            
        cond_bed_data = np.array([self.xx.flatten(),self.yy.flatten(),z_cond_bed.flatten()])
        cond_bed_df = pd.DataFrame(cond_bed_data.T, columns=['x','y','cond_bed'])
        
        resolution = self.resolution
    
        df_data = np.array([self.xx.flatten(),self.yy.flatten(),z.flatten(),self.data_mask.flatten(),self.mc_region_mask.flatten()])
        psimdf = pd.DataFrame(df_data.T, columns=['x','y','z','data_mask','mc_region_mask'])
        psimdf['resampled_times'] = 0
        
        #psimdf['data_mask'] = data_mask.flatten()
        data_index = psimdf[psimdf['data_mask']==1].index
        
        #psimdf['mc_region_mask'] = mc_region_mask.flatten()
        mask_index = psimdf[psimdf['mc_region_mask']==1].index
        
        # initialize loss
        if self.detrend_map == True:
            mc_res = Topography.get_mass_conservation_residual(bed_c + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        else:
            mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        
        data_diff = bed_c - cond_bed_c
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_mc_cache[0] = loss_prev_mc
        loss_data_cache[0] = loss_prev_data
        step_cache[0] = False
        if ~only_save_last_bed:
            bed_cache[0] = bed_c
        
        for i in range(n_iter):
            
            rsm_center_index = mask_index[rng.integers(low=0, high=len(mask_index))]
            rsm_x_center = psimdf.loc[rsm_center_index,'x']
            rsm_y_center = psimdf.loc[rsm_center_index,'y']

            block_size_x = rng.integers(low=self.block_min_x, high=self.block_max_x, size=1)[0]
            block_size_x = int(block_size_x/2)*self.resolution
            block_size_y = rng.integers(low=self.block_min_y, high=self.block_max_y, size=1)[0]
            block_size_y = int(block_size_y/2)*self.resolution

            blocks_cache[i,:]=[rsm_x_center,rsm_y_center,block_size_x,block_size_y]

            #left corner in terms of meters
            rsm_x_min = np.max([int(rsm_x_center - block_size_x),xmin])
            rsm_x_max = np.min([int(rsm_x_center + block_size_x),xmax])
            rsm_y_min = np.max([int(rsm_y_center - block_size_y),ymin])
            rsm_y_max = np.min([int(rsm_y_center + block_size_y),ymax])

            resampling_box_index = psimdf[(rsm_x_min<=psimdf['x'])&(psimdf['x']<rsm_x_max)&(rsm_y_min<=psimdf['y'])&(psimdf['y']<rsm_y_max)].index
            
            new_df = psimdf.copy() 
            
            # if enable random drop out
            if self.sgs_param[2] == True:
                
                # intersect_index: in_block_cond_data
                intersect_index = resampling_box_index.intersection(data_index)
                intersect_index = rng.choice(intersect_index, size=int(intersect_index.shape[0]*(1-self.sgs_param[3])), replace=False)
                
                if (np.sum(psimdf.loc[intersect_index,['x']].values != cond_bed_df.loc[intersect_index,['x']].values) != 0):
                    print('test of index sameness failed at iter ', i)
                
                if (np.sum(psimdf.loc[intersect_index,['y']].values != cond_bed_df.loc[intersect_index,['y']].values) != 0):
                    print('test of index sameness failed at iter ', i)
                    
                # restore 70% of the conditioning data
                new_df.loc[intersect_index,['z']] = cond_bed_df.loc[intersect_index,['cond_bed']].values
                
                # drop 30% of conditioning data inside the block
                drop_index = resampling_box_index.difference(intersect_index)
                
            else:
                
                drop_index = resampling_box_index.difference(data_index)

            new_df = new_df[~new_df.index.isin(drop_index)].copy()

            Pred_grid_xy_change = gs.Gridding.prediction_grid(rsm_x_min, rsm_x_max - resolution, rsm_y_min, rsm_y_max - resolution, resolution)
            x = np.reshape(Pred_grid_xy_change[:,0], (len(Pred_grid_xy_change[:,0]), 1))
            y = np.flip(np.reshape(Pred_grid_xy_change[:,1], (len(Pred_grid_xy_change[:,1]), 1)))
            Pred_grid_xy_change = np.concatenate((x,y),axis=1)

            if self.vario_param[5] == 'Matern':
                vario_p = self.vario_param
            else:
                vario_p = self.vario_param[:6]
            sim2 = gs.Interpolation.okrige_sgs(Pred_grid_xy_change, new_df, 'x', 'y', 'z', self.sgs_param[0], vario_p, self.sgs_param[1], quiet=True, seed=rng) 

            xy_grid = np.concatenate((Pred_grid_xy_change[:,0].reshape(-1,1),Pred_grid_xy_change[:,1].reshape(-1,1),np.array(sim2).reshape(-1,1)),axis=1)

            psimdf_next = psimdf.copy()
            psimdf_next.loc[resampling_box_index,['x','y','z']] = xy_grid
            if self.do_transform:
                bed_next = nst_trans.inverse_transform(np.array(psimdf_next['z']).reshape(-1,1)).reshape(rows,cols)
            else:
                bed_next = np.array(psimdf_next['z']).reshape(rows,cols)
            
            if self.detrend_map == True:
                mc_res = Topography.get_mass_conservation_residual(bed_next + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            else:
                mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            
            data_diff = bed_next - cond_bed_c
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
            
            #make sure no bed elevation is greater than surface elevation
            if self.detrend_map == True:
                thickness = self.surf - (bed_next + self.trend)
            else:
                thickness = self.surf - bed_next
            
            if np.sum((thickness<=0)[self.grounded_ice_mask==1]) > 0:
                loss_next = np.inf
            
            if loss_prev > loss_next:
                acceptance_rate = 1
                
            else:
                acceptance_rate = min(1,np.exp(loss_prev-loss_next))

            u = rng.random()
            
            if (u <= acceptance_rate):
                bed_c = bed_next
                psimdf = psimdf_next
                loss_cache[i] = loss_next
                loss_mc_cache[i] = loss_next_mc
                loss_data_cache[i] = loss_next_data
                step_cache[i] = True
                psimdf.loc[drop_index, 'resampled_times'] = psimdf.loc[drop_index, 'resampled_times'] + 1
                
                loss_prev = loss_next
                loss_prev_mc = loss_next_mc
                loss_prev_data = loss_next_data
            
            else:
                loss_cache[i] = loss_prev
                loss_mc_cache[i] = loss_prev_mc
                loss_data_cache[i] = loss_prev_data
                step_cache[i] = False

            if ~only_save_last_bed:
                if self.detrend_map == True:
                    bed_cache[i,:,:] = bed_c + self.trend
                else:
                    bed_cache[i,:,:] = bed_c

            if i % info_per_iter == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache)/(i+1)}')

        resampled_times = psimdf.resampled_times.values.reshape((rows,cols))

        if self.detrend_map == True:
            last_bed = bed_c + self.trend
        else:
            last_bed = bed_c

        if ~only_save_last_bed:
            return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            return last_bed, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
