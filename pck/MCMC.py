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

import Topography

"""
Fit and compare different variogram models to the given data.
Notice here by default nugget = 0. If want to use for the case of nugget != 0, need to rewrite the function.

Args:
    data (2D numpy array): A column of input data to compute the variogram on. (use .reshape((-1,1)) when the input data is not a column array)
    coords (2D numpy array): Coordinates corresponding to the data, where each row representing a location in space, and first column is for x coordinate and second column for y coordinate
    roughness_region_mask (2D array): Mask indicating region where roughness is evaluated (1 = included).
    maxlag (float): Maximum lag distance to consider in variogram.
    n_lags (int): Number of lags to compute the variogram. Default is 50.
    samples (float): Proportion of pairs of points used to compute the experimental variogram. Default is 0.6.
    subsample (int): Number of samples for quantile transformation. Default is 100000.
    data_for_trans (list): Optional, only used when the data used to compute the quantile transformation is different from the data transformed. This should be a column array containing data used to calculate quantile transformation

Returns:
    nst_trans: QuantileTransformer object used to normalize the data.
    transformed_data: Transformed data used in variogram calculation.
    params (list): List of parameters for Gaussian, Exponential, and Spherical variogram models, containing range, sill, and nugget
    fig: A matplotlib figure comparing experimental and modeled variograms.
"""
def fit_variogram(data, coords, roughness_region_mask, maxlag, n_lags=50, samples=0.6, subsample=100000, data_for_trans = []):

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

    tp1 = test1.parameters
    tp2 = test2.parameters
    tp3 = test3.parameters

    print('range, sill, and nugget for gaussian variogram is ', tp1)
    print('for exponential variogram is ', tp2)
    print('for spherical variogram is ', tp3)
    
    # extract experimental variogram values
    xdata1 = test1.bins
    ydata1 = test1.experimental
    xdata2 = test2.bins
    ydata2 = test2.experimental
    xdata3 = test3.bins
    ydata3 = test3.experimental

    # evaluate models
    xi = np.linspace(0, xdata2[-1], n_lags) 
    y_gauss = [models.gaussian(h, tp1[0], tp1[1], tp1[2]) for h in xi]
    y_exp = [models.exponential(h, tp2[0], tp2[1], tp2[2]) for h in xi]
    y_sph = [models.spherical(h, tp3[0], tp3[1], tp3[2]) for h in xi]

    # plot variogram model
    fig = plt.figure(figsize=(6,4))
    plt.plot(xi, y_gauss,'b--', label='Gaussian variogram')
    plt.plot(xi, y_exp,'b-', label='Exponential variogram')
    plt.plot(xi, y_sph,'b*-', label='Spherical variogram')
    plt.plot(xdata1, ydata1,'o', markersize=4, color='green', label='Experimental variogram gaussian')
    plt.plot(xdata2, ydata2,'o', markersize=4, color='orange', label='Experimental variogram exponential')
    plt.plot(xdata3, ydata3,'o', markersize=4, color='pink', label='Experimental variogram spherical')
    plt.title('Variogram for synthetic data')
    plt.xlabel('Lag [m]'); plt.ylabel('Semivariance')  
    plt.legend(loc='lower right')
    
    return nst_trans, transformed_data, [tp1, tp2, tp3], fig

"""
a class used to represent unconditional or conditional random field
"""
class RandField:
    
# =============================================================================
#     """Generate edge masks with logistic decay function for various block sizes.
#     Here, the weight is highest (1) at the conditioning data, and has a logistic decay based on distance to the nearest conditioning data
#     
#     Args:
#         min_block_w (int): Minimum block width, in unit of grid cell.
#         max_block_w (int): Maximum block width, in unit of grid cell.
#         min_block_h (int): Minimum block height, in unit of grid cell.
#         max_block_h (int): Maximum block height, in unit of grid cell.
#         logistic_param (list of floats): Parameters [L, x0, k, offset] for logistic function f(x) = (L / (1 + exp(-k(x - x0)))) - offset
#         maxdist (float): Maximum distance for scaling logistic decay. Should be set to the 'range' distance where the correlation between two points reach minimum
#         res (float): Resolution of grid.
#         num_step (int): Number of steps between min and max block sizes. Default is 5.
#     
#     Returns:
#         pairs (numpy array): Array of block width-height pairs.
#         edge_masks (list of 2D arrays): List of logistic decay masks.
#     """
# # =============================================================================
# =============================================================================
#     def __get_edge_mask(self,min_block_w, max_block_w, min_block_h, max_block_h, logistic_param, maxdist, res, num_step = 5):
#         
#         width = np.linspace(min_block_w,max_block_w,num_step,dtype=int)
#         height = np.linspace(min_block_h,max_block_h,num_step,dtype=int)
#         w,h = np.meshgrid(width,height)
#         pairs = np.array([(w//2*2).flatten(),(h//2*2).flatten()])
# 
#         edge_masks = []
# 
#         for i in range(pairs.shape[1]):
#             bwidth = pairs[:,i][0]
#             bheight = pairs[:,i][1]
#             xx,yy=np.meshgrid(range(bwidth),range(bheight))
#             xx = xx*res #grid cell resolution, such that maxdist is appropriate
#             yy = yy*res
#             cond_msk_edge = np.zeros((bheight,bwidth))
#             cond_msk_edge[0,:]=1
#             cond_msk_edge[bheight-1,:]=1
#             cond_msk_edge[:,0]=1
#             cond_msk_edge[:,bwidth-1]=1
#             dist_edge = RandField.min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
#             dist_rescale_edge = RandField.rescale(dist_edge, maxdist)
#             dist_logi_edge = RandField.logistic(dist_rescale_edge, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]
#             edge_masks.append(dist_logi_edge)
# 
#         return pairs, edge_masks
# =============================================================================
    def __init_func(self):
        print('before using the RandField object in the MCMC chain, please set up its block sizes using set_block_sizes function, and set up conditional weight using set_block_param function')

    """
    Initialize a RandField object for generating conditional or unconditional random field.

    Args:
        range_max_x (float): Maximum spatial correlation range in x-direction. The random field will randomly sampled between range_max_x and range_min_x to determine its range in the x direction, and similarly for range in y direction.
        range_max_y (float): Maximum range in y-direction.
        range_min_x (float): Minimum range in x-direction.
        range_min_y (float): Minimum range in y-direction.
        scale_min (float): Minimum vertical scaling (std dev) of field.
        scale_max (float): Maximum vertical scaling of field.
        nugget_max (float): Maximum nugget effect in the variogram.
        model_name (str): Variogram model type ('Gaussian', 'Exponential').
        isotropic (bool): Whether to enforce isotropic spatial correlation. If isotropic is set to False, then fields with random anistropy direction will be generated, where the strength of anistropy is dependent on the ratio between range in x-direction and in y-direction.
            
        rng (str or Generator): Random number generator or string 'default'.
    """
    def __init__(self,range_max_x,range_max_y,range_min_x,range_min_y,scale_min,scale_max,nugget_max,model_name,isotropic,rng='default'):
        
        if rng == 'default':
            self.rng = np.random.default_rng()
        else: # TODO, check if passed rng is a actual generator
            self.rng = rng
        
        self.range_max_x = range_max_x
        self.range_max_y = range_max_y
        self.range_min_x = range_min_x
        self.range_min_y = range_min_y
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.nugget_max = nugget_max
        self.model_name = model_name
        self.isotropic = isotropic
        
        self.__init_func()
           
    """
    Set the allowable range of block sizes used for random field generation.
    The function has no returned value. But the effect can be evaluated by the 'pair' attribute of the RandField object
    
    Args:
        min_block_x (int): Minimum block width.
        max_block_x (int): Maximum block width.
        min_block_y (int): Minimum block height.
        max_block_y (int): Maximum block height.
        steps (int): Number of steps between min and max, determining number of different sizes between min_x and max_x or between min_y and max_y Default is 5.
    """
    def set_block_sizes(self,min_block_x,max_block_x,min_block_y,max_block_y,steps=5):
        self.min_block_x = min_block_x
        self.min_block_y = min_block_y
        self.max_block_x = max_block_x
        self.max_block_y = max_block_y
        self.steps = steps

        self.pairs = self.get_block_sizes()
    
    """
    Set logistic function parameters for generating conditioning weight used on both conditioning to the data and conditioning to the edge of the block
    The logistic function has a format of f(x) = (L / (1 + exp(-k(x - x0)))) - offset, where the x here represent the distance to the nearest conditioning data
    In the generated conditioning weight, the weight is highest (1) at the conditioning data, and has a logistic decay based on distance to the nearest conditioning data
    
    
    Args:
        logis_func_L (float): L parameter of logistic function.
        logis_func_x0 (float): Midpoint x0 of logistic function.
        logis_func_k (float): Growth rate k of logistic function.
        logis_func_offset (float): Constant subtracted from logistic output.
        max_dist (float): Max distance for mask scaling.
        resolution (float): Grid cell resolution.
    """
    def set_block_param(self, logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution):
        if not hasattr(self, 'pairs'):
            raise Exception('It seems like the set_block_sizes has not been called yet before calling set_block_param')
        
        self.logistic_param = [logis_func_L, logis_func_x0, logis_func_k, logis_func_offset]
        self.max_dist = max_dist
        self.resolution = resolution
        
        self.edge_masks = self.get_edge_masks()
        
    
    def get_block_sizes(self):
        width = np.linspace(self.min_block_x,self.max_block_x,self.steps,dtype=int)
        height = np.linspace(self.min_block_y,self.max_block_y,self.steps,dtype=int)
        w,h = np.meshgrid(width,height)
        pairs = np.array([(w//2*2).flatten(),(h//2*2).flatten()])
        
        return pairs
    
    def get_edge_masks(self):
        
        edge_masks = []
        pairs = self.pairs
        res = self.resolution

        for i in range(pairs.shape[1]):
            bwidth = pairs[:,i][0]
            bheight = pairs[:,i][1]
            xx,yy=np.meshgrid(range(bwidth),range(bheight))
            xx = xx*res 
            yy = yy*res
            cond_msk_edge = np.zeros((bheight,bwidth))
            cond_msk_edge[0,:]=1
            cond_msk_edge[bheight-1,:]=1
            cond_msk_edge[:,0]=1
            cond_msk_edge[:,bwidth-1]=1
            dist_edge = RandField.min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
            dist_rescale_edge = RandField.rescale(dist_edge, self.max_dist)
            dist_logi_edge = RandField.logistic(dist_rescale_edge, self.logistic_param[0], self.logistic_param[1], self.logistic_param[2]) - self.logistic_param[3]
            edge_masks.append(dist_logi_edge)

        return edge_masks
    
    """
    Generate a random field using specified model parameter when initiating the RandField object.
    
    Args:
        X (1D numpy array): x-coordinates of the grid.
        Y (1D numpy array): y-coordinates of the grid.
        _mean (float): Mean of the field. Default is 0. Do not change it
        _var (float): Variance of the field. Default is 1. Do not change it
    
    Returns:
        field (2D numpy array): Realization of the random field.
    """
    def get_random_field(self,X,Y,_mean=0,_var=1):
        
        
        rng = self.rng
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
        else:
            print('error model name')
            return

        srf = gstools.SRF(model)
        field = srf.structured([X, Y]).T*scale + _mean

        return field
    
    def min_dist(hard_mat, xx, yy):
        dist = np.zeros(xx.shape)
        xx_hard = np.where(np.isnan(hard_mat), np.nan, xx)
        yy_hard = np.where(np.isnan(hard_mat), np.nan, yy)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                dist[i,j] = np.nanmin(np.sqrt(np.square(yy[i,j]-yy_hard)+np.square(xx[i,j]-xx_hard)))
        return dist

    def rescale(x, maxdist):
        return np.where(x>maxdist,1,(x/maxdist))

    def logistic(x, L, x0, k):
        return L/(1+np.exp(-k*(x-x0)))
    
    """
    Generate conditional random field weights from conditioning data mask.
    
    Args:
        xx (2D array): x-coordinate grid.
        yy (2D array): y-coordinate grid.
        cond_data_mask (2D array): Binary mask of conditioning data (1 = present).
    
    Returns:
        weight (2D array): weight used to condition the field to conditioning data
        dist (2D array): Distance to conditioning points.
        dist_rescale (2D array): Scaled distances, where 0 correspond to 0, and 1 correspond to self.max_dist.
        dist_logi (2D array): The raw logistic function values
        
    """
    def get_crf_weight(self,xx,yy,cond_data_mask):
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        dist = RandField.min_dist(np.where(cond_data_mask==0, np.nan, 1), xx, yy)
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi

    """
    Generate conditional random field weights from calculated distance. 
    Used in large domain where calculating distance takes too long and thus the distance could be previously saved in files
    
    Args:
        xx (2D array): x-coordinate grid.
        yy (2D array): y-coordinate grid.
        cond_data_mask (2D array): Binary mask of conditioning data (1 = present).
        dist (2D array): Distance to conditioning points.
    
    Returns:
        weight (2D array): weight used to condition the field to conditioning data
        dist_rescale (2D array): Scaled distances, where 0 correspond to 0, and 1 correspond to self.max_dist.
        dist_logi (2D array): The raw logistic function values
        
    """
    def get_crf_weight_from_dist(self,xx,yy,dist):
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi
    
    
    """
    Generate a random field block where the random field is derived from the initialization of the RandField object
    and the block size randomly selected from set_block_sizes function. 
    The block has a logistic decay to its block edge
    
    Returns:
        block (2D numpy array): Random field sample shaped by logistic mask.
    """
    def get_rfblock(self):
        res = self.resolution
        
        # randomly choose a size from the list
        block_size_i = self.rng.integers(low=0, high=self.pairs.shape[1], size=1)[0]
        block_size = self.pairs[:,block_size_i]
        
        #generate field
        x_uniq = np.arange(0,block_size[0]*res,res)
        y_uniq = np.arange(0,block_size[1]*res,res)

        #in-case of a weird bug
        while True:
            f = self.get_random_field(x_uniq, y_uniq)
            if (np.sum(np.isnan(f))) != 0:
                print('f have nan')
                continue
            else:
                break
            
        return f*self.edge_masks[block_size_i]
    
"""
parent class for both the crf_chain and sgs_chain
"""
class chain:

    def __init_func__(self):
        return
    
    """
    Initialize the Markov chain object for topography sampling.
    The function has no returns but the input argument are stored in the chain object
    
    Args:
        xx, yy (2D array): x and y coordinates of the map grid.
        bed (2D array): Initial bed topography.
        surf (2D array): Ice surface elevation.
        velx, vely (2D array): Ice surface velocity in x direction and in y direction.
        dhdt (2D array): Surface height change rate (annual average).
        smb (2D array): Surface mass balance (annual average).
        cond_bed (2D array): Conditioning radar bed measurements in the shape of the 2D domain. For locations without measurement, the value is nan
        data_mask (2D array): Mask of where conditioning data exists (1 = exist).
        grounded_ice_mask (2D array): Binary mask of grounded ice (1 = grounded).
        resolution (float): Spatial resolution in meters.
    
    Raises:
        Exception: If input arrays do not have matching shapes.
        
    """
    def __init__(self, xx, yy, bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution):
        self.xx = xx
        self.yy = yy
        self.bed = bed
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
        
        if (bed.shape!=surf.shape) or (bed.shape!=velx.shape) or (bed.shape!=vely.shape) or (bed.shape!=dhdt.shape) or (bed.shape!=smb.shape) or (bed.shape!=cond_bed.shape) or (bed.shape!=data_mask.shape):
            raise Exception('the shape of bed, surf, velx, vely, dhdt, smb, radar_bed, data_mask need to be same')
        
        self.__init_func__()
        
    """
    Define spatial region where ice surface velocity is high to constrain the location of updates.
    
    Args:
        update_in_region (bool): Whether to constrain updates to region_mask.
        region_mask (2D array): Binary mask specifying region of interest.
    
    Raises:
        ValueError: If region_mask has incorrect shape.
    """
    def set_high_vel_region(self, update_in_region, region_mask = []):
       
        if region_mask.shape != self.xx.shape:
            raise ValueError('the region_mask input is invalid. It has to be a 2D numpy array with the shape of the map')
        else:
            self.region_mask = region_mask
        
        if update_in_region == False:
            print('the update blocks is set to be randomly generated for any locations inside the entire map')
        self.update_in_region = update_in_region
    
    def __meanabs(data,mask):
        return np.nanmean(np.abs(data[mask==1]))
    
    def __meansq(data,mask):
        return np.nanmean(np.square(data[mask==1]))
    
    def __sumabs(data,mask):
        return np.nansum(np.abs(data[mask==1]))
    
    def __sumsq(data,mask):
        return np.nansum(np.square(data[mask==1]))
    
    def __return0(data,mask):
        return 0

    """
    Configure loss function used in MCMC chain
    Currently enable two losses to be used: mass conservation residual and misfit to radar measurements.
    The function has no return. Its effect can be checked in chain object's 'loss_function_list' attribute
    
    Args:
        map_func (str): Function to use for mass conservation residual ('meanabs', 'meansquare', 'sumabs', 'sumsquare', None). 
        Default is None. If set to None, then mass conservation residual is not used for calculating the loss.
        'sumsquare' correspond to treat the residual as Gaussian distribution
        
        diff_func (str): Function to use for misfit to radar measurements ('meanabs', 'meansquare', 'sumabs', 'sumsquare', None). 
        Default is None. If set to None, then mass conservation residual is not used for calculating the loss.
        'sumsquare' correspond to treat the misfit as Gaussian distribution
        
        sigma_mc (float): Standard deviation for mass conservation residual.
        sigma_data (float): Standard deviation for data misfit.
        massConvInRegion (bool): Whether to calculate mass conservation residual only in region_mask.
        dataDiffInRegion (bool): Whether to calculate data misfit loss only in region_mask.
    
    Raises:
        ValueError: If function names are invalid or sigmas are missing.

    """
    def set_loss_type(self, map_func = None, diff_func = None, sigma_mc = -1, sigma_data = -1, massConvInRegion = True, dataDiffInRegion = False):
    
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
    
    """
    Compute total loss from mass conservation residuals and data misfit by applying function defined in set_loss_type
    
    Args:
        massConvResidual (2D array): Mass conservation residual field.
        dataDiff (2D array): Difference between candidate and observed bed elevation.
    
    Returns:
        total_loss (float): Combined loss value.
        loss_mc (float): Loss due to mass conservation residual.
        loss_data (float): Loss due to data misfit.
    """
    def loss(self, massConvResidual, dataDiff):
        
        f1 = self.loss_function_list[0]
        f2 = self.loss_function_list[1]
            
        # TODO: is it inappropriate to use sum when the two loss have unequal number of grid cells?
        loss_mc = f1(massConvResidual, self.mc_region_mask) / (2*self.sigma_mc**2)
        loss_data = f2(dataDiff, (self.data_mask==1)&(self.data_region_mask==1)) / (2*self.sigma_data**2)
        
        return loss_mc + loss_data, loss_mc, loss_data

"""
Inherit the chain class. Used for creating conditioning random field MCMC chains, can also be used for unconditional random fields
"""        
class chain_crf(chain):
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using chainname.set_high_vel_region(update_in_region, region_mask)')
        print('then please set up the loss function using either chainname.set_loss_type or chainname.set_loss_func')
        print('an RandField object also need to be created correctly and passed in chainname.set_crf_data_weight(RF) and in chain.run(n_iter, RF)')
        return
    
    """
    Set types of the perturbation blocks. 
    For now, can choose from unconditional random field ('RF') or conditional random field created by logistic weighting (CRF_weight)
    
    Args:
        block_type (str): 'CRF_weight', 'CRF_rbf' (not implemented), or 'RF'.
    
    Raises:
        ValueError: If block_type is invalid.
    """
    def set_update_type(self, block_type):
        
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
    
    """
    Calculate and store CRF weights from conditioning data using RandField object.
    
    Args:
        RF (RandField): A RandField object configured for CRF weight generation.
    """
    def set_crf_data_weight(self, RF):
        
        crf_weight, dist, dist_rescale, dist_logi = RF.get_crf_weight(self.xx,self.yy,self.data_mask)
        self.crf_data_weight = crf_weight

    """
    Run the MCMC chain using block-based CRF/RF perturbations.
    
    Args:
        n_iter (int): Number of iterations in the MCMC chain.
        RF (RandField): Random field generator.
        rng (str or np.random.Generator): Random number generator. Default is 'default', which makes a random generator with random seeds
    
    Returns:
        bed_cache (4D array): Topography at each iteration.
        loss_mc_cache (1D array): Mass conservation residual loss at each iteration. If the mass conservation loss is not used, return array of 0
        loss_data_cache (1D array): Data misfit loss at each iteration. If the data misfit loss is not used, return array of 0
        loss_cache (1D array): Total loss at each iteration.
        step_cache (1D array): Boolean indicating if the step was accepted.
        resampled_times (2D array): Number of times each pixel was updated.
        blocks_cache (2D array): Info on block proposals at each iteration.
    """
    def run(self, n_iter, RF, rng='default'):
        
        if rng == 'default':
            rng = np.random.default_rng()
            
        if not isinstance(RF, RandField):
            raise TypeError('The arugment "RF" has to be an object of the class RandField')
        
       # initialize storage
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        loss_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        bed_cache = np.zeros((n_iter, self.xx.shape[0], self.xx.shape[1]))
        blocks_cache = np.full((n_iter, 4), np.nan)
        resampled_times = np.zeros(self.xx.shape)
        
        # TODO: should i have an additional property called initial_bed?
        bed_c = self.bed
        
        # initialize loss
        mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb)
        data_diff = bed_c - self.cond_bed
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_data_cache[0] = loss_prev_data
        loss_mc_cache[0] = loss_prev_mc
        step_cache[0] = False
        bed_cache[0] = bed_c
        
        crf_weight = self.crf_data_weight

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
            
            #TODO: Okay this is fine, the problem is more of the boundary of the high velocity region

            #find the index of the block side in the coordinate of the block
            mxmin = np.max([block_size[0]-bxmax,0])
            mxmax = np.min([bed_c.shape[0]-bxmin,block_size[0]])
            mymin = np.max([block_size[1]-bymax,0])
            mymax = np.min([bed_c.shape[1]-bymin,block_size[1]])
            
            #perturb
            if self.block_type == 'CRF_weight':
                perturb = f[mxmin:mxmax,mymin:mymax]*crf_weight[bxmin:bxmax,bymin:bymax]
            else:
                perturb = f[mxmin:mxmax,mymin:mymax]

            bed_next = bed_c.copy()
            bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb
            
            if self.update_in_region:
                bed_next = np.where(self.region_mask, bed_next, bed_c)
            else:
                bed_next = np.where(self.grounded_ice_mask, bed_next, bed_c)
                
            mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb)
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
            
            u = np.random.rand()
            if (u <= acceptance_rate):
                bed_c = bed_next
                
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

            bed_cache[i,:,:] = bed_c
            
            if i%1000 == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} data loss: {loss_data_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache[np.max([0,i-1000]):i])/(np.min([i,1000]))}') #to window acceptance rate

        return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache

#
#this function should also be inheritable
#

#sample by CRF created by logi function
#sample by RF: CRF with logi weight = 1 -> RF
#sample in high velocity region
#sample in the entire studying domain
#sample with different loss functions
#with or without difference in data variance or/and mc variance in high velocity region or not

class chain_sgs(chain):
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using chainname.set_update_in_region(region_mask) and chainname.set_high_vel_region(update_in_region)')
        print('please also set up the sgs parameters using chainname.set_sgs_param(self, block_size, sgs_param)')
        print('then please set up the loss function using either chainname.set_loss_type or chainname.set_loss_func')
        
    """
    Set the normal score transformation object (from scikit-learn package) used to normalize the bed elevation.
    
    Args:
        nst_trans (QuantileTransformer): A fitted scikit-learn transformer used to normalize input data.
    
    Note:
        This transformation must be fit beforehand (e.g., via `MCMC.fit_variogram`) and should match the scale of bed values.
    """
    def set_normal_transformation(self, nst_trans):
        self.nst_trans = nst_trans
      
    """
    Set the long-wavelength trend component of the bed topography.
    Notice that detrend topography means that the SGS simulation will only simulate the short-wavelength topography residuals that is not a part of the trend
    
    Args:
        trend (2D array): The trend surface to add back after SGS sampling.
        detrend_map (bool): If True, remove trend before transforming the bed elevation and add it back after inverse transform.
    
    Raises:
        ValueError: If detrend_map is True but trend has invalid shape.
    """
    def set_trend(self, trend = None, detrend_map = True):
        if detrend_map == True:
            if len(trend)!=len(self.xx) or trend.shape!=self.xx.shape:
                raise ValueError('if detrend_map is set to True, then the trend of the topography, which is a 2D numpy array, must be provided')
            else:
                self.trend = trend
        else:
            self.trend = None
        self.detrend_map = detrend_map
    
    """
    Specify variogram model and its parameters for SGS interpolation.
    
    Args:
        vario_type (str): Variogram model type. One of 'Gaussian', 'Exponential', 'Spherical', 'Matern' (to be implemented).
        vario_range (float or list): Correlation range(s). One value for isotropic; list of two for anisotropic.
        vario_sill (float): Variogram sill (variance).
        vario_nugget (float): Nugget effect.
        isotropic (bool): Whether the variogram is isotropic. Default is True.
        vario_smoothness (float): Smoothness parameter for Matern model (required if `vario_type` is 'Matern').
        vario_azimuth (float): Azimuth angle for anisotropic variograms in degrees.
    
    Raises:
        ValueError: If required parameters are missing or in the wrong format.
    """
    def set_variogram(self, vario_type, vario_range, vario_sill, vario_nugget, isotropic = True, vario_smoothness = None, vario_azimuth = None):
        
        if (vario_type == 'Gaussian') or (vario_type == 'Exponential') or (vario_type == 'Spherical'):
            print('the variogram is set to type', vario_type)
        elif vario_type == 'Matern':
            if (vario_smoothness == None) or (vario_smoothness <= 0):
                raise ValueError('vario_smoothness argument should be a positive float when the vario_type is Matern')
        else:
            raise ValueError('vario_type argument should be one of the following: Gaussian, Exponential, Spherical, or Matern')
        
        self.vario_type = vario_type
        
        # TODO, to add Matern, need to change it here
        if isotropic:
            vario_azimuth = 0
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type]
        else:
            if (len(vario_range) == 2):
                print('set to anistropic variogram with major range and minor range to be ', vario_range)
                self.vario_param = [vario_azimuth, vario_nugget, vario_range[0], vario_range[1], vario_sill, vario_type]
            else:
                raise ValueError ("vario_range need to be a list with two floats to specifying for major range and minor range of the variogram when isotropic is set to False")
    
    """
    Set parameters for Sequential Gaussian Simulation (SGS).
    
    Args:
        sgs_num_nearest_neighbors (int): Number of nearest neighbors used in simulation.
        sgs_searching_radius (float): Radius (in meters) to search for neighbors.
        sgs_rand_dropout_on (bool): Whether to randomly drop conditioning points in simulation block.
        dropout_rate (float): Proportion of conditioning data to drop if dropout is enabled (between 0 and 1).
    """
    def set_sgs_param(self, sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on = False, dropout_rate = 0):
        
        if sgs_rand_dropout_on == False:
            dropout_rate = 0
            print('because the sgs_rand_dropout_on is set to False, the dropout_rate is automatically set to 0')
            
        self.sgs_param = [sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on, dropout_rate]
    
    """
    Set minimum and maximum block sizes (in grid cells) for SGS updates.
    
    Args:
        block_min_x (int): Minimum width of block in x-direction. Unit in grid cells
        block_min_y (int): Minimum height of block in y-direction.
        block_max_x (int): Maximum width of block in x-direction.
        block_max_y (int): Maximum height of block in y-direction.
    """
    def set_block_sizes(self, block_min_x, block_min_y, block_max_x, block_max_y):
        self.block_min_x = block_min_x
        self.block_min_y = block_min_y
        self.block_max_x = block_max_x
        self.block_max_y = block_max_y

    """
    Run the MCMC chain using block-based SGS updates
    
    Args:
        n_iter (int): Number of iterations in the MCMC chain.
        rng (str or np.random.Generator): Random number generator. Default is 'default', which makes a random generator with random seeds
    
    Returns:
        bed_cache (4D array): Topography at each iteration.
        loss_mc_cache (1D array): Mass conservation residual loss at each iteration. If the mass conservation loss is not used, return array of 0
        loss_data_cache (1D array): Data misfit loss at each iteration. If the data misfit loss is not used, return array of 0
        loss_cache (1D array): Total loss at each iteration.
        step_cache (1D array): Boolean indicating if the step was accepted.
        resampled_times (2D array): Number of times each pixel was updated.
        blocks_cache (2D array): Info on block proposals at each iteration.
    """
    def run(self, n_iter, rng='default'):
        
        if rng == 'default':
            rng = np.random.default_rng()
        
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
        bed_cache = np.zeros((n_iter, rows, cols))
        blocks_cache = np.full((n_iter, 4), np.nan)
        
        # TODO, should i have an additional property called initial_bed?
        bed_c = self.bed

        nst_trans = self.nst_trans
        
        if self.detrend_map == True:
            z = nst_trans.transform((bed_c-self.trend).reshape(-1,1))
        else:
            z = nst_trans.transform(bed_c.reshape(-1,1))
            
        # Need an additional parameter to store normalized actual conditioning data
        z_cond_bed = nst_trans.transform(self.cond_bed.reshape(-1,1))
        cond_bed_data = np.array([self.xx.flatten(),self.yy.flatten(),z_cond_bed.flatten()])
        cond_bed_df = pd.DataFrame(cond_bed_data.T, columns=['x','y','cond_bed']) #cond_bed_df should share the same index as psimdf
        
        resolution = self.resolution
    
        df_data = np.array([self.xx.flatten(),self.yy.flatten(),z.flatten(),self.data_mask.flatten(),self.mc_region_mask.flatten()])
        psimdf = pd.DataFrame(df_data.T, columns=['x','y','z','data_mask','mc_region_mask'])
        psimdf['resampled_times'] = 0
        
        #psimdf['data_mask'] = data_mask.flatten()
        data_index = psimdf[psimdf['data_mask']==1].index
        #psimdf['mc_region_mask'] = mc_region_mask.flatten()
        mask_index = psimdf[psimdf['mc_region_mask']==1].index
        
        # initialize loss
        mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb)
        data_diff = bed_c - self.cond_bed
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_mc_cache[0] = loss_prev_mc
        loss_data_cache[0] = loss_prev_data
        step_cache[0] = False
        bed_cache[0] = bed_c
        
        for i in range(n_iter):
            
            # TODO, now by default it will sample in high velocity region
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

            # TODO, add seeding Generator into the gs
            sim2 = gs.Interpolation.okrige_sgs(Pred_grid_xy_change, new_df, 'x', 'y', 'z', self.sgs_param[0], self.vario_param, self.sgs_param[1], quiet=True) 

            xy_grid = np.concatenate((Pred_grid_xy_change[:,0].reshape(-1,1),Pred_grid_xy_change[:,1].reshape(-1,1),np.array(sim2).reshape(-1,1)),axis=1)

            psimdf_next = psimdf.copy()
            psimdf_next.loc[resampling_box_index,['x','y','z']] = xy_grid
            if self.detrend_map == True:
                bed_next = nst_trans.inverse_transform(np.array(psimdf_next['z']).reshape(-1,1)).reshape(rows,cols) + self.trend
            else:
                bed_next = nst_trans.inverse_transform(np.array(psimdf_next['z']).reshape(-1,1)).reshape(rows,cols)
            
            mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb)
            data_diff = bed_next - self.cond_bed
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
            
            #make sure no bed elevation is greater than surface elevation
            thickness = self.surf - bed_next
            
            if np.sum((thickness<=0)[self.grounded_ice_mask==1]) > 0:
                loss_next = np.inf
            
            if loss_prev > loss_next:
                acceptance_rate = 1
                
            else:
                acceptance_rate = min(1,np.exp(loss_prev-loss_next))

            u = np.random.rand()
            
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
                
            bed_cache[i,:,:] = bed_c

            if i % 1000 == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache)/(i+1)}')

        resampled_times = psimdf.resampled_times.values.reshape((rows,cols))
                
        return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache