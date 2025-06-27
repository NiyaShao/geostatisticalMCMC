"""
author: Niya Shao
date: July 22, 2024

a collection of python function used for MCMC with Geostatistics-based update
"""

###assume already have data prepared
FILEPATH_GRIDDED_COND = '' #gridded conditioning data
FILEPATH_VELX = ''
FILEPATH_VELY = ''
FILEPATH_BMDF = '' #bedmachine dataframe, expected to have 1. X 2. Y 3. bed 4. 
FILEPATH_DHDT = ''
FILEPATH_SMB = './Data/SMB_RACMO2.3p2_yearly_ANT27_1979_2016.nc'


###import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pyproj import CRS, Transformer
import verde as vd

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import pairwise_distances
import gstatsim as gs
import gstools as gstools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skgstat as skg
from skgstat import models
import random
import time
import math

from PIL import Image, ImageFilter
import cv2

from statsmodels.tsa.stattools import acf

###functions
#find high velocity area

#load smb to dataframe
#next: incorporate time as an input rather hard-coded
def load_smb(dataset_path,xx,yy,interp_method='spline',k=1):
    ds = xr.load_dataset(dataset_path)
    
    crs_rotated = CRS('-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0')
    polar = CRS.from_epsg(3031)
    transformer = Transformer.from_crs(crs_rotated, polar)
    lonlon, latlat = np.meshgrid(ds.rlon.values, ds.rlat.values)
    xx2, yy2 = transformer.transform(lonlon, latlat)
    
    msk = (xx2 > xx.min()) & (xx2 < xx.max()) & (yy2 > yy.min()) & (yy2 < yy.max())
    ix = xx2[msk]
    iy = yy2[msk]
    iz = ds.isel(time=-2)['smb'].values.squeeze()[msk]
    # assume unit is water equivalent mm / yr, correct units to m/yr
    iz = iz/920

    # damping controls the smoothness
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    
    interp.fit((ix, iy), iz)
    preds_smb = interp.predict((xx.flatten(), yy.flatten()))
    preds_smb = preds_smb.reshape(xx.shape)

    vmax = max(np.nanmax(preds_smb), np.nanmax(iz))
    vmin = min(np.nanmin(preds_smb), np.nanmin(iz))
    print(np.max(preds_smb), np.max(iz),np.min(preds_smb), np.min(iz))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4), gridspec_kw={'wspace':-0.1})
    
    im = ax1.pcolormesh(xx, yy, preds_smb, vmin=vmin, vmax=vmax)
    ax1.axis('scaled')
    ax1.set_title(interp_method + ' interpolation')
    plt.colorbar(im, ax=ax1, pad=0.03, aspect=40, label='m/yr')
    
    im = ax2.scatter(ix, iy, c=iz, s=150, vmin=vmin, vmax=vmax)
    ax2.axis('scaled')
    ax2.set_title('SMB data')
    ax2.set_yticks([])
    plt.colorbar(im, ax=ax2, pad=0.03, aspect=40)
    
    return preds_smb, fig

#load dhdt to dataframe
def load_dhdt(dataset_path,xx,yy,interp_method='linear',k=1):
    
    ds2 = xr.open_dataset(dataset_path)
    ds2 = ds2.sel(x=(ds2.x > xx.min()) & (ds2.x < xx.max()), y=(ds2.y > yy.min()) & (ds2.y < yy.max()))
    
    ref = ds2.sel(time=slice('2014-05-01', '2014-06-01'))
    later = ds2.sel(time=slice('2016-05-01', '2016-06-01'))
    
    dhdt = (later['height_change'].values-ref['height_change'].values)/2

    v_max = np.nanmax(dhdt)
    v_min = np.nanmin(dhdt)
    
    xx2, yy2 = np.meshgrid(ds2.x.values, ds2.y.values)
    
    coordinates = (
        xx2.flatten(),
        yy2.flatten()
    )
    data = dhdt.flatten()
    
    # damping controls the smoothness
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    
    interp.fit(coordinates, data)
    preds_h = interp.predict((xx.flatten(), yy.flatten()))
    preds_h = preds_h.reshape(xx.shape)
    
    fig = plt.pcolormesh(xx/1000, yy/1000, preds_h, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.axis('scaled')
    plt.title('regridded surface height change rate')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.colorbar(pad=0.03, aspect=40, label='m')
    
    return preds_h, fig
    
#load velocities to dataframe
##TODO: test this function
def load_vel(dataset_path,xx,yy,interp_method='linear',k=1):
    ds2 = xr.open_dataset(dataset_path)
    ds2 = ds2.sel(x=(ds2.x > xx.min()) & (ds2.x < xx.max()), y=(ds2.y > yy.min()) & (ds2.y < yy.max()))
    
    xx2, yy2 = np.meshgrid(ds2.x.values, ds2.y.values)
    coordinates = (
        xx2.flatten(),
        yy2.flatten()
    )
    
    velx_err_raw = ds2['ERRX'].values
    vely_err_raw = ds2['ERRY'].values
    
    velx_err = velx_err_raw.flatten()
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    interp.fit(coordinates, velx_err)
    velx_err = interp.predict((xx.flatten(), yy.flatten()))
    velx_err = velx_err.reshape(xx.shape)
    
    vely_err = vely_err_raw.flatten()
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    interp.fit(coordinates, vely_err)
    vely_err = interp.predict((xx.flatten(), yy.flatten()))
    vely_err = vely_err.reshape(xx.shape)
    
    velx_raw = ds2['VX'].values
    vely_raw = ds2['VY'].values
    
    velx = velx_raw.flatten()
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    interp.fit(coordinates, velx)
    velx = interp.predict((xx.flatten(), yy.flatten()))
    velx = velx.reshape(xx.shape)
    
    vely = vely_raw.flatten()
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        print('the interpolation method input is not correctly defined, exit the function')
        return 0,0
    interp.fit(coordinates, vely)
    vely = interp.predict((xx.flatten(), yy.flatten()))
    vely = vely.reshape(xx.shape)
    
    return velx, vely, velx_err, vely_err

def find_mask(velx, vely, velmag_max, grounded_ice_mask, floating_mask, distance_max, xx, yy):
    mask = (grounded_ice_mask) & (np.sqrt(velx**2+vely**2) >= velmag_max)
    mask = mask | floating_mask

    image = Image.fromarray((mask * 255).astype(np.uint8))
    image = image.filter(ImageFilter.ModeFilter(size=10))
    mask_mat = np.array(np.array(image)/255,dtype=int)

    mask_dist = np.zeros(xx.shape)
    xx_hard = np.where(mask_mat==0, np.nan, xx)
    yy_hard = np.where(mask_mat==0, np.nan, yy)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            mask_dist[i,j] = np.nanmin(np.sqrt(np.square(yy[i,j]-yy_hard)+np.square(xx[i,j]-xx_hard)))

    mask_final = ((mask_dist<distance_max) & grounded_ice_mask)
    return mask_final


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

#define residual
def calc_mass_conservation(bed, surf, velx, vely, dhdt, smb):
    thick = surf - bed
    uHx = velx*thick
    uHy = vely*thick
    
    dx = np.gradient(velx*thick, 1000, axis=1)
    dy = np.gradient(vely*thick, 1000, axis=0)

    #dx = np.diff(uHx, axis=1, prepend=np.mean(uHx)) / 1000
    #dy = np.diff(uHy, axis=0, prepend=np.mean(uHy)) / 1000
    
    return dx + dy + dhdt - smb

#display mass conservation residual, its sumabs, sumsq. display radar data difference, its sumabs, sumsq
def print_loss(bed, high_vel_region, high_vel_mask, conditioning_data, data_mask, surf, velx, vely, dhdt, smb):
    mc_res = calc_mass_conservation(bed, surf, velx, vely, dhdt, smb)
    
    print('for the entire domain:')
    print('mass conservation loss: sumsq - ', loss_mc(mc_res, 'sumsq', inside_high_vel_region = False), ', sumabs - ', loss_mc(mc_res, 'sumabs', inside_high_vel_region = False))
    print('data loss: sumsq - ', loss_data(bed, conditioning_data, data_mask, 'sumsq', inside_high_vel_region = False), ', sumabs - ', loss_data(bed, conditioning_data, data_mask, 'sumabs', inside_high_vel_region = False))

    print('for the high velocity region:')
    print('mass conservation loss: sumsq - ', loss_mc(mc_res, 'sumsq', high_vel_mask = high_vel_mask), ', sumabs - ', loss_mc(mc_res, 'sumabs', high_vel_mask = high_vel_mask))
    print('data loss: sumsq - ', loss_data(bed, conditioning_data, data_mask, 'sumsq', high_vel_mask = high_vel_mask), ', sumabs - ', loss_data(bed, conditioning_data, data_mask, 'sumabs', high_vel_mask = high_vel_mask))

#detrending

#normal score transformation and compute variogram, based on different options

#mc res loss function
#input: mc res (matrix), function type (string), high velocity region mask (matrix)
#output: res loss (a single float number)
#define loss function
def loss_mc(mc_res, loss_type, inside_high_vel_region = True, high_vel_mask = -1):
    
    if inside_high_vel_region:
        if high_vel_mask.shape != mc_res.shape:
            raise Exception('if inside_high_vel_region is True, then the high_vel_mask need to be defined')
    else:
        high_vel_mask = np.full(mc_res.shape,1)
    
    if loss_type == 'meanabs':
        loss = np.nanmean(np.abs(mc_res[high_vel_mask==1]))
    elif loss_type == 'meansq':
        loss = np.nanmean(np.square(mc_res[high_vel_mask==1]))
    elif loss_type == 'sumabs':
        loss = np.nansum(np.abs(mc_res[high_vel_mask==1]))
    elif loss_type == 'sumsq':
        loss = np.nansum(np.square(mc_res[high_vel_mask==1]))
    else:
        raise Exception("the loss_type parameter is not correct")

    return loss

#data loss function
#input: simulated value (matrix), conditional value (matrix), data mask(matrix), function type (string), high velocity region mask (matrix), [optional] outlier_delta (number)
#output: data loss (a single float number)

def loss_data(bed, conditioning_data, data_mask, loss_type, inside_high_vel_region = True, high_vel_mask = -1, outlier_delta = -1):
    if bed.shape != conditioning_data.shape:
        raise Exception('the bed and the conditioning data have to be the same size')

    if bed.shape != data_mask.shape:
        raise Exception('the bed and the data_mask have to be the same size')
    
    if inside_high_vel_region:
        if high_vel_mask.shape != bed.shape:
            raise Exception('if inside_high_vel_region is True, then the high_vel_mask need to be defined')
    else:
        high_vel_mask = np.full(bed.shape,1)
    
    mask_bed = bed[(data_mask==1)&(high_vel_mask==1)]
    cond_v = conditioning_data[(data_mask==1)&(high_vel_mask==1)]
    
    if loss_type == 'meanabs':
        loss = np.nanmean(np.abs(mask_bed-cond_v))
    elif loss_type == 'meansq':
        loss = np.nanmean(np.square(mask_bed-cond_v))
    elif loss_type == 'sumabs':
        loss = np.nansum(np.abs(mask_bed-cond_v))
    elif loss_type == 'sumsq':
        loss = np.nansum(np.square(mask_bed-cond_v))
    elif loss_type == 'huber':
        if outlier_delta < 0:
            raise Exception('outlier_delta, which is required for huber loss, is not set correctly. It need to be a positive value')
        else:
            res = mask_bed - cond_v
            huber = []
            for r in res:
                if np.abs(r) <= outlier_delta:
                    huber.append(np.square(r)*0.5)
                else:
                    huber.append(outlier_delta*(np.abs(r)-(0.5*outlier_delta)))
        loss = np.nansum(huber)
    else:
        raise Exception("the loss_type parameter is not correct")

    return loss

#generate random Gaussian field
def generate_RF(X,Y,step_range,nug_max,range_min,range_max,model_name='Gaussian',_mean=0,_var=1,isotropic=False):
    rng = np.random.default_rng()
    scale  = rng.uniform(low=np.min(step_range), high=np.max(step_range), size=1)[0]/3
    nug = rng.uniform(low=0.0, high=nug_max, size=1)[0]
    if not isotropic:
        range1 = rng.uniform(low=range_min[0], high=range_max[0], size=1)[0]
        range2 = rng.uniform(low=range_min[1], high=range_max[1], size=1)[0]
        angle = rng.uniform(low=0, high=180, size=1)[0]
    else:
        range1 = rng.uniform(low=range_min[0], high=range_max[0], size=1)[0]
        range2 = range1
        angle = 0.0
        
    if model_name == 'Gaussian':
        model = gstools.Gaussian(dim=2, var = _var,
                            len_scale = [range1/np.sqrt(3),range2/np.sqrt(3)],
                            angles = angle*np.pi/180,
                            nugget = nug)
    elif model_name == 'Exponential':
        #a bug here, when exponential, lenscale should be /3, not /sqrt(3)
        model = gstools.Exponential(dim=2, var = _var,
                    len_scale = [range1/np.sqrt(3),range2/np.sqrt(3)],
                    angles = angle*np.pi/180,
                    nugget = nug)
    else:
        print('error model name')
        return

    srf = gstools.SRF(model)
    field = srf.structured([X, Y]).T*scale + _mean

    return field

#logistic related functions
#min_dist, rescale, logistic, get_crf_wegith are written by michael
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

def get_crf_weight(xx,yy,logistic_param,cond_data_mask,max_dist):
    dist = min_dist(np.where(cond_data_mask==0, np.nan, 1), xx, yy)
    dist_rescale = rescale(dist, max_dist)
    dist_logi = logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

    weight = dist_logi - np.min(dist_logi)
    return weight, dist, dist_rescale, dist_logi

def get_crf_weight_from_dist(xx,yy,logistic_param,cond_data_mask,max_dist,dist):
    dist_rescale = rescale(dist, max_dist)
    dist_logi = logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

    weight = dist_logi - np.min(dist_logi)
    return weight, dist, dist_rescale, dist_logi
    

#sample by field perturbation
def get_edge_mask(min_block_w, max_block_w, min_block_h, max_block_h, logistic_param, maxdist, res, num_step = 5):
    
    width = np.linspace(min_block_w,max_block_w,num_step,dtype=int)
    height = np.linspace(min_block_h,max_block_h,num_step,dtype=int)
    w,h = np.meshgrid(width,height)
    pairs = np.array([(w//2*2).flatten(),(h//2*2).flatten()])

    edge_masks = []

    for i in range(pairs.shape[1]):
        bwidth = pairs[:,i][0]
        bheight = pairs[:,i][1]
        xx,yy=np.meshgrid(range(bwidth),range(bheight))
        xx = xx*res #grid cell resolution, such that maxdist is appropriate
        yy = yy*res
        cond_msk_edge = np.zeros((bheight,bwidth))
        cond_msk_edge[0,:]=1
        cond_msk_edge[bheight-1,:]=1
        cond_msk_edge[:,0]=1
        cond_msk_edge[:,bwidth-1]=1
        dist_edge = min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
        dist_rescale_edge = rescale(dist_edge, maxdist)
        dist_logi_edge = logistic(dist_rescale_edge, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]
        edge_masks.append(dist_logi_edge)

    return pairs, edge_masks

    #sample by CRF created by logi function
    #sample by RF: CRF with logi weight = 1 -> RF
    #sample in high velocity region
    #sample in the entire studying domain
    #sample with different loss functions
    #with or without difference in data variance or/and mc variance in high velocity region or not

def sample_both_block_logi(bed, surf, velx, vely, dhdt, smb,
                           cond_bed, data_mask, resolution,
                           n_iter, block_size, rfgen_param, mc_loss_type, sigma_mc, 
                           logistic_param = -1, mc_region_mask = -1, sigma_data = -1, maxdist = -1, crf_weight = -1,
                           data_loss_type = 'default', data_loss = False, CRF = True, dataloss_in_high_vel = True,
                           mcloss_in_high_vel = True, onlychange_in_highvel = True,
                           outlier_delta = -1, isotropic=False, model_name='Gaussian'):

    #check input parameter
    if (bed.shape!=surf.shape) or (bed.shape!=velx.shape) or (bed.shape!=vely.shape) or (bed.shape!=dhdt.shape) or (bed.shape!=smb.shape) or (bed.shape!=cond_bed.shape) or (bed.shape!=data_mask.shape):
        raise Exception('the shape of bed, surf, velx, vely, dhdt, smb, radar_bed, data_mask need to be same')
    if (data_loss) and (sigma_data <= 0):
        raise Exception('if include data loss, the sigma_data have to be positive')
    if CRF and ((maxdist <= 0) or (len(logistic_param)!=4 or (crf_weight.shape!=bed.shape))):
        raise Exception('if update with conditional field, the maxdist need to be greater than 0 and the logistic param need be set to 4 numbers [L, x0, k, offset] and crf_weight need to have a same shape as bed')
    if (dataloss_in_high_vel or mcloss_in_high_vel or onlychange_in_highvel) and (mc_region_mask.shape!=bed.shape):
        raise Exception('if sample in_high_vel_region, then mc_region_mask need to be properly defined')

        
    # initialize storage
    loss_mc_cache = np.zeros(n_iter)
    loss_data_cache = np.zeros(n_iter)
    loss_cache = np.zeros(n_iter)
    step_cache = np.zeros(n_iter)
    bed_cache = np.zeros((n_iter, bed.shape[0], bed.shape[1]))
    blocks_cache = np.full((n_iter, 4), np.nan)
    resampled_times = np.zeros(bed.shape)
    

    #if loss is limited by high vel region mask
    if not (dataloss_in_high_vel or mcloss_in_high_vel or onlychange_in_highvel):
        mc_region_mask = np.full(bed.shape,1)

    # initialize loss
    mc_res = calc_mass_conservation(bed, surf, velx, vely, dhdt, smb)

    loss_prev_mc = loss_mc(mc_res, mc_loss_type, inside_high_vel_region = mcloss_in_high_vel, high_vel_mask = mc_region_mask)
    if data_loss:
        loss_prev_data = loss_data(bed, cond_bed, data_mask, data_loss_type, 
                                   inside_high_vel_region = dataloss_in_high_vel, high_vel_mask = mc_region_mask, 
                                   outlier_delta = outlier_delta)
        loss_prev = (loss_prev_mc / (2*sigma_mc**2)) + (loss_prev_data / (2*sigma_data**2))
    else:
        loss_prev = (loss_prev_mc / (2*sigma_mc**2))

    loss_cache[0] = loss_prev
    if data_loss:
        loss_data_cache[0] = loss_prev_data
    loss_mc_cache[0] = loss_prev_mc
    step_cache[0] = False
    bed_cache[0] = bed
    
    # get param
    range_max = rfgen_param[0]
    range_min = rfgen_param[1]
    step_range = rfgen_param[2]
    nug_max = rfgen_param[3]
    res = resolution
    
    rng = np.random.default_rng()
    
    min_block = block_size[0]
    max_block = block_size[1]

    #generate all possible field size and mask
    whpairs, edge_masks = get_edge_mask(min_block[0],max_block[0],min_block[1],max_block[1],logistic_param,maxdist,resolution)

    for i in range(1,n_iter):
        
        #choose block size
        block_size_i = rng.integers(low=0, high=whpairs.shape[1], size=1)[0]
        block_size = whpairs[:,block_size_i]

        if onlychange_in_highvel:
            while True:
                indexx = rng.integers(low=0, high=mc_region_mask.shape[0], size=1)[0]
                indexy = rng.integers(low=0, high=mc_region_mask.shape[1], size=1)[0]
                if mc_region_mask[indexx,indexy] == 1:
                    break
        else:
            indexx = rng.integers(low=0, high=bed.shape[0], size=1)[0]
            indexy = rng.integers(low=0, high=bed.shape[1], size=1)[0]

        #find the index of the block side, make sure the block is within the edge of the map
        bxmin = np.max((0,int(indexx-block_size[1]/2)))
        bxmax = np.min((bed.shape[0],int(indexx+block_size[1]/2)))
        bymin = np.max((0,int(indexy-block_size[0]/2)))
        bymax = np.min((bed.shape[1],int(indexy+block_size[0]/2)))

        #record block
        blocks_cache[i,:]=[indexx,indexy,block_size[0],block_size[1]]

        #generate field
        x_uniq = np.arange(0,block_size[0]*res,res)
        y_uniq = np.arange(0,block_size[1]*res,res)

        #in-case of a weird bug
        while True:
            f = generate_RF(x_uniq,y_uniq,step_range,nug_max,range_min,range_max,model_name=model_name,isotropic=isotropic)
            if (np.sum(np.isnan(f))) != 0:
                print('f have nan')
                continue
            else:
                break

        #find the index of the block side in the coordinate of the block
        mxmin = np.max([f.shape[0]-bxmax,0])
        mxmax = np.min([bed.shape[0]-bxmin,f.shape[0]])
        mymin = np.max([f.shape[1]-bymax,0])
        mymax = np.min([bed.shape[1]-bymin,f.shape[1]])
        
        #perturb
        if CRF:
            perturb = (f*edge_masks[block_size_i])[mxmin:mxmax,mymin:mymax]*crf_weight[bxmin:bxmax,bymin:bymax]
        else:
            perturb = (f*edge_masks[block_size_i])[mxmin:mxmax,mymin:mymax]
        
        bed_next = bed.copy()
        bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb
        
        if onlychange_in_highvel:
            bed_next = np.where(mc_region_mask, bed_next, bed)

        #make sure no bed elevation is greater than surface elevation
        #bed_next = np.where(bed_next>surf, surf, bed_next)

        #calculate new MC error only in the fast region
        mc_res = calc_mass_conservation(bed_next, surf, velx, vely, dhdt, smb)
        
        #loss
        loss_next_mc = loss_mc(mc_res, mc_loss_type, inside_high_vel_region = mcloss_in_high_vel, 
                               high_vel_mask = mc_region_mask)
        if data_loss:
            loss_next_data = loss_data(bed_next, cond_bed, data_mask, data_loss_type, 
                                       inside_high_vel_region = dataloss_in_high_vel, high_vel_mask = mc_region_mask, 
                                       outlier_delta = outlier_delta)
            loss_next = (loss_next_mc / (2*sigma_mc**2)) + (loss_next_data / (2*sigma_data**2))
        else:
            loss_next = (loss_next_mc / (2*sigma_mc**2))
            
        #make sure no bed elevation is greater than surface elevation
        block_thickness = surf[bxmin:bxmax,bymin:bymax] - bed_next[bxmin:bxmax,bymin:bymax]
        block_mc_region_mask = mc_region_mask[bxmin:bxmax,bymin:bymax]
        if np.sum((block_thickness<=0)[block_mc_region_mask==1]) > 0:
            loss_next = np.inf

        if loss_prev > loss_next:
            acceptance_rate = 1
        else:
            acceptance_rate = min(1,np.exp(loss_prev-loss_next))
        
        u = np.random.rand()
        if (u <= acceptance_rate):
            bed = bed_next
            
            loss_prev = loss_next
            loss_prev_mc = loss_next_mc
            loss_cache[i] = loss_next
            loss_mc_cache[i] = loss_next_mc

            if data_loss:
                loss_prev_data = loss_next_data
                loss_data_cache[i] = loss_next_data
            
            step_cache[i] = True
            resampled_times[bxmin:bxmax,bymin:bymax] += mc_region_mask[bxmin:bxmax,bymin:bymax]
            
        else:
            loss_mc_cache[i] = loss_prev_mc
            loss_cache[i] = loss_prev
            if data_loss:
                loss_data_cache[i] = loss_prev_data
            step_cache[i] = False

        bed_cache[i,:,:] = bed
        
        if i%100 == 0:
            print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} data loss: {loss_data_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache[np.max([0,i-1000]):i])/(np.min([i,1000]))}') #to window acceptance rate

    if data_loss:
        return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
    else:
        return bed_cache, loss_mc_cache, loss_cache, step_cache, resampled_times, blocks_cache
    

def exclude_data_rf(df_in, rf_bed, cond_bed, num_of_std, xx, yy, shallow, dfmaskname = 'bedmachine_mask'):
    
    df = df_in.copy()
    
    # plot the radar difference, give std
    fig = plt.figure(figsize=(8,6*3))
    gs = fig.add_gridspec(3,1,height_ratios = [2,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    
    rfradardiff = rf_bed - cond_bed
    stdrf = np.std(rfradardiff[~np.isnan(rfradardiff)])
    
    print('the standard deviation of difference to conditioning data is', stdrf)
    
    fig_bed = ax1.pcolormesh(xx/1000,yy/1000,rf_bed,cmap='gist_earth',vmin=-2500,vmax=2000)
    fig_diff = ax1.pcolormesh(xx/1000, yy/1000,rfradardiff,vmin=-1000, vmax=1000,cmap='RdBu')
    plt.colorbar(fig_bed,ax=ax1, aspect=40, label='m',orientation='horizontal')
    plt.colorbar(fig_diff,ax=ax1, aspect=40, label='m',orientation='horizontal')
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_title('final topography minus radar data')
    ax1.axis('scaled')
    
    ax3.pcolormesh(xx/1000, yy/1000,  (rfradardiff<stdrf*num_of_std)&(rfradardiff>-stdrf*num_of_std), cmap='YlGn')
    ax3.set_xlabel('X [km]')
    ax3.set_ylabel('Y [km]')
    ax3.set_title('if exclude positive and negative radardiff')
    ax3.axis('scaled')

    ax2.pcolormesh(xx/1000, yy/1000,  (rfradardiff>-stdrf*num_of_std), cmap='RdPu')
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_title('if only exclude negative radardiff (bed<rf)')
    ax2.axis('scaled')
    
    #exclude data
    df['bedQCrf'] = [np.nan]*df.shape[0]
    df['bedrf'] = rf_bed.flatten()
    num_excluded_data = 0
    for index, row in df.iterrows():
        if ((row[dfmaskname] == 3) | (row[dfmaskname] == 0)): #if in ice shelf
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        # elif (row['bedmachine_mask'] == 0): #or sea floor
        #     df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        elif pd.isna(row['bed']):
            continue
        elif (row['bed'] < row['bedrf'] + stdrf*num_of_std) and (row['bed'] > row['bedrf'] - stdrf*num_of_std) and (~shallow):
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        elif (row['bed'] < row['bedrf'] + stdrf*1.5) and (shallow):
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        else:
            num_excluded_data += 1
            
    print('the exclusion rate is',num_excluded_data / df[df['bed'].isnull()==False].shape[0])
            
    return df

def exclude_data_neariceshelf(df_in,resolution,distance_in_grid,dfmaskname='bedmachine_mask',bedindex='bedQCrf'):
    df = df_in.copy()
    distance = distance_in_grid*resolution
    for index,row in df.iterrows():
        if row[dfmaskname]==3:
            x_min = row['X']-distance
            x_max = row['X']+distance
            y_min = row['Y']-distance
            y_max = row['Y']+distance
            near_index = df[(df['X']<x_max) & (df['X']>x_min) & (df['Y']<y_max) & (df['Y']>y_min) & (df[dfmaskname]==2)].index
            df.loc[near_index,bedindex] = np.nan
    
    return df

def sgs_chain(psimdf, X_name, Y_name, Z_name, nst_trans, trend, resolution, variogram,
              data_mask, mc_region_mask,
              surf, velx, vely, dhdt, smb,
              n_iter, block_size, 
              mc_loss_type, sigma_mc, mcloss_in_high_vel,
              searching_radius, num_nearest_neighbors,
              rand_dropout, dropoutrate):
    
    #make sure psimdf only have three columns specified by the names

    rng = np.random.default_rng()
    
    min_block = block_size[0]
    max_block = block_size[1]
    rows = len(np.unique(psimdf[Y_name]))
    cols = len(np.unique(psimdf[X_name]))
    
    xmin = np.min(psimdf[X_name])
    xmax = np.max(psimdf[X_name])
    ymin = np.min(psimdf[Y_name])
    ymax = np.max(psimdf[Y_name])
    
    loss_cache = np.zeros(n_iter)
    loss_mc_cache = np.zeros(n_iter)
    step_cache = np.zeros(n_iter)
    bed_cache = np.zeros((n_iter, rows, cols))
    blocks_cache = np.full((n_iter, 4), np.nan)
    
    psimdf['data_mask'] = data_mask.flatten()
    data_index = psimdf[psimdf['data_mask']==1].index
    psimdf['mc_region_mask'] = mc_region_mask.flatten()
    mask_index = psimdf[psimdf['mc_region_mask']==1].index
    
    psimdf['resampled_times']=0
    
    bed = nst_trans.inverse_transform(np.array(psimdf[Z_name]).reshape(-1,1)).reshape(rows,cols) + trend
    mc_res = calc_mass_conservation(bed, surf, velx, vely, dhdt, smb)
    loss_prev_mc = loss_mc(mc_res, mc_loss_type, inside_high_vel_region = mcloss_in_high_vel, high_vel_mask = mc_region_mask)
    loss_prev = (loss_prev_mc / (2*sigma_mc**2))

    loss_cache[0] = loss_prev
    loss_mc_cache[0] = loss_prev_mc
    step_cache[0] = False
    bed_cache[0] = bed
    
    for i in range(n_iter):

        rsm_center_index = mask_index[int(np.random.rand()*len(mask_index))]
        rsm_x_center = psimdf.loc[rsm_center_index,'X']
        rsm_y_center = psimdf.loc[rsm_center_index,'Y']

        block_size_x = rng.integers(low=min_block[0], high=max_block[0], size=1)[0]
        block_size_x = int(block_size_x/2)*resolution #half of the block size
        block_size_y = rng.integers(low=min_block[1], high=max_block[1], size=1)[0]
        block_size_y = int(block_size_y/2)*resolution

        blocks_cache[i,:]=[rsm_x_center,rsm_y_center,block_size_x,block_size_y]

        #left corner in terms of meters
        rsm_x_min = np.max([int(rsm_x_center - block_size_x),xmin])
        rsm_x_max = np.min([int(rsm_x_center + block_size_x),xmax])
        rsm_y_min = np.max([int(rsm_y_center - block_size_y),ymin])
        rsm_y_max = np.min([int(rsm_y_center + block_size_y),ymax])

        rsm_x_dim = rsm_x_max - rsm_x_min
        rsm_y_dim = rsm_y_max - rsm_y_min

        resampling_box_index = psimdf[(rsm_x_min<=psimdf[X_name])&(psimdf[X_name]<rsm_x_max)&(rsm_y_min<=psimdf[Y_name])&(psimdf[Y_name]<rsm_y_max)].index
        if rand_dropout == True:
            intersect_index = resampling_box_index.intersection(data_index)
            intersect_index = np.random.choice(intersect_index, size=int(intersect_index.shape[0]*(1-dropoutrate)), replace=False, p=None)
            drop_index = resampling_box_index.difference(intersect_index)
        else:
            drop_index = resampling_box_index.difference(data_index)

        new_df = psimdf[~psimdf.index.isin(drop_index)].copy()

        Pred_grid_xy_change = gs.Gridding.prediction_grid(rsm_x_min, rsm_x_max - resolution, rsm_y_min, rsm_y_max - resolution, resolution)
        x = np.reshape(Pred_grid_xy_change[:,0], (len(Pred_grid_xy_change[:,0]), 1))
        y = np.flip(np.reshape(Pred_grid_xy_change[:,1], (len(Pred_grid_xy_change[:,1]), 1)))
        Pred_grid_xy_change = np.concatenate((x,y),axis=1)

        sim2 = gs.Interpolation.okrige_sgs(Pred_grid_xy_change, new_df, X_name, Y_name, Z_name, num_nearest_neighbors, variogram, searching_radius, quiet=True) 

        xy_grid = np.concatenate((Pred_grid_xy_change[:,0].reshape(-1,1),Pred_grid_xy_change[:,1].reshape(-1,1),np.array(sim2).reshape(-1,1)),axis=1)

        psimdf_next = psimdf.copy()
        psimdf_next.loc[resampling_box_index,[X_name,Y_name,Z_name]] = xy_grid
        bed_next = nst_trans.inverse_transform(np.array(psimdf_next[Z_name]).reshape(-1,1)).reshape(rows,cols) + trend
        
        #make sure no bed elevation is greater than surface elevation
        #bed_next = np.where(bed_next>surf, surf, bed_next)
        
        mc_res = calc_mass_conservation(bed_next, surf, velx, vely, dhdt, smb)
            
        loss_next_mc = loss_mc(mc_res, mc_loss_type, inside_high_vel_region = mcloss_in_high_vel, high_vel_mask = mc_region_mask)
        loss_next = (loss_next_mc / (2*sigma_mc**2))
        
        #make sure no bed elevation is greater than surface elevation
        thickness = surf - bed_next
        if np.sum((thickness<=0)[mc_region_mask==1]) > 0:
            loss_next = np.inf
        
        if loss_prev > loss_next:
            acceptance_rate = 1
        else:
            acceptance_rate = min(1,np.exp(loss_prev-loss_next))

        u = np.random.rand()
        if (u <= acceptance_rate):
            bed = bed_next
            psimdf = psimdf_next
            loss_cache[i] = loss_next
            loss_prev = loss_next
            loss_mc_cache[i] = loss_next_mc
            loss_prev_mc = loss_next_mc
            step_cache[i] = True
            loss_prev = loss_next
            psimdf.loc[drop_index, 'resampled_times'] = psimdf.loc[drop_index, 'resampled_times'] + 1
        else:
            loss_cache[i] = loss_prev
            loss_mc_cache[i] = loss_prev_mc
            step_cache[i] = False
        bed_cache[i,:,:] = bed

        if i % 100 == 0:
            print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache)/(i+1)}')

    resampled_times = psimdf.resampled_times.values.reshape((rows,cols))
            
    return bed_cache, loss_mc_cache, loss_cache, step_cache, resampled_times, blocks_cache


#plot all testing figures
def plot_figs(bed_cache, loss_mc_cache, loss_cache, step_cache, resampled_times, blocks_cache, surf, velx, vely, dhdt, smb, cond_bed, data_mask, resolution):
    return 0

#mc loss and data loss (sumsq and sumabs) in the entire domain and high velocity region
def plot_loss(loss_mc_cache, step_cache, mc_loss_bm = -1, has_dataloss = False, loss_data_cache = -1, data_loss_bm = -1):
    n_chains = loss_mc_cache.shape[0]
    fig = plt.figure()
    
    if has_dataloss:
        gs = fig.add_gridspec(3,1,height_ratios = [3,3,1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])

        for i in range(n_chains):
            ax1.plot(loss_data_cache[i,:])
        ax1.axhline(y = data_loss_bm, color = 'r', linestyle = '-', label='bedmachine data loss')
        ax1.set_ylabel('loss')
        ax1.set_title('data loss')
    else:
        gs = fig.add_gridspec(2,1,height_ratios = [3,1])
        ax2 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0])

    for i in range(n_chains):
        ax2.plot(loss_mc_cache[i,:])
    if mc_loss_bm != -1:
        ax2.axhline(y = mc_loss_bm, color = 'r', linestyle = '-', label='bedmachine mc loss')
    ax2.set_ylabel('loss')
    ax2.set_title('mass conservation loss')

    for i in range(n_chains):
        win_accept_rate = np.array(
            [(np.sum(step_cache[i,j-1000:j])/1000.0)
             for j in range(1000,step_cache.shape[1])]
        )
        
    ax3.plot(np.arange(1000,step_cache.shape[1]),win_accept_rate)
    ax3.set_xlim(0,step_cache.shape[1])
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('average acceptance rate of last 1000 iterations')
    
    return fig
    
#std and sample times of the bed
def plot_sample(bed_cache, resampled_times, xx, yy, vmax_std = -1, vmax_times = -1):
    fig = plt.figure(figsize=(18,9))
    
    gs = fig.add_gridspec(1,2,width_ratios = [1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    if vmax_std == -1:
        std = ax1.pcolormesh(xx/1000, yy/1000,np.std(bed_cache[:,:,:],axis=0),cmap='viridis')
    else:
        std = ax1.pcolormesh(xx/1000, yy/1000,np.std(bed_cache[:,:,:],axis=0),cmap='viridis',vmax=vmax_std)
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_title('loaded topography standard deviation')
    ax1.axis('scaled')

    if vmax_times == -1:
        resample = ax2.pcolormesh(xx/1000,yy/1000,np.sum(resampled_times[:,:,:],axis=0),cmap='viridis')
    else:
        resample = ax2.pcolormesh(xx/1000,yy/1000,np.sum(resampled_times[:,:,:],axis=0),cmap='viridis',vmax=vmax_times)
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_title('resampled times')
    ax2.axis('scaled')

    plt.colorbar(std, pad=0.1, ax=ax1, aspect=30, location='bottom', label='bed elevation std (m)')
    plt.colorbar(resample, pad=0.1, ax=ax2, aspect=30, location='bottom', label='times (1)')

    return fig

#last topo, first topo, their difference
#last topo, bm, their difference
#difference between topo and conditioning data, overlayed by mc_region
def plot_diff(bed_cache, bm_bed, cond_bed, mc_region_mask, bed_min, bed_max, xx, yy, value_range=-1, cmap='RdBu'):
    last_bed = bed_cache[-1]
    first_bed = bed_cache[0]
    if value_range == -1:
        value_range = np.nanmax(np.abs(last_bed - first_bed))
    
    fig = plt.figure(figsize=(16,24))
    
    gs = fig.add_gridspec(3,3,height_ratios = [1,1,1],width_ratios=[1,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 0])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 2])
    ax9 = fig.add_subplot(gs[2, 1])

    ax1.pcolormesh(xx/1000, yy/1000,last_bed,cmap='gist_earth',vmax=bed_max,vmin=bed_min)
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_title('last topo')
    ax1.axis('scaled')
            
    ax2.pcolormesh(xx/1000,yy/1000,first_bed,cmap='gist_earth',vmax=bed_max,vmin=bed_min)
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_title('first topo')
    ax2.axis('scaled')
    
    topo = ax3.pcolormesh(xx/1000,yy/1000,bm_bed,cmap='gist_earth',vmax=bed_max,vmin=bed_min)
    ax3.set_xlabel('X [km]')
    ax3.set_ylabel('Y [km]')
    ax3.set_title('BedMachine bed')
    ax3.axis('scaled')
    plt.colorbar(topo, pad=0.02, ax=ax9, aspect=30, location='bottom', label='bed elevation (m)',)

    change = ax4.pcolormesh(xx/1000,yy/1000,(last_bed - first_bed),vmax=value_range,vmin=-1*value_range,cmap="RdBu")
    ax4.set_xlabel('X [km]')
    ax4.set_ylabel('Y [km]')
    ax4.set_title('change in chain')
    ax4.axis('scaled')

    diff = ax5.pcolormesh(xx/1000,yy/1000,(last_bed - bm_bed),vmax=value_range,vmin=-1*value_range,cmap="RdBu")
    ax5.set_xlabel('X [km]')
    ax5.set_ylabel('Y [km]')
    ax5.set_title('last bed - bm bed')
    ax5.axis('scaled')

    plt.colorbar(change, pad=0.02, ax=ax9, location='bottom', aspect=30, label='bed elevation difference (m)')
    #plt.colorbar(diff, pad=0.02, ax=ax5,location='bottom', aspect=30, label='bed elevation difference (m)')
    
    ax6.pcolormesh(xx/1000,yy/1000,cond_bed,cmap='gist_earth',vmax=bed_max,vmin=bed_min)
    ax6.set_xlabel('X [km]')
    ax6.set_ylabel('Y [km]')
    ax6.set_title('conditioning data')
    ax6.axis('scaled')

    ax7.pcolormesh(xx/1000,yy/1000,last_bed - cond_bed,vmax=value_range,vmin=-1*value_range,cmap="RdBu")
    ax7.set_xlabel('X [km]')
    ax7.set_ylabel('Y [km]')
    ax7.set_title('last bed - conditioning data')
    ax7.axis('scaled')

    ax8.pcolormesh(xx/1000,yy/1000,bm_bed - cond_bed,vmax=value_range,vmin=-1*value_range,cmap="RdBu")
    ax8.set_xlabel('X [km]')
    ax8.set_ylabel('Y [km]')
    ax8.set_title('bedmachine - conditioning data')
    ax8.axis('scaled')

    plt.tight_layout()
    return fig
    
#mass conservation residual & its sumsq & sumabs loss & the difference in distribution compared to bedmachine
def plot_mcres_datares(mc_res, radardiff, high_vel_region, sigma_mcres_laplacian = -1, sigma_mcres_gaussian = -1, sigma_data_laplacian = -1, sigma_data_gaussian = -1):
    xl3=np.linspace(-100, 100, num=1000)
    if sigma_mcres_gaussian != -1:
        gaussian_model=1/(sigma_mcres_gaussian*np.sqrt(2*np.pi))*np.exp(-0.5*np.square(xl3/sigma_mcres_gaussian))
    if sigma_mcres_laplacian != -1:
        laplacian_model=1/(2*sigma_mcres_laplacian)*np.exp(-1*np.abs(xl3)/sigma_mcres_laplacian)

    mc_res_bm_f = mc_res_bm.flatten()
    mc_res_bm_f = mc_res_bm_f[~np.isnan(mc_res_bm_f)]
    mc_res_bm_sr2_f = mc_res_bm_sr2.flatten()
    mc_res_bm_sr2_f = mc_res_bm_sr2_f[~np.isnan(mc_res_bm_sr2_f)]
    bins=np.histogram(np.hstack((mc_res_bm_f,mc_res_bm_sr2_f)), bins=5000)[1] #get the bin edges

    #plt.hist(mc_res_bm_f, bins=bins, facecolor='blue', alpha=0.2,density=True,label='entire domain')
    plt.hist(mc_res_bm_sr2_f, bins=bins, facecolor='red', alpha=0.3,density=True,label='BM source == 2')
    plt.plot(xl3, gaussian_model, color='Green',alpha=0.6)
    #plt.plot(xl3, laplacian_model, color='Orange',alpha=0.6)
    plt.xlim([-50,50]);
    #plt.ylim([0,0.04])
    plt.legend()
    plt.xlabel('mass conv residual');
    plt.ylabel('Frequency');
    plt.title('histogram of mass conservation residual (bedmachine)')
    plt.grid(True)
    plt.savefig('hist1.png')


#generate video for the topo chain
#axis 0 of the data is time-dimension, axis 1 and 2 are space-dimensions
def plot_video(video_path, data, vmin, vmax, colormapname = 'gist_earth', fps = 25):
    width = data.shape[2]
    height = data.shape[1]
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height), True)
    normal_range = vmax-vmin
    video_np = np.array(((data + vmin)/ normal_range * 255), dtype='uint8')
    cmap = get_mpl_colormap(colormapname)
    for i in range(data.shape[0]):
        image_gray = np.flip(video_np[i,:,:], axis=0)
        image_c = cv2.applyColorMap(image_gray, cmap)
        video.write(image_c)
    cv2.destroyAllWindows()
    video.release()
    
#
def plot_distribution():
    return -1

# TODO: not tested
def find_example_loc(num_pt, data_mask, high_vel_region, xx, yy, bed = -1):
    
    rng = np.random.default_rng()
    
    actual_xy_list = np.zeros((num_pt,2))
    index_xy_list = np.zeros((num_pt,2))
    
    for i in range(num_pt):
    
        while(True):

            indexx = rng.integers(low=0, high=data_mask.shape[0], size=1)[0]
            indexy = rng.integers(low=0, high=data_mask.shape[1], size=1)[0]

            if (data_mask[indexx,indexy]==0) & (high_vel_region[indexx,indexy]==1):
                break

        x = xx[indexx,indexy]
        y = yy[indexx,indexy]
        
        index_xy_list[i,0] = indexx
        index_xy_list[i,1] = indexy
        actual_xy_list[i,0] = x
        actual_xy_list[i,1] = y
    
    return index_xy_list, actual_xy_list
    
    
def plot_autocorr(example_values):
    fig = plt.figure(figsize=(12,20))
    
    gs = fig.add_gridspec(2,1,height_ratios = [1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    color = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
    
    for i in range(example_values.shape[0]):
        ax1.plot(example_values[i,:], c=color[i%9])
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('the bed elevation')
    ax1.set_title('bed elevation at example locations')
    
    for i in range(example_values.shape[0]):
        autocorr = acf(example_values[i,:], nlags=example_values.shape[1]-1)
        ax2.plot(autocorr,c=color[i%9])
    ax2.set_xlabel('lag')
    ax2.set_ylabel('autocorrelation')
    ax2.set_title('autocorrelation of example locations')
    
    return fig