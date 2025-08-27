#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 20:33:53 2025

@author: niyashao
"""

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import gstools as gstools
import skgstat as skg
from skgstat import models

import random
import time
import math
import os

from PIL import Image, ImageFilter

import Topography
import MCMC

import MCMC_newsgs as MCMCn

import gstatsim_custom as gsim

# load data from saved csv
df = pd.read_csv('../Data/compiled_Denman_QC_55_shallow.csv')
df=df.rename(columns={"bed": "bed_notpreprocessed"})
df=df.rename(columns={"bedQCrf": "bed"})

x_uniq = np.unique(df.X)
y_uniq = np.unique(df.Y)

xmin = np.min(x_uniq)
xmax = np.max(x_uniq)
ymin = np.min(y_uniq)
ymax = np.max(y_uniq)

cols = len(x_uniq)
rows = len(y_uniq)

resolution = 1000

xx, yy = np.meshgrid(x_uniq, y_uniq)

velx, vely, velxerr, velyerr, figvel = Topography.load_vel_measures('../../Data/antarctica_ice_velocity_450m_v2.nc', xx, yy)
dhdt, fig = Topography.load_dhdt('../Data/ANT_G1920_GroundedIceHeight_v01.nc',xx,yy,interp_method='linear',begin_year = 2013,end_year=2015,month=7)
smb, fig = Topography.load_smb_racmo('../Data/SMB_RACMO2.3p2_yearly_ANT27_1979_2016.nc', xx, yy, interp_method='spline',time=2014)
bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig = Topography.load_bedmachine('../../Data/BedMachineAntarctica-v3.nc', xx, yy)

#data_index = df[df["bed"].isnull() == False].index
#mask_index = df[df["mask"]==1].index

# calculate high velocity region
ocean_mask = (bm_mask == 0) | (bm_mask == 3)
grounded_ice_mask = (bm_mask == 2)
highvel_mask = Topography.get_highvel_boundary(velx, vely, 50, grounded_ice_mask, ocean_mask, 3000, xx, yy)

# start a sgs chain
cond_bed = df['bed'].values.reshape(xx.shape)
data_mask = ~np.isnan(cond_bed)

# find variograms
df_bed = df.copy()
df_bed = df_bed[df_bed["bed"].isnull() == False]
data = df_bed['bed'].values.reshape(-1,1)
coords = df_bed[['X','Y']].values
roughness_region_mask = (df_bed['bedmachine_mask'].values)==2

nst_trans, Nbed_radar, varios, fig = MCMCn.fit_variogram(data, coords, roughness_region_mask, maxlag=70000, n_lags=70)

vario = {
    'azimuth' : 0.0,
    'nugget' : 0.0,
    'major_range' : varios[2][0],
    'minor_range' : varios[2][0],
    'sill' : varios[2][1],
    'vtype' : 'spherical',
}

rad = 50e3
k = 48

grid_norm = np.full(cond_bed.shape, np.nan)
np.place(grid_norm, data_mask, Nbed_radar)

#sgs_bed = MCMCn.sgs(xx, yy, grid_norm, vario, rad, k, seed=0)
#sgs_bed = nst_trans.inverse_transform(sgs_bed.reshape(-1,1)).reshape(rows,cols)

sgs_bed = np.loadtxt('../Tutorials/sgs_bed.txt')
thickness = bm_surface - sgs_bed
sgs_bed = np.where((thickness<=0)&(bm_mask==2), bm_surface-1, sgs_bed)



# testing for the new MCMC

csgs = MCMCn.chain_sgs(xx, yy, sgs_bed, bm_surface, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
csgs.set_high_vel_region(True,highvel_mask)
csgs.set_loss_type(map_func='sumsquare', sigma_mc=3, massConvInRegion=True)

#trend = np.loadtxt('../trend_51_crf.txt')
#csgs.set_trend(trend = trend, detrend_map = True)
csgs.set_trend(trend = [], detrend_map = False)
csgs.set_normal_transformation(nst_trans)

csgs.set_variogram('Spherical',varios[2][0],varios[2][1],0,isotropic=True,vario_azimuth=0)

csgs.set_sgs_param(48, 50e3, sgs_rand_dropout_on=False)

# TODO make naming consistent betweeen sgs chain and crf chain
min_block_x = 10
max_block_x = 30
min_block_y = 10
max_block_y = 30
csgs.set_block_sizes(min_block_x, min_block_y, max_block_x, max_block_y)

start = time.time()
bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, blocks_cache = csgs.run(n_iter=1000)
print('it takes ', start - time.time())


# testing for the old MCMC

csgs = MCMC.chain_sgs(xx, yy, sgs_bed, bm_surface, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
csgs.set_high_vel_region(True,highvel_mask)
csgs.set_loss_type(map_func='sumsquare', sigma_mc=3, massConvInRegion=True)

#trend = np.loadtxt('../trend_51_crf.txt')
#csgs.set_trend(trend = trend, detrend_map = True)
csgs.set_trend(trend = [], detrend_map = False)
csgs.set_normal_transformation(nst_trans)

csgs.set_variogram('Spherical',varios[2][0],varios[2][1],0,isotropic=True,vario_azimuth=0)

csgs.set_sgs_param(48, 50e3, sgs_rand_dropout_on=False)

# TODO make naming consistent betweeen sgs chain and crf chain
min_block_x = 10
max_block_x = 30
min_block_y = 10
max_block_y = 30
csgs.set_block_sizes(min_block_x, min_block_y, max_block_x, max_block_y)

start = time.time()
bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, blocks_cache = csgs.run(n_iter=1000)
print('it takes ', start - time.time())


#newsim = gsim.interpolate.sgs(xx, yy, cond_bed, vario, rad, k, seed=0)

#newsim = MCMCn.sgs(xx, yy, grid_norm, vario, rad, k, seed=0)

