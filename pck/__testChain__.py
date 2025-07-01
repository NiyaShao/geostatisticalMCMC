# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
#from scipy.interpolate import RBFInterpolator


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import QuantileTransformer
#from sklearn.metrics import pairwise_distances
import gstatsim as gs
import gstools as gstools
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import skgstat as skg
from skgstat import models
#from matplotlib.colors import LightSource

import random
import time
import math
import os

from PIL import Image, ImageFilter

import Topography
import MCMC

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

data_index = df[df["bed"].isnull() == False].index
mask_index = df[df["mask"]==1].index

# find variograms
df_bed = df.copy()
df_bed = df_bed[df_bed["bed"].isnull() == False]
data = df_bed['bed'].values.reshape(-1,1)
coords = df_bed[['X','Y']].values
roughness_region_mask = (df_bed['bedmachine_mask'].values)==2

nst_trans, Nbed_radar, varios, fig = MCMC.fit_variogram(data, coords, roughness_region_mask, maxlag=70000, n_lags=70)

cond_bed = df['bed'].values.reshape(xx.shape)
data_mask = cond_bed != np.nan
bed = np.full(xx.shape,np.mean(cond_bed[data_mask]))

# calculate high velocity region
ocean_mask = (bm_mask == 0) | (bm_mask == 3)
highvel_mask = Topography.get_highvel_boundary(velx, vely, 50, bm_mask==2, ocean_mask, 3000, xx, yy)

# start a crf chain
ccrf = MCMC.chain_crf(xx, yy, bed, bm_surface, velx, vely, dhdt, smb, cond_bed, data_mask, resolution)
ccrf.set_high_vel_region(True,highvel_mask)
ccrf.set_loss_type(map_func='sumsquare',sigma_mc = 3,massConvInRegion=True)

#rf1 = MCMC.RandField(range_max_x, range_max_y, range_min_x, range_min_y, step_min, step_max, nugget_max, random_field_model, isotropic = True, max_dist = )
#min_block_x = 10
#max_block_x = 30
#min_block_y = 10
#max_block_y = 30
#rf1.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)
#logis_func_L = 2
#logis_func_x0 = 0
#logis_func_k = 6
#logis_func_offset = 1
#max_dist = ?
#rf1.set_block_param(logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution)

#ccrf.set_crf_data_weight(rf1)

#seed = 0
#ccrf.run(n_iter=1000, RF=rf1, rng=np.random.default_rng(seed))

