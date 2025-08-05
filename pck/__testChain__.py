# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gstatsim as gs
import Topography
import MCMC
import time

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
sgs_bed = np.loadtxt('../sgs_bed_50_crf.txt')

thickness = bm_surface - sgs_bed
sgs_bed = np.where((thickness<=0)&(bm_mask==2), bm_surface-1, sgs_bed)


csgs = MCMC.chain_sgs(xx, yy, sgs_bed, bm_surface, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
csgs.set_high_vel_region(True,highvel_mask)
csgs.set_loss_type(map_func='sumsquare', diff_func='sumsquare', sigma_mc=3, sigma_data=50, massConvInRegion=True)

trend = np.loadtxt('../trend_51_crf.txt')
csgs.set_trend(trend = trend, detrend_map = True)


# find variograms
df_bed = df.copy()
df_bed['detrended_bed'] = df_bed['bed'].values - trend.flatten()
df_bed = df_bed[df_bed["detrended_bed"].isnull() == False]
data = df_bed['detrended_bed'].values.reshape(-1,1)
#df_bed = df_bed[df_bed["bed"].isnull() == False]
#data = df_bed['bed'].values.reshape(-1,1)

coords = df_bed[['X','Y']].values
roughness_region_mask = (df_bed['bedmachine_mask'].values)==2

nst_trans, Nbed_radar, varios, fig = MCMC.fit_variogram(data, coords, roughness_region_mask, maxlag=70000, n_lags=70)

csgs.set_normal_transformation(nst_trans)

csgs.set_variogram('Spherical',[varios[2][0],varios[2][0]-5000],varios[2][1],0,isotropic=False,vario_azimuth=20)

csgs.set_sgs_param(48, 50e3, sgs_rand_dropout_on=True, dropout_rate=0.3)

# TODO make naming consistent betweeen sgs chain and crf chain
min_block_x = 10
max_block_x = 30
min_block_y = 10
max_block_y = 30
csgs.set_block_sizes(min_block_x, min_block_y, max_block_x, max_block_y)

seed = 1
randomGenerator = np.random.default_rng(seed)
bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache = csgs.run(n_iter=200, rng=randomGenerator)
 

# TODO: change from crf to wrf, or just rf
# =============================================================================
# # start a crf chain
# cond_bed = df['bed'].values.reshape(xx.shape)
# data_mask = ~np.isnan(cond_bed)
# sgs_bed = np.loadtxt('../sgs_bed_50_crf.txt')
# mean = np.mean(cond_bed[data_mask])
# #bed = np.full(xx.shape,np.mean(cond_bed[data_mask]))
# bed = np.where(grounded_ice_mask,mean,cond_bed)
# 
# ccrf = MCMC.chain_crf(xx, yy, bed, bm_surface, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
# ccrf.set_high_vel_region(False,highvel_mask)
# ccrf.set_loss_type(map_func='sumsquare', diff_func='sumsquare', sigma_mc=3, sigma_data=80, massConvInRegion=False)
# 
# range_max_x = 50e3 #in terms of meters in lateral dimension, regardless of resolution of the map
# range_max_y = 50e3
# range_min_x = 10e3
# range_min_y = 10e3
# step_min = 100 #in terms of meters in vertical dimension, how much you want to multiply the perturbation by
# step_max = 400
# nugget_max = 0 #short-scale roughness
# random_field_model = 'Exponential'
# isotropic = True
# max_dist = varios[1][0]
# rf1 = MCMC.RandField(range_max_x, range_max_y, range_min_x, range_min_y, step_min, step_max, nugget_max, random_field_model, isotropic)
# 
# min_block_x = 20
# max_block_x = 50
# min_block_y = 20
# max_block_y = 50
# rf1.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)
# 
# logis_func_L = 2
# logis_func_x0 = 0
# logis_func_k = 6
# logis_func_offset = 1
# rf1.set_block_param(logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution)
# 
# ccrf.set_crf_data_weight(rf1)
# ccrf.set_update_type('RF')
# 
# seed = 0
# randomGenerator = np.random.default_rng(seed)
# bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache = ccrf.run(n_iter=5000, RF=rf1, rng=randomGenerator)
# 
# =============================================================================
# TODO: write a function for loading topography from txt
# =============================================================================
# begin_time = time.time()
# n_iter_list = [5000]*10
# sep_list = [100]*10
# file_postfix_list = ['testRF'+str(i) for i in range(10)]
# 
# for i in range(0,len(n_iter_list)):
#     print('the i is ', i)
#     
#     n_iter = n_iter_list[i]
#     file_postfix = file_postfix_list[i]
#     sep = sep_list[i]
#     bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache = ccrf.run(n_iter=5000, RF=rf1, rng=randomGenerator)
# 
#     ccrf.bed = bed_cache[-1]
# 
#     if sep == 1: # save all data
#         np.savetxt("bed_cache"+file_postfix+".txt", bed_cache[:n_iter,:,:].reshape(n_iter, -1))
#     else: # save a map per sep
#         choice = np.round(np.linspace(0, n_iter-sep, num=int(n_iter/sep))).astype(int)
#         np.savetxt("bed_cache"+file_postfix+".txt", bed_cache[choice,:,:].reshape(len(choice), -1))
# 
#     np.savetxt("loss_cache"+file_postfix+".txt",loss_cache[:n_iter])
#     np.savetxt("loss_mc_cache"+file_postfix+".txt",loss_mc_cache[:n_iter])
#     np.savetxt("loss_data_cache"+file_postfix+".txt",loss_data_cache[:n_iter])
#     np.savetxt("step_cache"+file_postfix+".txt",step_cache[:n_iter])
#     np.savetxt("resampled_times"+file_postfix+".txt",resampled_times)
#     np.savetxt("blocks_cache"+file_postfix+".txt",blocks_cache)
# 
# end_time = time.time()
# used_time = end_time - begin_time
# print('used ', used_time, ' seconds')
# =============================================================================


## TEST PLAN
# test mean bed with RF update and both data & massConv loss for the entire map
#       tested
# test sgs bed with CRF update with only massConv loss
#       tested
# test sgs bed with SGS update with both data and massConv loss without detrending with isotropic sgs
#       random dropout not working, need to fix that
#       tested
# test sgs bed with SGS update with only massConv loss with detrending with anistropic sgs
# test sgs bed with SGS update with only massConv loss with detrending with isotrpic sgs with dropout