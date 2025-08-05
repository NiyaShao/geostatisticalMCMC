#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:13:58 2025

@author: niyashao
"""

import numpy as np
import Topography

def test_1():
    
    print('testing for loading racmo dataset')
    dataset_path = '../Data/SMB_RACMO2.3p2_yearly_ANT27_1979_2016.nc'
    smb, fig = Topography.load_smb_racmo(dataset_path, xx, yy, interp_method='spline',time=2014)
    return smb, fig


def test_2():
    
    print('testing for loading height change rate dataset')
    dhdt, fig = Topography.load_dhdt('../Data/ANT_G1920_GroundedIceHeight_v01.nc',xx,yy,interp_method='linear',begin_year = 2010,end_year=2020,month=6)    
    return dhdt, fig

def test_3():
    print('testing for loading InSAR_MEaSUREs velocity dataset')
    velx, vely, velxerr, velyerr, figvel = Topography.load_vel_measures('../../Data/antarctica_ice_velocity_450m_v2.nc', xx, yy)
    return velx, vely, velxerr, velyerr, figvel

def test_4():
    print('testing for loading BedMachine dataset')
    bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig = Topography.load_bedmachine('../../Data/BedMachineAntarctica-v3.nc', xx, yy)
    return bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig

def test_5():
    print('testing for loading radar dataset')
    df, df_out, fig = Topography.load_radar('../Data/radarTest', './radar_compiled_test.csv', include_only_thickness_data=False)
    return df, df_out, fig

# =============================================================================
# df = pd.read_csv('../Data/compiled_Denman_QC_55_shallow.csv')
# df = df.rename(columns={"bed": "bed_notpreprocessed"})
# df = df.rename(columns={"bedQCrf": "bed"})
# =============================================================================

# =============================================================================
# x_uniq = np.unique(df.X)
# y_uniq = np.unique(df.Y)
# 
# xmin = np.min(x_uniq)
# xmax = np.max(x_uniq)
# ymin = np.min(y_uniq)
# ymax = np.max(y_uniq)
# 
# cols = len(x_uniq)
# rows = len(y_uniq)
# 
# resolution = 1000
# 
# xx, yy = np.meshgrid(x_uniq, y_uniq)
# xx.shape
# =============================================================================

#smb, fig_smb = test_1()

#dhdt, fig_dhdt = test_2()

#velx, vely, velxerr, velyerr, fig_vel = test_3()

#bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig_bm = test_4()

#df, df_out, fig_radar = test_5()

# getting high velocity boundary
#ocean_mask = (bm_mask == 0) | (bm_mask == 3)
#high_vel_mask = Topography.get_highvel_boundary(velx, vely, 50, bm_mask==2, ocean_mask, 3000, xx, yy)

# determine the domain's size and resolution
xmin = 2300000
xmax = 2600000
ymax = -300000
ymin = -500000

res = 1000

print('testing for loading radar dataset')
df, df_out, fig = Topography.load_radar('../Data/radarTest', './radar_compiled_test.csv', include_only_thickness_data=False)

print('gridding radar data')
df=df[(df['x']>=xmin) & (df['x']<=xmax) & (df['y']>=ymin) & (df['y']<=ymax)]
df_grid, grid_matrix, rows, cols = Topography.grid_data(df, 'x', 'y', 'bed', res, xmin, xmax, ymin, ymax)
df_grid = df_grid.rename(columns = {"Z": "bed"})

x_uniq = np.unique(df_grid.X)
y_uniq = np.unique(df_grid.Y)

xx, yy = np.meshgrid(x_uniq, y_uniq)
xx.shape

print('converting the radar data based on the geoid')
geoid_bedmap = Topography.convert_geoid(xx, yy, '../Data/geoid_EIGEN-GL04C.gdf')
geoid_bm = Topography.convert_geoid(xx, yy, '../Data/geoid_EIGEN-6C4.gdf')
geoid_diff = geoid_bm - geoid_bedmap
beforeChange = df_grid['bed'].values
df_grid['bed'] = df_grid['bed'] + geoid_diff

# =============================================================================
# print('testing for loading InSAR_MEaSUREs velocity dataset')
# velx, vely, velxerr, velyerr, figvel = Topography.load_vel_measures('../../Data/antarctica_ice_velocity_450m_v2.nc', xx, yy)
# 
# print('testing for loading height change rate dataset')
# dhdt, fig = Topography.load_dhdt('../Data/ANT_G1920_GroundedIceHeight_v01.nc',xx,yy,interp_method='linear',begin_year = 2013,end_year=2015,month=7)
# 
# print('testing for loading racmo dataset')   
# smb, fig = Topography.load_smb_racmo('../Data/SMB_RACMO2.3p2_yearly_ANT27_1979_2016.nc', xx, yy, interp_method='spline',time=2014)
# 
# print('testing for loading BedMachine dataset')
# bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig = Topography.load_bedmachine('../../Data/BedMachineAntarctica-v3.nc', xx, yy)
# =============================================================================

