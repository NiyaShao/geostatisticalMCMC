#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:13:58 2025

@author: niyashao
"""

import numpy as np
import pandas as pd

import Topography

def test_1():
    
    print('testing for loading racmo dataset')

    dataset_path = '../Data/SMB_RACMO2.3p2_yearly_ANT27_1979_2016.nc'
    
    smb, fig = Topography.load_smb_racmo(dataset_path, xx, yy, interp_method='spline',time=2015)
    
    return smb, fig


def test_2():
    
    dhdt, fig = Topography.load_dhdt('../Data/ANT_G1920_GroundedIceHeight_v01.nc',xx,yy,interp_method='linear')
    
    return dhdt, fig

def test_3():
    velx, vely, velxerr, velyerr = Topography.load_vel_measures('', xx, yy)
    return velx, vely, velxerr, velyerr

df = pd.read_csv('../Data/compiled_Denman_QC_55_shallow.csv')
df = df.rename(columns={"bed": "bed_notpreprocessed"})
df = df.rename(columns={"bedQCrf": "bed"})

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
xx.shape

velx, vely, velxerr, velyerr = test_3()
