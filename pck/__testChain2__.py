#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:50:38 2025

@author: niyashao
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import Topography
import MCMC

resolution = 1e3

x = np.linspace(0, 10)*resolution
y = np.linspace(0, 10)*resolution
xxt, yyt = np.meshgrid(x,y)

range_max_x = 50e3 #in terms of meters in lateral dimension, regardless of resolution of the map
range_max_y = 50e3
range_min_x = 10e3
range_min_y = 10e3
step_min = 50 #in terms of meters in vertical dimension, how much you want to multiply the perturbation by
step_max = 200
nugget_max = 0 #short-scale roughness
random_field_model = 'Exponential'
isotropic = False
max_dist = 46e3
rf1 = MCMC.RandField(range_max_x, range_max_y, range_min_x, range_min_y, step_min, step_max, nugget_max, random_field_model, isotropic)

min_block_x = 10
max_block_x = 30
min_block_y = 10
max_block_y = 30
rf1.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)

logis_func_L = 2
logis_func_x0 = 0
logis_func_k = 6
logis_func_offset = 1
max_dist = 20e3
rf1.set_block_param(logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution)

f = rf1.get_random_field(x, y)

plt.pcolormesh(xxt,yyt,f)
plt.axis('scaled')