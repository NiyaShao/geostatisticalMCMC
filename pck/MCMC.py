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

class RandField:
    #sample by field perturbation
    def get_edge_mask(self,min_block_w, max_block_w, min_block_h, max_block_h, logistic_param, maxdist, res, num_step = 5):
        
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
            dist_edge = RandField.min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
            dist_rescale_edge = RandField.rescale(dist_edge, maxdist)
            dist_logi_edge = RandField.logistic(dist_rescale_edge, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]
            edge_masks.append(dist_logi_edge)

        return pairs, edge_masks
    
    def __init__(self,block_sizes,range_max_x,range_max_y,range_min_x,range_min_y,step_min,step_max,nugget_max,random_field_model,isotropic,rng=np.random.default_rng()):
        
        self.block_sizes
        self.range_max_x = range_max_x
        self.range_max_y = range_max_y
        self.range_min_x = range_min_x
        self.range_min_y = range_min_y
        self.step_min = step_min
        self.step_max = step_max
        self.nugget_max = nugget_max
        self.random_field_model = random_field_model
        self.isotropic = isotropic
        self.rng = rng
           
    def set_block_sizes(self,min_block_x,max_block_x,min_block_y,max_block_y,steps=5):
        self.min_block_x = min_block_x
        self.min_block_y = min_block_y
        self.max_block_x = max_block_x
        self.max_block_y = max_block_y
        self.steps = steps

        self.pairs = self.get_block_sizes()
        
    def set_block_param(self, logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution):
        # TODO set block sizes must be done before set block param, need to check that
        
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
            dist_rescale_edge = RandField.rescale(dist_edge, self.maxdist)
            dist_logi_edge = RandField.logistic(dist_rescale_edge, self.logistic_param[0], self.logistic_param[1], self.logistic_param[2]) - self.logistic_param[3]
            edge_masks.append(dist_logi_edge)

        return edge_masks
    
    #generate random Gaussian field
    def get_random_field(self,X,Y,_mean=0,_var=1):
        
        
        rng = self.rng
        scale  = rng.uniform(low=self.step_min, high=self.step_max, size=1)[0]/3
        nug = rng.uniform(low=0.0, high=self.nugget_max, size=1)[0]
        
        if not self.isotropic:
            range1 = rng.uniform(low=self.range_min_x, high=self.range_max_x, size=1)[0]
            range2 = rng.uniform(low=self.range_min_y, high=self.range_max_x, size=1)[0]
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

    def get_crf_weight(self,xx,yy,cond_data_mask,max_dist):
        logistic_param = self.logistic_param
        dist = RandField.min_dist(np.where(cond_data_mask==0, np.nan, 1), xx, yy)
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi

    def get_crf_weight_from_dist(xx,yy,logistic_param,cond_data_mask,max_dist,dist):
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi
    
        
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
    

class chain:

    def __init_func__(self):
        return
    
    def __init__(self, xx, yy, bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, resolution):
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
        self.resolution = resolution
        self.loss_function_list = []
        
        if (bed.shape!=surf.shape) or (bed.shape!=velx.shape) or (bed.shape!=vely.shape) or (bed.shape!=dhdt.shape) or (bed.shape!=smb.shape) or (bed.shape!=cond_bed.shape) or (bed.shape!=data_mask.shape):
            raise Exception('the shape of bed, surf, velx, vely, dhdt, smb, radar_bed, data_mask need to be same')
        
        self.__init_func__()
    
    def set_high_vel_region(self, update_in_region, region_mask = []):
        if update_in_region == True:
            if region_mask.shape != self.xx.shape:
                raise ValueError('the region_mask input is invalid. It has to be a 2D numpy array with the shape of the map')
            else:
                self.region_mask = region_mask
        else:
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


    def set_loss_type(self, map_func = None, diff_func = None, sigma_mc = 0, sigma_data = 0, massConvInRegion = True, dataDiffInRegion = False):
    
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
            
        self.sigma_mc = sigma_mc
        self.sigma_data = sigma_data
        self.map_func = map_func
        self.diff_func = diff_func
    
        if map_func == 'meanabs':
            function_list.append(chain_crf.__meanabs)
        elif map_func == 'meansquare':
            function_list.append(chain_crf.__meansq)
        elif map_func == 'sumabs':
            function_list.append(chain_crf.__sumabs)
        elif map_func == 'sumsquare':
            function_list.append(chain_crf.__sumsquare)
        elif map_func == None:
            function_list.append(chain_crf.__return0)
        else:
            raise Exception("the massConvInRegion parameter is not set to correct value.")
            
            
        if diff_func == 'meanabs':
            function_list.append(chain_crf.__meanabs)
        elif diff_func == 'meansquare':
            function_list.append(chain_crf.__meansq)
        elif diff_func == 'sumabs':
            function_list.append(chain_crf.__sumabs)
        elif diff_func == 'sumsquare':
            function_list.append(chain_crf.__sumsquare)
        elif map_func == None:
            function_list.append(chain_crf.__return0)
        else:
            raise Exception("the massConvInRegion parameter is not set to correct value.")
      
        self.loss_function_list = function_list
    
            
    def loss(self, massConvResidual, dataDiff):
        
        f1 = self.function_list[0]
        f2 = self.function_list[1]
            
        # TODO: is it inappropriate to use sum when the two loss have unequal number of grid cells?
        loss_mc = f1(massConvResidual, self.mc_region_mask) / (2*self.sigma_mc**2)
        loss_data = f2(dataDiff, (self.data_mask==1)&(self.data_region_mask==1)) / (2*self.sigma_data**2)
        
        return loss_mc + loss_data, loss_mc, loss_data
        
class chain_crf(chain):
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using chainname.set_update_in_region(region_mask) and chainname.set_high_vel_region(update_in_region)')
        print('please also set up the random field parameters and block type using chainname.set_update_type(self, block_type, block_size, RFparam)')
        print('then please set up the loss function using either chainname.set_loss_type or chainname.set_loss_func')
        return
    
    def set_update_type(self, block_type, block_size, RFparam):
        
        if block_type == 'rbf_CRF':
            print('The update block is set to conditional random field generated by rbf method (not implemented yet)')
        elif block_type == 'logi_CRF':
            print('The update block is set to conditional random field generated by calculating weights with logistic function')
        elif block_type == 'RF':
            print('The update block is set to Random Field')
        else:
            raise ValueError('')
            
        self.block_size = block_size
        self.block_type = block_type
        self.RFparam = RFparam
        
        return
    
        
    def run(self, n_iter, RF, rng=np.random.default_rng()):
            
        if not isinstance(RF, RandField):
            raise TypeError('The arugment "RF" has to be an object of the class RandField')
        
    #self.script = types.MethodType(script, self)   # replace the method
    #passing loss function as a parameter https://www.geeksforgeeks.org/python/passing-function-as-an-argument-in-python/       
        # initialize storage
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        loss_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        bed_cache = np.zeros((n_iter, self.xx.shape[0], self.xx.shape[1]))
        blocks_cache = np.full((n_iter, 4), np.nan)
        resampled_times = np.zeros(self.xx.shape)
        
        # should i have an additional property called initial_bed?
        bed_c = self.bed
        
        # initialize loss
        mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb)
        data_diff = self.bed - self.cond_bed
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_data_cache[0] = loss_prev_data
        loss_mc_cache[0] = loss_prev_mc
        step_cache[0] = False
        bed_cache[0] = bed_c
        
        # TODO, this should be stored inside the chain instead of generate it every time
        crf_weight, dist, dist_rescale, dist_logi = RF.get_crf_weight(self.xx,self.yy,self.data_mask,max_dist=RF.max_dist)

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

            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = np.max((0,int(indexx-block_size[1]/2)))
            bxmax = np.min((bed_c.shape[0],int(indexx+block_size[1]/2)))
            bymin = np.max((0,int(indexy-block_size[0]/2)))
            bymax = np.min((bed_c.shape[1],int(indexy+block_size[0]/2)))
            
            #TODO: Okay this is fine, the problem is more of the boundary of the high velocity region

            #find the index of the block side in the coordinate of the block
            mxmin = np.max([f.shape[0]-bxmax,0])
            mxmax = np.min([bed_c.shape[0]-bxmin,f.shape[0]])
            mymin = np.max([f.shape[1]-bymax,0])
            mymax = np.min([bed_c.shape[1]-bymin,f.shape[1]])
            
            #perturb
            if self.block_type == 'CRF_logi':
                perturb = f[mxmin:mxmax,mymin:mymax]*self.crf_weight[bxmin:bxmax,bymin:bymax]
            else:
                perturb = f[mxmin:mxmax,mymin:mymax]

            bed_next = bed_c.copy()
            bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb
            
            if self.update_in_region:
                bed_next = np.where(self.region_mask, bed_next, bed_c)
                
            mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb)
            data_diff = self.bed_next - self.cond_bed
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
           
            #make sure no bed elevation is greater than surface elevation
            block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - bed_next[bxmin:bxmax,bymin:bymax]
            block_region_mask = self.region_mask[bxmin:bxmax,bymin:bymax]
            
            if np.sum((block_thickness<=0)[block_region_mask==1]) > 0:
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
                loss_prev_data = loss_next_data
                loss_data_cache[i] = loss_next_data
                
                step_cache[i] = True
                resampled_times[bxmin:bxmax,bymin:bymax] += self.region_mask[bxmin:bxmax,bymin:bymax]
                
            else:
                loss_mc_cache[i] = loss_prev_mc
                loss_cache[i] = loss_prev
                loss_data_cache[i] = loss_prev_data
                step_cache[i] = False

            bed_cache[i,:,:] = bed
            
            if i%1000 == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} data loss: {loss_data_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache[np.max([0,i-1000]):i])/(np.min([i,1000]))}') #to window acceptance rate

        return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache

    
    
    
### Geostatistics

def fit_variogram(data, coords, roughness_region_mask, maxlag, n_lags=50, samples=0.6, subsample=100000, data_for_trans = []):

    if len(data_for_trans)==0:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=0,subsample=subsample).fit(data)
    else:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=0,subsample=subsample).fit(data_for_trans)
        
    transformed_data = nst_trans.transform(data)
    
    coords = coords[roughness_region_mask==1]
    values = transformed_data[roughness_region_mask==1].flatten()


#should be the parameter of the chain's class
#xx, yy
#bed, surf, velx, vely, dhdt, smb
#cond_bed, data_mask, resolution

#should be parameter of the run function
#n_iter, block_size, rfgen_param, random_field_model, isotropic,

#set_update_type(type,block_size,rfgen_param,random_field_model,isotropic)
# if type = 'rbf_CRF'
# if type = 'logi_CRF'
# if type = 'RF'

#set_update_region(within_region = True/False, region_mask = [])
# if within_region = True:
#   check region_mask

#set_loss_type(res = True/False, diff = True/False, map_func = defaultFunc, diff_func = defaultFunc)
#make it a wrapper of the general MCMC loss, I mean if people want use it for other MCMC they can use pyMC

#passing loss function as a parameter https://www.geeksforgeeks.org/python/passing-function-as-an-argument-in-python/


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
        
    def set_normal_transformation(self, nst_trans):
        self.nst_trans = nst_trans

    # TODO, make it an option to detrend or not        
    def set_trend(self, simulate_detrended_map, trend):
        self.trend = trend
    
    def set_variogram(self, vario_type, vario_range, vario_sill, vario_nugget, isotropic = True, vario_smoothness = None, vario_azimuth = None):
        
        if (vario_type == 'Gaussian') & (vario_type == 'Exponential') & (vario_type == 'Spherical'):
            print('the variogram is set to type', vario_type)
        elif vario_type == 'Matern':
            if (vario_smoothness == None) or (vario_smoothness <= 0):
                raise ValueError('vario_smoothness argument should be a positive float when the vario_type is Matern')
        else:
            raise ValueError('vario_type argument should be one of the following: Gaussian, Exponential, Spherical, or Matern')
        
        self.vario_type = vario_type
        
        if isotropic:
            vario_azimuth = 0
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type]
        else:
            if (len(vario_range) == 2):
                print('set to anistropic variogram with major range and minor range to be ', vario_range)
                self.vario_param = [vario_azimuth, vario_nugget, vario_range[0], vario_range[1], vario_sill, vario_type]
            else:
                raise ValueError ("vario_range need to be a list with two floats to specifying for major range and minor range of the variogram when isotropic is set to False")
        
        self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_sill, vario_smoothness]
        
    def set_sgs_param(self, sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on = False, dropout_rate = 0):
        
        self.sgs_param = [sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on, dropout_rate]
    
    def set_blocks(self, block_min_x, block_min_y, block_max_x, block_max_y):
        self.block_min_x = block_min_x
        self.block_min_y = block_min_y
        self.block_max_x = block_max_x
        self.block_max_y = block_max_y

    def run(self, n_iter, rng=np.random.default_rng()):
        
        xmin = np.min(self.xx)
        xmax = np.max(self.xx)
        ymin = np.min(self.yy)
        ymax = np.max(self.yy)
        
        rows = self.xx.shape[0]
        cols = self.xx.shape[1]
        
        loss_cache = np.zeros(n_iter)
        loss_mc_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        bed_cache = np.zeros((n_iter, rows, cols))
        blocks_cache = np.full((n_iter, 4), np.nan)
        
        # TODO, should i have an additional property called initial_bed?
        bed_c = self.bed
        
        # TODO, when should I input in trend
        trend = self.trend
        
        # TODO, nst_trans
        nst_trans = self.nst_trans
        
        # z = bed_c.flatten()
        z = nst_trans.inverse_transform(bed_c.reshape(-1,1)).reshape(rows,cols) + trend
        
        resolution = self.resolution
    
        df_data = np.array([self.xx.flatten(),self.yy.flatten(),z,self.data_mask.flatten(),self.mc_region_mask.flatten()])
        psimdf = pd.DataFrame(df_data, columns={'x','y','z','data_mask','mc_region_mask'})
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
        step_cache[0] = False
        bed_cache[0] = bed_c
        
        for i in range(n_iter):
            
            # TODO here bias toward resample locations that do not have measurements
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
            
            if self.rand_dropout == True:
                
                intersect_index = resampling_box_index.intersection(data_index)
                intersect_index = self.rng.choice(intersect_index, size=int(intersect_index.shape[0]*(1-self.dropoutrate)), replace=False)
                
                drop_index = resampling_box_index.difference(intersect_index)
                
            else:
                
                drop_index = resampling_box_index.difference(data_index)

            new_df = psimdf[~psimdf.index.isin(drop_index)].copy()

            Pred_grid_xy_change = gs.Gridding.prediction_grid(rsm_x_min, rsm_x_max - resolution, rsm_y_min, rsm_y_max - resolution, resolution)
            x = np.reshape(Pred_grid_xy_change[:,0], (len(Pred_grid_xy_change[:,0]), 1))
            y = np.flip(np.reshape(Pred_grid_xy_change[:,1], (len(Pred_grid_xy_change[:,1]), 1)))
            Pred_grid_xy_change = np.concatenate((x,y),axis=1)

            sim2 = gs.Interpolation.okrige_sgs(Pred_grid_xy_change, new_df, 'x', 'y', 'z', self.sgs_param[0], self.variogram, self.sgs_param[1], quiet=True) 

            xy_grid = np.concatenate((Pred_grid_xy_change[:,0].reshape(-1,1),Pred_grid_xy_change[:,1].reshape(-1,1),np.array(sim2).reshape(-1,1)),axis=1)

            psimdf_next = psimdf.copy()
            psimdf_next.loc[resampling_box_index,['x','y','z']] = xy_grid
            bed_next = nst_trans.inverse_transform(np.array(psimdf_next['z']).reshape(-1,1)).reshape(rows,cols) + trend
            
            mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb)
            data_diff = bed_next - self.cond_bed
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
            
            #make sure no bed elevation is greater than surface elevation
            thickness = self.surf - bed_next
            
            if np.sum((thickness<=0)[self.mc_region_mask==1]) > 0:
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
                
            bed_cache[i,:,:] = bed_c

            if i % 1000 == 0:
                print(f'i: {i} mc loss: {loss_mc_cache[i]:.3e} loss: {loss_cache[i]:.3e} acceptance rate: {np.sum(step_cache)/(i+1)}')

        resampled_times = psimdf.resampled_times.values.reshape((rows,cols))
                
        return bed_cache, loss_mc_cache, loss_cache, step_cache, resampled_times, blocks_cache
        