"""
utilities_gpu.py

Helper functions for:
1. Gaussian Transformation (Normal Score Transform) on GPU/CPU.
2. Calculating experimental variograms (wraps scikit-gstat).
3. Utility distance functions.
"""

import cupy as cp
import numpy as np
from cuml.preprocessing import QuantileTransformer as GPU_QuantileTransformer
from copy import deepcopy
import skgstat as skg

def gaussian_transformation_gpu(grid, cond_msk, n_quantiles=500, cpu_fit=False, cpu_transformer=None, random_state=0, dtype=cp.float64):
    """
    Apply Normal Score Transformation (Gaussian Anamorphosis).
    Transforms data to a standard normal distribution (mean=0, std=1), which is
    required for Multi-Gaussian Kriging/Simulation.

    Parameters:
    -----------
    grid : cp.ndarray
        The data grid.
    cond_msk : cp.ndarray
        Boolean mask of where data exists.
    cpu_fit : bool
        If True, use sklearn (CPU) transformer for exact reproducibility with CPU codes.
        If False, use cuML (GPU) transformer for speed.
    """
    data_cond = grid[cond_msk].reshape(-1, 1)
    
    if cpu_fit:
        if cpu_transformer is None:
            raise ValueError("cpu_fit=True requires cpu_transformer (sklearn) to be provided.")
        # Move to CPU, transform, move back
        data_cond_cpu = cp.asnumpy(data_cond)
        norm = cpu_transformer.transform(data_cond_cpu).squeeze()
        norm_gpu = cp.asarray(norm, dtype=dtype)
        nqt = cpu_transformer
    else:
        # Fit cuML (GPU) transformer
        # Note: We move to CPU to fit to ensure deterministic behavior if needed,
        # then move back. cuML can fit on GPU directly too.
        data_cond_cpu = cp.asnumpy(data_cond)
        nqt = GPU_QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=random_state
        ).fit(data_cond_cpu)
        
        norm = nqt.transform(data_cond_cpu).squeeze()
        norm_gpu = cp.asarray(norm, dtype=dtype)
    
    grid_norm = cp.full(grid.shape, cp.nan, dtype=dtype)
    grid_norm[cond_msk] = norm_gpu
    
    return grid_norm, nqt

def dists_to_cond_gpu(xx, yy, grid, dtype=cp.float64):
    """Calculate nearest distance to any conditioning data point for every grid cell."""
    cond_msk = ~cp.isnan(grid)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    n_points = xx_flat.size
    
    # Process in chunks to save memory
    chunk = 20000
    min_dists = cp.full(n_points, cp.inf, dtype=dtype)
    
    for i in range(0, n_points, chunk):
        end = min(n_points, i + chunk)
        xs = xx_flat[i:end][:, None].astype(dtype)
        ys = yy_flat[i:end][:, None].astype(dtype)
        
        dx = xs - x_cond[None, :]
        dy = ys - y_cond[None, :]
        d = cp.sqrt(dx**2 + dy**2, dtype=dtype)
        
        min_dists[i:end] = cp.min(d, axis=1)
        
    return min_dists.reshape(xx.shape)

def get_random_generator_gpu(seed):
    """Standardize random number generator creation."""
    if seed is None:
        return cp.random.default_rng()
    elif isinstance(seed, int):
        return cp.random.default_rng(seed=seed)
    else:
        return seed

def variograms_gpu(xx, yy, grid, bin_func='even', maxlag=100e3, n_lags=70, 
                   covmodels=['gaussian', 'spherical', 'exponential', 'matern'],
                   downsample=None, cpu_fit_transformer=None, use_cpu_transformer=False):
    """
    Compute experimental variograms using scikit-gstat (runs on CPU).

    Parameters:
    -----------
    xx, yy, grid : cp.ndarray
        Input data on GPU.
    covmodels : list
        List of theoretical models to fit to the experimental data.
    downsample : int
        If data is too large, take every Nth point to speed up calculation.

    Returns:
    --------
    vgrams : dict
        Parameters of fitted theoretical models.
    experimental : array
        The experimental semi-variance values.
    bins : array
        The lag distance bins.
    """
    cond_msk = ~cp.isnan(grid)
    
    # Transform data first
    grid_norm, nst_trans = gaussian_transformation_gpu(
        grid, cond_msk, cpu_fit=use_cpu_transformer, cpu_transformer=cpu_fit_transformer
    )
    
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_norm = grid_norm[cond_msk]
    
    coords_cond = cp.stack([x_cond, y_cond], axis=1)
    
    # Downsample if requested
    if isinstance(downsample, int):
        coords_cond = coords_cond[::downsample]
        data_norm = data_norm[::downsample]
        
    # Move to CPU for skgstat
    coords_cpu = cp.asnumpy(coords_cond)
    data_cpu = cp.asnumpy(data_norm)
    
    # Compute variogram
    V = skg.Variogram(coords_cpu, data_cpu, bin_func=bin_func, n_lags=n_lags, maxlag=maxlag, normalize=False)
    
    vgrams = {}
    for cov in covmodels:
        V_i = deepcopy(V)
        V_i.model = cov
        vgrams[cov] = V_i.parameters
        
    return vgrams, V.experimental, V.bins
