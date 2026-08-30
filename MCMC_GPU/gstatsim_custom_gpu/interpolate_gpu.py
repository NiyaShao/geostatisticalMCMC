"""
interpolate_gpu.py

Main entry point for GPU-accelerated Geostatistics.

OPTIMIZED for high-VRAM systems (e.g., A100/H100).

Functions:
- krige_gpu: Perform Simple or Ordinary Kriging estimation.
- sgs_gpu: Perform Sequential Gaussian Simulation (SGS).

Both functions handle:
1. Normal Score Transformation
2. Batch processing (to fit in GPU memory)
3. Neighbor search
4. Solving Kriging systems
5. Back-transformation
"""

import cupy as cp
from copy import deepcopy
import numbers
from tqdm import tqdm
from .utilities_gpu import gaussian_transformation_gpu, get_random_generator_gpu
from ._krige_gpu import batch_ok_solve_gpu, batch_sk_solve_gpu
from .neighbors_gpu import batch_neighbors_distance_based, make_circle_stencil_gpu_safe

def _sanity_checks_gpu(xx, yy, grid, vario, radius, num_points, ktype):
    """Validate inputs before starting."""
    if not isinstance(xx, cp.ndarray) or xx.ndim != 2:
        raise ValueError('xx must be a 2D CuPy array')
    if not isinstance(yy, cp.ndarray) or yy.ndim != 2:
        raise ValueError('yy must be a 2D CuPy array')
    
    expected = ['major_range', 'minor_range', 'azimuth','sill','nugget','vtype']
    for k in expected:
        if k not in vario:
            raise ValueError(f"Missing variogram key {k}")
            
    if vario['vtype'].lower() == 'matern' and 's' not in vario:
        raise ValueError("Matern requires 's' parameter")

def _preprocess_gpu_safe(xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb, dtype=cp.float64):
    """Common setup: Gaussian transform, index generation, and stencil creation."""
    cond_msk = ~cp.isnan(grid)

    # 1. Normal Score Transform
    out_grid, nst_trans = gaussian_transformation_gpu(grid, cond_msk, dtype=dtype)

    # 2. Determine simulation path (mask)
    if sim_mask is None:
        sim_mask = cp.full(xx.shape, True)

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')
    inds = cp.stack([ii[sim_mask].flatten(), jj[sim_mask].flatten()], axis=1)

    # 3. Clean Variogram dict
    vario = deepcopy(variogram)
    for k in vario:
        if isinstance(vario[k], numbers.Number):
            vario[k] = float(vario[k])
            
    global_mean = float(cp.mean(out_grid[cond_msk]))

    # 4. Prepare Stencil for neighbor search
    if stencil is None:
        stencil_res = make_circle_stencil_gpu_safe(xx[0, :], radius, max_memory_gb)
        stencil = stencil_res if stencil_res[0] is not None else None
        
    return out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil

def krige_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok', 
              sim_mask=None, quiet=False, stencil=None, max_memory_gb=150.0, batch_size=None,
              use_sector_balance=True, n_sectors=8, dtype=cp.float64):
    """
    Perform Kriging Estimation (Ordinary or Simple) on GPU.
    """
    _sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)

    # Setup
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess_gpu_safe(
        xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb, dtype=dtype
    )

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')

    # Determine Batch Size (Auto-calculate if None)
    if batch_size is None:
        bytes_per_point = num_points * 5 * 8
        batch_size = int((max_memory_gb * (1024**3)) // bytes_per_point)
        batch_size = max(16384, min(batch_size, inds.shape[0]))

    n_points = inds.shape[0]
    pbar = tqdm(total=n_points, desc="Kriging", unit="pts")

    # Process in Batches
    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, bstart + batch_size)
        batch_inds = inds[bstart:bend].astype(cp.int32)

        # 1. Find Neighbors
        neighbors, counts = batch_neighbors_distance_based(
            batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb, dtype=dtype
        )

        # 2. Filter Valid Points
        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], 
                               yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)
        
        valid_mask = counts > 0
        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue

        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        # 3. Solve Kriging System
        if ktype == 'ok':
            ests, vars_ = batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario, dtype=dtype)
        else:
            ests, vars_ = batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean, dtype=dtype)

        # 4. Store Results
        # Optimization: Vectorized write
        valid_indices = cp.where(valid_mask)[0]
        
        # Map local valid indices back to grid indices
        valid_rows = batch_inds[valid_indices, 0]
        valid_cols = batch_inds[valid_indices, 1]
        
        out_grid[valid_rows, valid_cols] = ests
        
        pbar.update(bend - bstart)

    pbar.close()

    # Back-transform to original distribution
    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)
    return sim_trans

def sgs_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok', 
            sim_mask=None, quiet=False, stencil=None, seed=None, max_memory_gb=150.0,
            batch_size=None, use_sector_balance=True, n_sectors=8, dtype=cp.float64):
    """
    Sequential Gaussian Simulation (SGS) on GPU.
    OPTIMIZED VERSION.
    
    The key difference from Kriging is that simulated values are added to the 
    conditioning data ('cond_msk') instantly, affecting subsequent points.
    """
    _sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)

    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess_gpu_safe(
        xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb, dtype=dtype
    )

    rng = cp.random.default_rng(seed)

    # Random Simulation Path
    shuffled = cp.copy(inds)
    cp.random.shuffle(shuffled)

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')

    # -------------------------------------------------------------------------
    # OPTIMIZATION: Robust Batch Size Calculation
    # -------------------------------------------------------------------------
    if batch_size is None:
        # Approximate VRAM usage per point (bytes):
        # 1. Neighbor Search: Stencil size * bytes (distances)
        # 2. Kriging Matrix: (K+1)^2 * bytes
        # 3. Intermediate tensors (coords, etc): K * 4*bytes
        
        bytes_per = 4 if dtype == cp.float32 else 8
        
        # Estimate stencil size (area of circle in pixels)
        dx = float(xx[0,1] - xx[0,0])
        stencil_pixels = int(3.14159 * (radius/dx)**2)
        
        mem_per_point = (stencil_pixels * bytes_per) + (num_points**2 * bytes_per) + (num_points * 128)
        
        # Use 80% of available memory safe margin
        avail_mem = max_memory_gb * (1024**3) * 0.8
        calc_batch = int(avail_mem // mem_per_point)
        
        # Clamp between reasonable limits (e.g., 4k to 200k)
        batch_size = max(4096, min(calc_batch, 200000))
        
        if not quiet:
            print(f"Calculated optimal batch size: {batch_size} (Stencil ~{stencil_pixels} px)")

    n_points = shuffled.shape[0]
    pbar = tqdm(total=n_points, desc="SGS", unit="pts")

    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, bstart + batch_size)
        batch_inds = shuffled[bstart:bend].astype(cp.int32)

        # 1. Find Neighbors (includes previously simulated points in this loop!)
        neighbors, counts = batch_neighbors_distance_based(
            batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb, dtype=dtype
        )

        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], 
                               yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)
        
        valid_mask = counts > 0
        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue

        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        # 2. Solve Kriging System
        if ktype == 'ok':
            ests, vars_ = batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario, dtype=dtype)
        else:
            ests, vars_ = batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean, dtype=dtype)

        # 3. Sample from Local Conditional Distribution (Normal)
        vars_safe = cp.abs(vars_)
        
        # Use random generator with correct dtype
        std_norm = rng.standard_normal(size=ests.shape, dtype=dtype)
        samp = std_norm * cp.sqrt(vars_safe) + ests

        # 4. Update Grid and Mask (Critical for SGS)
        # ---------------------------------------------------------------------
        # OPTIMIZATION: Vectorized Grid Update
        # ---------------------------------------------------------------------
        # Replaces the slow CPU loop. Directly writes to grid using fancy indexing.
        
        # Get the global grid indices (i, j) for the valid points in this batch
        # valid_mask filters the batch; batch_inds holds the (i,j)
        valid_rows = batch_inds[valid_mask, 0]
        valid_cols = batch_inds[valid_mask, 1]
        
        # Write values
        out_grid[valid_rows, valid_cols] = samp
        
        # Update conditioning mask
        cond_msk[valid_rows, valid_cols] = True
        
        pbar.update(bend - bstart)

    pbar.close()

    # --- SAFETY FIX: Clamp values to avoid Inf/NaN before inverse transform ---
    # float32 simulation can generate values slightly outside the fitted normal range
    # or effectively infinite due to precision issues.
    
    # Replace NaNs with 0.0 (mean of standard normal) or nearest valid neighbor if strictly needed
    # But for inverse_transform, 0.0 is safe (maps to median).
    out_grid = cp.nan_to_num(out_grid, nan=0.0, posinf=5.0, neginf=-5.0)
    
    # Clip to +/- 5 standard deviations (extremely generous for normal dist)
    # This prevents 'inf' which crashes cuML's inverse_transform
    out_grid = cp.clip(out_grid, -5.0, 5.0)

    # Back-transform to original distribution
    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)
    return sim_trans
