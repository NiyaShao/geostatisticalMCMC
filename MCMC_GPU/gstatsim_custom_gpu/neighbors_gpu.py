"""
neighbors_gpu.py

Efficient nearest-neighbor search on structured grids using GPU.
Instead of computing the distance from every target point to every conditioning point,
this module uses a sliding window (stencil) approach. It only searches for neighbors
within the 'radius' defined by the variogram/user.
"""

import cupy as cp
import math

def _cond_coords_from_mask(xx, yy, grid, cond_msk=None):
    """Extract coordinates and values of conditioning data based on a mask."""
    if cond_msk is None:
        cond_msk = ~cp.isnan(grid)
        
    cond_ii, cond_jj = cp.where(cond_msk)
    
    if cond_ii.size == 0:
        return cp.empty((0, 2)), cp.empty((0,)), cond_ii, cond_jj
        
    cond_x = xx[cond_ii, cond_jj]
    cond_y = yy[cond_ii, cond_jj]
    vals = grid[cond_ii, cond_jj]
    
    coords = cp.stack([cond_x, cond_y], axis=1)
    return coords, vals, cond_ii, cond_jj

def batch_neighbors_distance_based(batch_inds, ii, jj, xx, yy, grid, cond_msk,
                                   radius, num_points, max_memory_gb=20.0,
                                   dtype=cp.float64):
    """
    OPTIMIZED neighbor search for a batch of target points.

    Parameters:
    -----------
    batch_inds : array (B, 2)
        Indices (i, j) of the points to simulate/interpolate.
    xx, yy : array
        Coordinate grids.
    grid : array
        Data grid containing conditioning data (and NaNs).
    radius : float
        Search radius in physical units.
    num_points : int
        Max number of neighbors to find (K).
    dtype : cp.dtype
        Precision for distance calculations.

    Returns:
    --------
    neigh : array (B, K, 5)
        [x, y, value, i, j] for each neighbor.
    nb_counts : array (B,)
        Actual number of neighbors found for each point.
    """
    
    # SAFETY: Ensure we don't carry huge tensors if the batch is massive
    # The existing implementation is actually quite memory efficient
    # because it calculates 'dists' only on the window, not the whole grid.
    
    # Just ensure batch_inds is int32 for index speed
    batch_inds = batch_inds.astype(cp.int32)
    B = batch_inds.shape[0]
    K = int(num_points)
    
    grid_rows, grid_cols = grid.shape
    
    if B == 0:
        return cp.empty((0, K, 5), dtype=dtype), cp.zeros(0, dtype=cp.int32)

    # --- 1. Define Search Window ---
    # Calculate how many grid cells correspond to the radius
    dx = cp.abs(xx[0, 1] - xx[0, 0])
    W = int(cp.ceil(radius / dx).get())

    # Create relative indices for the search window (box)
    di, dj = cp.meshgrid(cp.arange(-W, W + 1, dtype=cp.int32),
                         cp.arange(-W, W + 1, dtype=cp.int32),
                         indexing='ij')
    
    # Filter box to a circle to reduce candidate points by ~21%
    stencil_dist_sq = di.ravel()**2 + dj.ravel()**2
    keep_mask = stencil_dist_sq <= W**2
    
    di_flat = di.ravel()[keep_mask]
    dj_flat = dj.ravel()[keep_mask]
    window_size = di_flat.size

    # --- 2. Gather Local Data for the Batch ---
    batch_i = batch_inds[:, 0]
    batch_j = batch_inds[:, 1]
    
    # Broadcast to create absolute indices for all windows at once
    # Shapes: (B, 1) + (1, WinSize) -> (B, WinSize)
    window_i = batch_i[:, None] + di_flat[None, :]
    window_j = batch_j[:, None] + dj_flat[None, :]
    
    # Clip to grid bounds (handle edges)
    cp.clip(window_i, 0, grid_rows - 1, out=window_i)
    cp.clip(window_j, 0, grid_cols - 1, out=window_j)

    # --- 3. Find Neighbors in Local Windows ---
    local_vals = grid[window_i, window_j]
    local_cond = cond_msk[window_i, window_j]
    
    # Identify valid neighbors (must be conditioning data and not NaN)
    is_valid_neighbor = local_cond & ~cp.isnan(local_vals)

    # Calculate physical distances
    target_x = xx[batch_i, batch_j]
    target_y = yy[batch_i, batch_j]
    local_x = xx[window_i, window_j]
    local_y = yy[window_i, window_j]
    
    # Use brute-force on window (usually faster than KDTree on GPU for fixed grid stencils)
    dist_sq = (target_x[:, None] - local_x)**2 + (target_y[:, None] - local_y)**2
    dists = cp.sqrt(dist_sq)
    
    # Set invalid points to infinity so they are sorted last
    inf_val = cp.finfo(dtype).max if dtype == cp.float32 else cp.inf
    masked_dists = cp.where(is_valid_neighbor & (dists <= radius), dists, inf_val)

    # --- 4. Partition and Sort to Get Top K ---
    K_eff = min(K, window_size)
    
    # argpartition puts the smallest K elements first (unsorted)
    partition_idx = cp.argpartition(masked_dists, K_eff - 1, axis=1)[:, :K_eff]
    
    # Sort the top K to order them by distance
    top_k_dists = cp.take_along_axis(masked_dists, partition_idx, axis=1)
    sort_order = cp.argsort(top_k_dists, axis=1)
    
    final_window_indices = cp.take_along_axis(partition_idx, sort_order, axis=1)

    # --- 5. Assemble Output Array ---
    neigh = cp.full((B, K, 5), cp.nan, dtype=dtype)
    
    # Helper for fast gather
    def gather(arr): return cp.take_along_axis(arr, final_window_indices, axis=1)

    # Extract data using the sorted indices
    final_x = gather(local_x)
    final_y = gather(local_y)
    final_vals = gather(local_vals)
    final_ii = gather(window_i)
    final_jj = gather(window_j)
    final_sorted_dists = gather(top_k_dists)
    
    # Final check for validity (dists < inf)
    valid_mask = final_sorted_dists < (inf_val / 2) # Check against huge value

    # Fill output: neigh = [x, y, value, i, j]
    neigh[:, :K_eff, 0] = cp.where(valid_mask, final_x, cp.nan)
    neigh[:, :K_eff, 1] = cp.where(valid_mask, final_y, cp.nan)
    neigh[:, :K_eff, 2] = cp.where(valid_mask, final_vals, cp.nan)
    neigh[:, :K_eff, 3] = cp.where(valid_mask, final_ii.astype(dtype), cp.nan)
    neigh[:, :K_eff, 4] = cp.where(valid_mask, final_jj.astype(dtype), cp.nan)
    
    nb_counts = cp.sum(valid_mask, axis=1).astype(cp.int32)
    
    return neigh, nb_counts

def make_circle_stencil_gpu_safe(x, rad, max_memory_gb=1.0):
    """Pre-compute a circular boolean mask for neighbor searching."""
    dx = cp.abs(x[1] - x[0])
    if dx == 0: return None, None, None
    
    ncells = int(cp.ceil(rad / dx).get())
    x_stencil = cp.linspace(-rad, rad, 2 * ncells + 1)
    xx_st, yy_st = cp.meshgrid(x_stencil, x_stencil)
    distances = cp.sqrt(xx_st**2 + yy_st**2)
    
    stencil = distances < rad
    return stencil, xx_st, yy_st

def to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def to_cpu(*arrays):
    return [cp.asnumpy(a) for a in arrays]
