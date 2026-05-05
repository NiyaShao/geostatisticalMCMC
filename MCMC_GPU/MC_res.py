"""
Created on Mond Mar 4

@author: tylerrleee
"""

import cupy as cp
import numpy as np

#  Raw CUDA kernel 
_mc_residual_kernel = cp.RawKernel(r'''
extern "C" __global__

void mc_residual(
                const double* __restrict__ bed,
                const double* __restrict__ surf,
                const double* __restrict__ velx,
                const double* __restrict__ vely,
                const double* __restrict__ dhdt,
                const double* __restrict__ smb,
                double* __restrict__ res,
                const int rows,
                const int cols,
                const double inv_h,      // 1.0 / resolution
                const double inv_2h      // 1.0 / (2.0 * resolution)
            ) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * cols) return;
 
    int r = idx / cols;
    int c = idx % cols;
 
    // ICE THICKNESS
    double thick = surf[idx] - bed[idx];
 
    // GRADIENT COLUMNS
    //  GET FLUX  | flux_x = velx * thick  →  d(flux_x)/dx along axis=1 (columns) 
    double dx_val;
    if (cols == 1) {
        dx_val = 0.0;
    } else if (c == 0) {
        // forward difference
        int idx_r = r * cols + 1;
        double fx_here = velx[idx] * thick;
        double fx_right = velx[idx_r] * (surf[idx_r] - bed[idx_r]);
        dx_val = (fx_right - fx_here) * inv_h;
    } else if (c == cols - 1) {
        // backward difference
        int idx_l = r * cols + (c - 1);
        double fx_here = velx[idx] * thick;
        double fx_left = velx[idx_l] * (surf[idx_l] - bed[idx_l]);
        dx_val = (fx_here - fx_left) * inv_h;
    } else {
        // central difference
        int idx_l = r * cols + (c - 1);
        int idx_r = r * cols + (c + 1);
        double fx_left  = velx[idx_l] * (surf[idx_l] - bed[idx_l]);
        double fx_right = velx[idx_r] * (surf[idx_r] - bed[idx_r]);
        dx_val = (fx_right - fx_left) * inv_2h;
    }
    
     // GRADIENT ROWS
    // GET FLUX flux_y = vely * thick  →  d(flux_y)/dy along axis=0 (rows) ─
    double dy_val;
    if (rows == 1) {
        dy_val = 0.0;
    } else if (r == 0) {
        // forward difference
        int idx_d = (1) * cols + c;
        double fy_here = vely[idx] * thick;
        double fy_down = vely[idx_d] * (surf[idx_d] - bed[idx_d]);
        dy_val = (fy_down - fy_here) * inv_h;
    } else if (r == rows - 1) {
        // backward difference
        int idx_u = (r - 1) * cols + c;
        double fy_here = vely[idx] * thick;
        double fy_up   = vely[idx_u] * (surf[idx_u] - bed[idx_u]);
        dy_val = (fy_here - fy_up) * inv_h;
    } else {
        // central difference
        int idx_u = (r - 1) * cols + c;
        int idx_d = (r + 1) * cols + c;
        double fy_up   = vely[idx_u] * (surf[idx_u] - bed[idx_u]);
        double fy_down = vely[idx_d] * (surf[idx_d] - bed[idx_d]);
        dy_val = (fy_down - fy_up) * inv_2h;
    }
 
    // RESIDUAL
    res[idx] = dx_val + dy_val + dhdt[idx] - smb[idx];
}
''', 'mc_residual')
 
 
def get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution):
    """
    Drop-in replacement for get_mass_conservation_residual_GPU.
    Single kernel launch instead of 6+.
 
    All inputs must be 2D CuPy float64 arrays of the same shape.
    resolution is a Python float or int.
    Returns a 2D CuPy float64 array of the same shape.
    """
    
    bed  = cp.asarray(bed, dtype=cp.float64)
    surf = cp.asarray(surf, dtype=cp.float64)
    velx = cp.asarray(velx, dtype=cp.float64)
    vely = cp.asarray(vely, dtype=cp.float64)
    dhdt = cp.asarray(dhdt, dtype=cp.float64)
    smb  = cp.asarray(smb, dtype=cp.float64)
    
    rows, cols = bed.shape
    n = rows * cols
 
    res = cp.empty_like(bed)
 
    inv_h  = 1.0 / float(resolution)
    inv_2h = 1.0 / (2.0 * float(resolution))
 
    threads = 256 # TODO dynamically change this based on available threads
    blocks = (n + threads - 1) // threads
 
    _mc_residual_kernel(
        (blocks,), (threads,),
        (bed, surf, velx, vely, dhdt, smb, res,
         np.int32(rows), np.int32(cols),
         np.float64(inv_h), np.float64(inv_2h))
    )
 
    return res
 
 
def get_mass_conservation_residual_fused_local(
    bed_local, surf_local, velx_local, vely_local,
    dhdt_local, smb_local, resolution
):
    """
    Same as above but for small local blocks extracted from the full grid.
    Ensures contiguous memory layout before launching the kernel.
    """
    bed_local  = cp.asarray(bed_local, dtype=cp.float64)
    surf_local = cp.asarray(surf_local, dtype=cp.float64)
    velx_local = cp.asarray(velx_local, dtype=cp.float64)
    vely_local = cp.asarray(vely_local, dtype=cp.float64)
    dhdt_local = cp.asarray(dhdt_local, dtype=cp.float64)
    smb_local  = cp.asarray(smb_local, dtype=cp.float64)
    
    # Slices from CuPy arrays may not be contiguous
    bed_local  = cp.ascontiguousarray(bed_local)
    surf_local = cp.ascontiguousarray(surf_local)
    velx_local = cp.ascontiguousarray(velx_local)
    vely_local = cp.ascontiguousarray(vely_local)
    dhdt_local = cp.ascontiguousarray(dhdt_local)
    smb_local  = cp.ascontiguousarray(smb_local)
 
    return get_mass_conservation_residual_fused(
        bed_local, surf_local, velx_local, vely_local,
        dhdt_local, smb_local, resolution
    )