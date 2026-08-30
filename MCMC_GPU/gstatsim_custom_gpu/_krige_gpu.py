"""
krige_gpu.py

Core solvers for Kriging systems on GPU.
OPTIMIZED for high-performance computing.

Supports:
- Ordinary Kriging (OK): Assumes unknown constant mean (sum weights = 1).
- Simple Kriging (SK): Assumes known global mean.

It uses batch linear algebra (cp.linalg.solve / Cholesky) to solve thousands of
kriging systems simultaneously.
"""
import math
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from .covariance_gpu import batch_covariance_gpu

# -----------------------------------------------------------------------------
# Helper: Optimized Pairwise Distance (Memory Efficient)
# -----------------------------------------------------------------------------
def _pairwise_distance_squared_matmul(X, Y, dtype=cp.float64):
    """
    Compute squared Euclidean distances using Matrix Multiplication (GEMM).
    Uses identity: |X-Y|^2 = |X|^2 + |Y|^2 - 2(X @ Y.T)
    
    This avoids creating the massive (B, M, N, D) difference tensor which
    causes OOM errors on large batches.
    """
    X = X.astype(dtype)
    Y = Y.astype(dtype)
    
    # X: (B, M, D), Y: (B, N, D)
    # 1. Compute squared norms
    X_sq = cp.sum(X**2, axis=2, keepdims=True) # (B, M, 1)
    Y_sq = cp.sum(Y**2, axis=2, keepdims=True) # (B, N, 1)
    Y_sq_T = Y_sq.transpose(0, 2, 1)           # (B, 1, N)

    # 2. Compute dot product (The heavy lifting, highly optimized on GPU)
    # We transpose Y at the last two dims: (B, N, D) -> (B, D, N)
    dot = cp.matmul(X, Y.transpose(0, 2, 1))

    # 3. Combine
    dist_sq = X_sq + Y_sq_T - 2 * dot
    
    # Clip negative values (numerical noise around 0)
    return cp.maximum(dist_sq, dtype(0.0))

def make_rotation_matrix_gpu(azimuth, major_range, minor_range, dtype=cp.float64):
    """
    Constructs the rotation/scaling matrix for anisotropic variograms.
    Coordinates are rotated and scaled so that distance becomes isotropic.
    """
    # Helper to handle scalar/array inputs
    def _to_float(x):
        if hasattr(x, 'item'): return float(x.item())
        if hasattr(x, '__len__') and len(x) == 1: return float(x[0])
        return float(x)

    azimuth = _to_float(azimuth)
    major_range = _to_float(major_range)
    minor_range = _to_float(minor_range)

    theta = (azimuth / 180.0) * math.pi
    
    # Rotation matrix (2D)
    rot = cp.array([[math.cos(theta), -math.sin(theta)], 
                    [math.sin(theta),  math.cos(theta)]], dtype=dtype)
    
    # Scaling matrix (normalize ranges to 1.0)
    scale = cp.array([[1.0 / major_range, 0.0],
                      [0.0, 1.0 / minor_range]], dtype=dtype)
    
    return rot.dot(scale)

def batch_ok_solve_gpu(sim_points, neighbors_array, vario, jitter_rel=1e-6, dtype=cp.float64):
    """
    Solves the Ordinary Kriging system for a batch of points.
    
    System:
    [ Sigma   1 ] [ w ]   [ rho ]
    [ 1^T     0 ] [ u ] = [  1  ]

    Parameters:
    -----------
    sim_points : array (B, 2)
        Target coordinates.
    neighbors_array : array (B, K, 5)
        Neighbor data [x, y, val, i, j].
    vario : dict
        Variogram parameters.

    Returns:
    --------
    est : array (B,)
        Kriging estimates (mean).
    var : array (B,)
        Kriging variances.
    """
    def _get_scalar(key, default):
        v = vario.get(key, default)
        
        # 1. Handle CuPy / NumPy arrays 
        if hasattr(v, 'ravel'):
            return float(v.ravel()[0])
            
        # 2. Handle standard Python lists or tuples
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return float(v[0])
            
        # 3. Handle isolated numpy/cupy scalars (e.g., np.float64)
        if hasattr(v, 'item'):
            return float(v.item())
            
        # 4. Standard Python floats/ints
        return float(v)
    
    B, K, _ = neighbors_array.shape
    if K == 0 or B == 0:
        return cp.empty((0,), dtype=dtype), cp.empty((0,), dtype=dtype)

    coords = neighbors_array[:, :, 0:2].astype(dtype) # (B,K,2)
    vals = neighbors_array[:, :, 2].astype(dtype)     # (B,K)
    
    valid = cp.isfinite(vals)
    mfloat = valid.astype(dtype)
    
    # Masking: Replace NaNs in coords with 0 to prevent propagation in dot products
    coords = cp.where(valid[..., None], coords, dtype(0.0))

    # 1. Anisotropy handling
    """rot = make_rotation_matrix_gpu(vario.get('azimuth', 0.0), 
                                   vario.get('major_range', 1.0), 
                                   vario.get('minor_range', 1.0), dtype=dtype)"""
    rot = make_rotation_matrix_gpu(_get_scalar('azimuth', 0.0), 
                                   _get_scalar('major_range', 1.0), 
                                   _get_scalar('minor_range', 1.0), 
                                   dtype=dtype)
    
    # Apply rotation to all coords efficiently
    coords_r = coords @ rot.T    # (B,K,2)
    sim_r = sim_points.astype(dtype) @ rot.T # (B,2)

    # 2. Build LHS Matrix (Covariance between neighbors)
    # OPTIMIZATION: Use Matmul for distances (Saves 50%+ Memory)
    dist_sq = _pairwise_distance_squared_matmul(coords_r, coords_r, dtype=dtype)
    NR = cp.sqrt(dist_sq)

    vtype = vario.get('vtype', 'matern').lower()
    
    """sill = float(vario.get('sill', 1.0))
    nugget = float(vario.get('nugget', 0.0))
    s = float(vario.get('s', 1.5))"""
    
    sill = _get_scalar('sill', 1.0)
    nugget = _get_scalar('nugget', 0.0)
    s = _get_scalar('s', 1.5)

    Sigma = batch_covariance_gpu(NR, vtype, sill, nugget, 
                                 batch_size=max(2_000_000, NR.size // 2), s=s, dtype=dtype)

    # Mask invalid entries (where neighbors were NaN)
    valid_mask = mfloat[:, :, None] * mfloat[:, None, :]
    Sigma = Sigma * valid_mask

    # Regularization (nugget effect on diagonal + jitter) to ensure stability
    maxS = cp.max(cp.abs(Sigma), axis=(1,2), keepdims=True)
    eps = jitter_rel * (maxS + dtype(1.0))
    eye_batch = cp.eye(K, dtype=dtype)[None, :, :]
    
    Sigma = Sigma + eps * eye_batch

    # 3. Assemble Augmented Matrix A
    A = cp.zeros((B, K+1, K+1), dtype=dtype)
    A[:, :K, :K] = Sigma
    A[:, :K, K] = mfloat  # Lagrange multipliers column
    A[:, K, :K] = mfloat  # Lagrange multipliers row

    # 4. Build RHS Vector b (Covariance between target and neighbors)
    # OPTIMIZATION: Matmul for vector-matrix distance
    # sim_r is (B, 2), treat as (B, 1, 2)
    dist_sq_sim = _pairwise_distance_squared_matmul(coords_r, sim_r[:, None, :], dtype=dtype)
    nr = cp.sqrt(dist_sq_sim[:, :, 0]) # Squeeze last dim

    rho = batch_covariance_gpu(nr, vtype, sill, nugget, 
                               batch_size=max(1_000_000, nr.size), s=s, dtype=dtype)
    rho = rho * mfloat

    b = cp.zeros((B, K+1), dtype=dtype)
    b[:, :K] = rho
    b[:, K] = dtype(1.0) # Constraint sum(weights) = 1

    # 5. Solve linear system Ax = b
    # Note: cp.linalg.solve uses Batched LU (getrf/getrs) from cuSOLVER
    try:
        x = cp.linalg.solve(A, b[:, :, None])[:, :, 0]
    except cp.linalg.LinAlgError:
        # Fallback: add more noise to diagonal if singular
        A[:, :K, :K] += dtype(1e-5) * eye_batch
        x = cp.linalg.solve(A, b[:, :, None])[:, :, 0]
        
    w = x[:, :K] # Weights
    
    # Normalize weights for numerical safety
    wsum = cp.sum(w, axis=1, keepdims=True)
    w = cp.where(cp.abs(wsum) > 0, w / wsum, w)

    # 6. Compute Estimate and Variance
    vals_safe = cp.where(valid, vals, dtype(0.0))
    est = cp.sum(w * vals_safe, axis=1)
    var = dtype(sill) - cp.sum(w * rho, axis=1)

    return est, var

def _solve_group_sk(sim_points, neighbors_array, vario, global_mean, dtype=cp.float64):
    """
    Helper for Simple Kriging (single group size).
    OPTIMIZATION: Uses Cholesky Decomposition for 2x speedup over LU.
    """
    B, K, _ = neighbors_array.shape
    
    coords = neighbors_array[:, :, 0:2].astype(dtype)
    vals = neighbors_array[:, :, 2].astype(dtype)

    rot = make_rotation_matrix_gpu(vario.get('azimuth', 0.0), 
                                   vario.get('major_range', 1.0), 
                                   vario.get('minor_range', 1.0), dtype=dtype)
    
    coords_r = coords @ rot.T
    sim_r = sim_points.astype(dtype) @ rot.T

    # Compute Sigma (Covariance Matrix)
    dist_sq = _pairwise_distance_squared_matmul(coords_r, coords_r, dtype=dtype)
    NR = cp.sqrt(dist_sq)

    vtype = vario.get('vtype', 'matern').lower()
    sill = vario.get('sill', 1.0)
    nugget = vario.get('nugget', 0.0)
    s = vario.get('s', 1.5)

    Sigma = batch_covariance_gpu(NR, vtype, sill, nugget, s=s, dtype=dtype)

    # Regularization
    eps = dtype(1e-6) * (cp.max(cp.abs(Sigma)) + dtype(1.0))
    Sigma = Sigma + cp.eye(K, dtype=dtype)[None,:,:] * eps

    # Compute rho (RHS)
    dist_sq_sim = _pairwise_distance_squared_matmul(coords_r, sim_r[:, None, :], dtype=dtype)
    nr = cp.sqrt(dist_sq_sim[:, :, 0])

    rho = batch_covariance_gpu(nr, vtype, sill, nugget, s=s, dtype=dtype)

    # Solve SK: w = Sigma^-1 * rho
    # OPTIMIZATION: Cholesky Solve (Sigma is Positive Definite for SK)
    # L * L.T * w = rho
    try:
        L = cp.linalg.cholesky(Sigma)
        # y = solve(L, rho)
        y = solve_triangular(L, rho[:, :, None], lower=True)
        # w = solve(L.T, y)
        w = solve_triangular(L.transpose(0, 2, 1), y, lower=False)[:, :, 0]
    except cp.linalg.LinAlgError:
        # Fallback to standard solver if not PD (numerical noise)
        w = cp.linalg.solve(Sigma, rho[:, :, None])[:, :, 0]

    # SK Estimate: m + w*(obs - m)
    global_mean = dtype(global_mean)
    est = global_mean + cp.sum(w * (vals - global_mean), axis=1)
    var = dtype(sill) - cp.sum(w * rho, axis=1)

    return est, var

def _group_by_counts(neighbors_array, counts):
    """Groups batch items by number of valid neighbors for SK processing."""
    counts_cpu = cp.asnumpy(counts)
    groups = {}
    for idx, kcnt in enumerate(counts_cpu):
        if kcnt <= 0: continue
        groups.setdefault(int(kcnt), []).append(int(idx))
    return [(K, cp.asarray(rows, dtype=cp.int32)) for K, rows in groups.items()]

def batch_sk_solve_gpu(sim_points, neighbors_array, vario, global_mean, batch_size=4096, dtype=cp.float64):
    """Solves Simple Kriging by grouping systems of the same size."""
    valid = ~cp.isnan(neighbors_array[:, :, 2])
    counts = cp.sum(valid, axis=1).astype(cp.int32)
    
    if not cp.any(counts > 0):
        return cp.zeros((0,), dtype=dtype), cp.zeros((0,), dtype=dtype)
        
    groups = _group_by_counts(neighbors_array, counts)
    
    all_est = cp.empty((neighbors_array.shape[0],), dtype=dtype)
    all_var = cp.empty_like(all_est)
    
    for K, rows in groups:
        neigh = neighbors_array[rows, :K, :]
        sims = sim_points[rows]
        est, var = _solve_group_sk(sims, neigh, vario, global_mean, dtype=dtype)
        all_est[rows] = est
        all_var[rows] = var
        
    return all_est, all_var
