"""
besselk_gpu.py

This module implements the Modified Bessel function of the second kind (K_nu)
for GPU execution using CuPy.

It uses a raw CUDA kernel containing Temme's method and Chebyshev polynomial
expansions to ensure high precision and performance, which is necessary for
calculating Matern covariance functions on the GPU.
"""

import cupy as cp

# -------------------------------------------------------------------------
# CUDA Kernel Definition
# -------------------------------------------------------------------------
# This string contains the raw C++ CUDA code compiled at runtime.
# It defines the mathematical constants and iterative algorithms (Temme's method)
# required to evaluate Bessel K functions efficiently on the device.
_besselk_cuda_advanced = r'''
// CUDA device constants and data structures
struct cheb_series {
    const double* c;
    double a;
    double b;
    int order;
};

// Mathematical constants
#define M_PI 3.141592653589793238462643383279502884
#define M_LN10 2.302585092994045684017991454684364208
#define DBL_EPSILON 2.2204460492503131e-16
#define SQRT_DBL_MAX 1.3407807929942596e+154

// Define INFINITY and NAN for NVRTC (since math.h is not available)
#define INFINITY __int_as_float(0x7f800000)
#define NAN __int_as_float(0x7fc00000)

// Chebyshev coefficients from original implementation (SLATEC/GSL derived)
__constant__ double g1_dat[14] = {
    -1.14516408366268311786898152867,
    0.00636085311347084238122955495,
    0.00186245193007206848934643657,
    0.000152833085873453507081227824,
    0.000017017464011802038795324732,
    -6.4597502923347254354668326451e-07,
    -5.1819848432519380894104312968e-08,
    4.5189092894858183051123180797e-10,
    3.2433227371020873043666259180e-11,
    6.8309434024947522875432400828e-13,
    2.8353502755172101513119628130e-14,
    -7.9883905769323592875638087541e-16,
    -3.3726677300771949833341213457e-17,
    -3.6586334809210520744054437104e-20
};

__constant__ double g2_dat[15] = {
    1.882645524949671835019616975350,
    -0.077490658396167518329547945212,
    -0.018256714847324929419579340950,
    0.0006338030209074895795923971731,
    0.0000762290543508729021194461175,
    -9.5501647561720443519853993526e-07,
    -8.8927268107886351912431512955e-08,
    -1.9521334772319613740511880132e-09,
    -9.4003052735885162111769579771e-11,
    4.6875133849532393179290879101e-12,
    2.2658535746925759582447545145e-13,
    -1.1725509698488015111878735251e-15,
    -7.0441338200245222530843155877e-17,
    -2.4377878310107693650659740228e-18,
    -7.5225243218253901727164675011e-20
};

__constant__ int d_intervals = 128;

// Helper: Evaluate Chebyshev series
__device__ void cheb_eval_cuda(const double* c_data, int order, double a, double b, 
                               const double x, double* result) {
    int j;
    double d = 0.0;
    double dd = 0.0;
    double y = (2.0 * x - a - b) / (b - a);
    double y2 = 2.0 * y;
    
    for (j = order; j >= 1; j--) {
        double temp = d;
        d = y2 * d - dd + c_data[j];
        dd = temp;
    }
    
    d = y * d - dd + 0.5 * c_data[0];
    *result = d;
}

// Helper: Temme's gamma function approximation
__device__ void temme_gamma(const double nu, double* g_1pnu, double* g_1mnu, 
                            double* g1, double* g2) {
    double anu = fabs(nu);
    double x = 4.0 * anu - 1.0;
    double r_g1, r_g2;
    
    cheb_eval_cuda(g1_dat, 13, -1.0, 1.0, x, &r_g1);
    cheb_eval_cuda(g2_dat, 14, -1.0, 1.0, x, &r_g2);
    
    *g1 = r_g1;
    *g2 = r_g2;
    *g_1mnu = 1.0 / (r_g2 + nu * r_g1);
    *g_1pnu = 1.0 / (r_g2 - nu * r_g1);
}

// Helper: Scaled Bessel K using Temme's method
__device__ void besselk_scaled_temme(const double nu, const double x, 
                                     double* K_nu, double* K_nup1, double* Kp_nu) {
    const int max_iter = 15000;
    const double half_x = 0.5 * x;
    const double ln_half_x = log(half_x);
    const double half_x_nu = exp(nu * ln_half_x);
    const double pi_nu = M_PI * nu;
    const double sigma = -nu * ln_half_x;
    const double sinrat = (fabs(pi_nu) < DBL_EPSILON ? 1.0 : pi_nu/sin(pi_nu));
    const double sinhrat = (fabs(sigma) < DBL_EPSILON ? 1.0 : sinh(sigma)/sigma);
    const double ex = exp(x);
    
    double sum0, sum1;
    double fk, pk, qk, hk, ck;
    int k = 0;
    double g_1pnu, g_1mnu, g1, g2;
    
    temme_gamma(nu, &g_1pnu, &g_1mnu, &g1, &g2);
    
    fk = sinrat * (cosh(sigma) * g1 - sinhrat * ln_half_x * g2);
    pk = 0.5 / half_x_nu * g_1pnu;
    qk = 0.5 * half_x_nu * g_1mnu;
    hk = pk;
    ck = 1.0;
    sum0 = fk;
    sum1 = hk;
    
    while(k < max_iter) {
        double del0, del1;
        k++;
        fk = (k * fk + pk + qk) / (k * k - nu * nu);
        ck *= half_x * half_x / k;
        pk /= (k - nu);
        qk /= (k + nu);
        hk = -k * fk + pk;
        del0 = ck * fk;
        del1 = ck * hk;
        sum0 += del0;
        sum1 += del1;
        if(fabs(del0) < 0.5 * fabs(sum0) * DBL_EPSILON) break;
    }
    
    *K_nu = sum0 * ex;
    *K_nup1 = sum1 * 2.0 / x * ex;
    *Kp_nu = -(*K_nup1) + nu / x * (*K_nu);
}

// Helper: Integral function for large x approximation
__device__ double f(double t, double v, double x) {
    double ct = cosh(t);
    double cvt = cosh(v * t);
    return log(cvt) - x * ct;
}

// Main device function to switch between algorithms based on x
__device__ double modified_besselk(double nu, double x) {
    if (x <= 0) return INFINITY;
    if (nu < 0) nu = -nu;
    
    // For small x, use Temme's method
    if (x <= 0.1) {
        int N = (int)(nu + 0.5);
        double mu = nu - N;
        double K_mu, K_mup1, Kp_mu;
        double K_nu, K_nup1, K_num1;
        int n, e10 = 0;
        
        besselk_scaled_temme(mu, x, &K_mu, &K_mup1, &Kp_mu);
        K_nu = K_mu;
        K_nup1 = K_mup1;
        
        for(n = 0; n < N; n++) {
            K_num1 = K_nu;
            K_nu = K_nup1;
            if (fabs(K_nu) > SQRT_DBL_MAX) {
                double p = floor(log(fabs(K_nu)) / M_LN10);
                double factor = pow(10.0, p);
                K_num1 /= factor;
                K_nu /= factor;
                e10 += p;
            }
            K_nup1 = 2.0 * (mu + n + 1) / x * K_nu + K_num1;
        }
        return K_nu * exp(-x);
    } 
    else {
        // For larger x, use numerical integration (steepest descent/trapezoidal)
        const double h = 9.0 / d_intervals;
        double max_term = -INFINITY;
        double sum = 0.0;
        
        // First pass: find max term for log-sum-exp
        for (int m = 0; m <= d_intervals; m++) {
            double t_m = m * h;
            double c_m = (m == 0 || m == d_intervals) ? 0.5 : 1.0;
            double g_m = f(t_m, nu, x);
            double term = log(c_m) + g_m;
            max_term = fmax(max_term, term);
        }
        
        // Second pass: compute sum
        for (int m = 0; m <= d_intervals; m++) {
            double t_m = m * h;
            double c_m = (m == 0 || m == d_intervals) ? 0.5 : 1.0;
            double g_m = f(t_m, nu, x);
            double term = log(c_m) + g_m;
            sum += exp(term - max_term);
        }
        
        double res = log(h) + max_term + log(sum);
        return exp(res);
    }
}

extern "C" {
    __global__ void besselk_kernel(const double* __restrict__ nu, 
                                   const double* __restrict__ x,
                                   double* __restrict__ out,
                                   const int n,
                                   const int scaled) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        
        double nu_val = nu[idx];
        double x_val = x[idx];
        
        // Handle invalid inputs
        if (!(nu_val > 0.0) || !(x_val > 0.0) || isnan(nu_val) || isnan(x_val)) {
            out[idx] = NAN;
            return;
        }
        
        double result = modified_besselk(nu_val, x_val);
        
        // Apply scaling if requested
        if (scaled && isfinite(result)) {
            result *= exp(x_val);
        }
        
        out[idx] = result;
    }
}
'''

# -------------------------------------------------------------------------
# Kernel Compilation and Interface
# -------------------------------------------------------------------------
# Global variables for lazy compilation (compile only when first used)
_besselk_module = None
_besselk_kernel = None

def _get_kernel():
    """Lazy compilation of the CUDA kernel."""
    global _besselk_module, _besselk_kernel
    if _besselk_module is None:
        try:
            _besselk_module = cp.RawModule(
                code=_besselk_cuda_advanced,
                options=('-std=c++14', '--use_fast_math'),
                name_expressions=('besselk_kernel',)
            )
            _besselk_kernel = _besselk_module.get_function('besselk_kernel')
        except Exception as e:
            raise RuntimeError(f"Failed to compile CUDA kernel: {e}")
    return _besselk_kernel

def kv_gpu(nu, x, scaled=False, dtype=cp.float64):
    """
    Compute modified Bessel function of the second kind K_nu(x) on GPU.

    Parameters:
    -----------
    nu : array_like or float
        Order of the Bessel function (nu > 0).
    x : array_like or float
        Argument of the Bessel function (x > 0).
    scaled : bool, optional
        If True, return K_nu(x) * exp(x). Default is False.
    dtype : cp.dtype
        The requested output precision. 
        Note: Internal calculation is always float64.

    Returns:
    --------
    cupy.ndarray
        Values of K_nu(x).
    """
    # Convert inputs to CuPy arrays (FORCE FLOAT64 FOR KERNEL)
    nu_arr = cp.asarray(nu, dtype=cp.float64)
    x_arr = cp.asarray(x, dtype=cp.float64)
    
    # Handle broadcasting (e.g., if nu is scalar and x is array)
    out_shape = cp.broadcast_shapes(nu_arr.shape, x_arr.shape)
    nu_broadcast = cp.broadcast_to(nu_arr, out_shape)
    x_broadcast = cp.broadcast_to(x_arr, out_shape)
    
    # Flatten for linear kernel execution
    nu_flat = nu_broadcast.ravel()
    x_flat = x_broadcast.ravel()
    
    # Prepare output array (float64)
    out = cp.empty_like(nu_flat, dtype=cp.float64)
    n = out.size
    
    if n == 0:
        return out.reshape(out_shape).astype(dtype)
        
    # Get the compiled kernel
    kernel = _get_kernel()
    
    # Launch parameters
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (nu_flat, x_flat, out, n, int(bool(scaled)))
    )
    
    # Return as requested dtype
    return out.reshape(out_shape).astype(dtype)

# -------------------------------------------------------------------------
# Compatibility Functions (For C-style interface if needed)
# -------------------------------------------------------------------------

def setIntervals(nbins):
    """Set integration intervals (No-op, kept for API compatibility)."""
    pass

def initBesselK():
    """Initialize defaults (No-op, kept for API compatibility)."""
    pass

def BesselK_CUDA(host_x, host_v, host_result, n):
    """
    Compute Bessel K function with a C-compatible interface.

    Parameters:
    -----------
    host_x, host_v : array_like
        Inputs.
    host_result : array_like
        Output array (modified in-place).
    n : int
        Number of elements to process.
    """
    # Convert to CuPy if needed
    x_gpu = cp.asarray(host_x) if not isinstance(host_x, cp.ndarray) else host_x
    v_gpu = cp.asarray(host_v) if not isinstance(host_v, cp.ndarray) else host_v
    
    # Compute results
    results = kv_gpu(v_gpu[:n], x_gpu[:n], scaled=False)
    
    # Copy back to host_result
    if isinstance(host_result, cp.ndarray):
        host_result[:n] = results
    else:
        # Assume numpy array
        host_result[:n] = cp.asnumpy(results)
