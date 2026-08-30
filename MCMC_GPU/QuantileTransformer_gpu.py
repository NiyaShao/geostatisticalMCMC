"""
Created on Mond Mar 4

@author: tylerrleee
"""

"""
GPU-accelerated Normal Score Transformation using CuPy

Assumes:
- Output_distribution = 'normal'
- Copies a fitted sklearn-QT quantiles_ and references_
- Only run forward / inverse mapping on GPU

Reference:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py#L2800


"""
import numpy as np
import cupy as cp
from cupyx.scipy.special import ndtr, ndtri  # standard normal CDF / PPF on GPU
# ndtri: Inverse of the cumulative distribution function of the standard (PPF)
# ndtr: Cumulative distribution function of normal distribution (CDF)

# Same threshold sklearn uses to decide when a value is "at" a bound
BOUNDS_THRESHOLD = 1e-7

class NormalScoreTransformation:
    """
    CuPy implementation of a normal-score (Gaussian) quantile transform.

    Parameters
    ----------
    quantiles_ : array-like of shape (n_quantiles, n_features)
        Empirical quantiles of the training data (one column per feature).
        Typically taken from a fitted ``sklearn.preprocessing.QuantileTransformer``.
    references_ : array-like of shape (n_quantiles,)
        The reference quantile probabilities in [0, 1], shared across features.
        Typically ``np.linspace(0, 1, n_quantiles)`` from sklearn.

    Notes
    -----
    Mirrors sklearn's ``QuantileTransformer(output_distribution='normal')``
    column-wise mapping:

        forward:  x -> CDF_emp(x) -> Phi^{-1}(.)        (normal scores)
        inverse:  z -> Phi(z)     -> quantile_emp(.)    (back to data space)

    The forward pass uses the symmetric average of ascending and descending
    interpolation (as sklearn does) to handle ties in repeated quantiles.
    """

    def __init__(self, quantiles_, references_):
        
        # Move to GPU 
        self.quantiles_ = cp.asarray(quantiles_, dtype=cp.float64)
        self.references_ = cp.asarray(references_, dtype=cp.float64).ravel()
        
        # Dim check
        if self.quantiles_.ndim != 2:
            raise ValueError(
                f"quantiles_ must be 2D (n_quantiles, n_features); "
                f"got shape {self.quantiles_.shape}"
            )
        # Shape check
        if self.quantiles_.shape[0] != self.references_.shape[0]:
            raise ValueError(
                f"quantiles_ has {self.quantiles_.shape[0]} rows but "
                f"references_ has {self.references_.shape[0]} entries; "
                f"they must match."
            )
        
        self.n_quantiles_, self.n_features_ = self.quantiles_.shape

        # Clip values matching sklearn so the inverse is consistent
        # ndtri is the inverse standard-normal CDF.
        
        eps = float(np.spacing(np.float64(1.0)))
        self._clip_min = float(ndtri(BOUNDS_THRESHOLD - eps))
        self._clip_max = float(ndtri(1.0 - (BOUNDS_THRESHOLD - eps)))

    # Vectorized per-column linear interpolation
    # replaces np.interp 
    # Source: https://github.com/numpy/numpy/blob/v2.1.0/numpy/lib/_function_base_impl.py#L1524-L1663
    
    @staticmethod
    def _interp_columns(X, xp_2d, fp_1d):
        """(
        Per-column linear interpolation, vectorized across all features.


    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

        """
        n_q, n_features = xp_2d.shape # (n_samples, n_featureS)
        
        # Find bracket index for every point
        # Run searchsorted per column | whats the leftmost index to keep xp sorted
        idx = cp.empty(X.shape, dtype=cp.int64)
        for j in range(n_features):
            idx[:, j] = cp.searchsorted(xp_2d[:, j], X[:, j], side="right")
        
        # get bracket endpoints
        lo = cp.clip(idx - 1, 0, n_q - 1)
        hi = cp.clip(idx,     0, n_q - 1)
        
        # Get corner's values
        col_idx = cp.broadcast_to(cp.arange(n_features), X.shape)
        x_lo = xp_2d[lo, col_idx]
        x_hi = xp_2d[hi, col_idx]
        y_lo = fp_1d[lo]
        y_hi = fp_1d[hi]
        
        # Compute interpolation weight
        denom = x_hi - x_lo # width
        safe = denom != 0   # avoid zero division
        t = cp.where(safe, (X - x_lo) / cp.where(safe, denom, 1.0), 0.0)

        # Returning the final interpolated values
        return y_lo + t * (y_hi - y_lo)
 


    # Forward transform: data -> normal scores     
    def transform(self, X):
        """
        Map data to the standard normal distribution feature-wise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data; will be moved to GPU if it isn't already.

        Returns
        -------
        cupy.ndarray of shape (n_samples, n_features)
            Normal scores
        """
        X_in = cp.asarray(X)
        out_dtype = X_in.dtype if X_in.dtype in (cp.float32, cp.float64) else cp.float64
        X = X_in.astype(cp.float64, copy=True)
        
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features_}); "
                f"got {X.shape}"
            )

        q = self.quantiles_
        r = self.references_
        finite_mask = ~cp.isnan(X)

        # Bounds check (in data space) BEFORE interpolation
        
        lower_bound_x = q[0:1, :]   # (1, n_features)
        upper_bound_x = q[-1:, :]
        lower_bounds_idx = (X - BOUNDS_THRESHOLD) < lower_bound_x
        upper_bounds_idx = (X + BOUNDS_THRESHOLD) > upper_bound_x

        # Symmetric average of ascending and descending interpolation
        # (handles repeated quantile values; same trick sklearn uses).
        forward  = self._interp_columns( X, q, r)
        backward = self._interp_columns(-X, -q[::-1, :], -r[::-1])
        U = 0.5 * (forward - backward)  # uniform-distributed in [0, 1]

        # Clamp boundary cases to exact 0 / 1 
        U = cp.where(upper_bounds_idx, 1.0, U)
        U = cp.where(lower_bounds_idx, 0.0, U)

        # Map uniform -> standard normal via the probit, then clip the tails
        # so the inverse is well-defined.
        Z = ndtri(U)
        Z = cp.clip(Z, self._clip_min, self._clip_max)

        # Preserve NaNs exactly where the input had them.
        Z = cp.where(finite_mask, Z, cp.nan)
        return Z.astype(out_dtype, copy=False)

    # Inverse transform: normal scores -> data                         
    def inverse_transform(self, X):
        """
        Map normal scores back to the original data distribution feature-wise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Normal-scored data.

        Returns
        -------
        cupy.ndarray of shape (n_samples, n_features)
            Data in the original space.
        """
        X_in = cp.asarray(X)
        out_dtype = X_in.dtype if X_in.dtype in (cp.float32, cp.float64) else cp.float64
        X = X_in.astype(cp.float64, copy=True)
        
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features_}); "
                f"got {X.shape}"
            )

        q = self.quantiles_
        r = self.references_
        finite_mask = ~cp.isnan(X)

        # First convert normal scores back to uniform via the standard-normal CDF
        U = ndtr(X)

        # Boundary detection happens on U vs. [0, 1] using BOUNDS_THRESHOLD,
        lower_bounds_idx = (U - BOUNDS_THRESHOLD) < 0.0
        upper_bounds_idx = (U + BOUNDS_THRESHOLD) > 1.0

        # Inverse interpolation: x_p (the references, shared) -> f_p (per-column quantiles).
        r_2d = cp.broadcast_to(r[:, None], q.shape)
        # We need np.interp(U[:, j], references_, quantiles_[:, j]) per column.
        out = cp.empty_like(U)
        for j in range(self.n_features_):
            out[:, j] = cp.interp(U[:, j], r, q[:, j])

        out = cp.where(upper_bounds_idx, q[-1:, :].ravel(), out)
        out = cp.where(lower_bounds_idx, q[0:1, :].ravel(), out)

        out = cp.where(finite_mask, out, cp.nan)
        return out.astype(out_dtype, copy=False)