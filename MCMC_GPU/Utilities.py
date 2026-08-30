import cupy as cp
from cupyx.scipy.spatial import KDTree
import verde as vd


def _interpolate(interp_method, fromx, fromy, data, tox, toy, k):
    # interpolate
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        raise ValueError('the interp_method is not correctly defined, exit the function')
    
    interp.fit((fromx, fromy), data)
    result = interp.predict((tox, toy))
    
    return result

def min_dist_from_mask_cp(xx: cp.ndarray, yy: cp.ndarray, mask: cp.ndarray):

    assert isinstance(xx, cp.ndarray), 'Error: xx is not a Cupy Array'
    assert isinstance(yy, cp.ndarray), 'Error: yy is not a Cupy Array'
    assert isinstance(mask, cp.ndarray), 'Error: mask is not a Cupy Array'

    tree = KDTree(cp.array([xx[mask], yy[mask]]).T)
    distance = tree.query(cp.array([xx.ravel(), yy.ravel()]).T)[0].reshape(xx.shape)
    return distance
