"""
Created on Mond Mar 4

@author: tylerrleee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gstatsMCMC import Topography
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
import time
import multiprocessing as mp
from pathlib import Path
import os
import sys
import scipy as sp
import json
import psutil
import pickle
from typing import List


def _get_n_gpus() -> int:
    """Get GPU count
    Importing CuPy for a single process is really slow, to use cupy.cuda.runtime.getDeviceCount()
    so we use subprocess output
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        )
        return len(out.decode().strip().split('\n'))
    except Exception:
        return 0


    
def smallScaleChain_mp(n_chains : int , 
                       n_workers : int, 
                       shared_data : dict, 
                       per_chain_data : List[dict], 
                       ):
    
    """
    Run batch of small_scalechains using multiprocessing via spawn context
    
    Parameters
    ----------
    n_chains       : int           – chains in this batch
    n_workers      : int           – worker processes (== n_chains for 1-per-GPU)
    shared_data    : dict          – NumPy arrays + plain-Python scalars
    per_chain_data : list[dict]    – one dict per chain
 
    Returns
    -------
    list of result tuples
    """

    os.system('cls' if os.name == 'nt' else 'clear')
    
    ctx = mp.get_context('spawn')
    tic = time.time()

    params = [(shared_data, pcd) for pcd in per_chain_data]

    print(f'Running {n_chains} MCMC chains...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush()
    
    with ctx.Pool(n_workers) as pool:
        result = pool.starmap(msc_run_wrapper, params)
    
    print('\n')
    print(r'''
           _o                  _                 _o_   o   o
      o    (^)  _             (o)    >')         (^)  (^) (^)
   _ (^) ('>~ _(v)_      _   //-\\   /V\      ('> ~ __.~   ~
 ('v')~ // \\  /-\      (.)-=_\_/)   (_)>     (V)  ~  ~~ /__ /\
//-=-\\ (\_/) (\_/)      V \ _)>~    ~~      <(__\[     ](__=_')
(\_=_/)  ^ ^   ^ ^       ~  ~~                ~~~~        ~~~~~
_^^_^^   __  ..-.___..---I~~~:_  .__...--.._.;-'I~~~~-.____...;-
 |~|~~~~~| ~~|  _   |    |  _| ~~|  |  |  |  |_ |      | _ |  |
_.-~~_.-~-~._.-~~._.-~-~_.-~~_.-~~_.-~-~._.-~~._.-~-~_.-~~_.-~-~
    ''')

    toc = time.time()
    print(f'Completed in {toc-tic} seconds')

    return result


def msc_run_wrapper(shared_data, per_chain):
    '''
    Worker:
    1. Pin GPU to each chain
    2. Build chain from parameters (Numpy-based)
    3. Run chains
    4. Save chains

    '''
    # We import this in process as part of the spawn context
    import cupy as cp
    import scipy as sp
    from MCMC_GPU import MCMC_cu, QuantileTransformer_gpu

    #1. Pin GPU to each chain
    gpu_id = per_chain['gpu_id']
    cp.cuda.Device(gpu_id).use()
    
    # 2. house cleaning - Suppress init prints
    old_stout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    # 3. unpack + assign data 
    # these are all shared + consistent in all chain
    xx                = shared_data['xx']
    yy                = shared_data['yy']
    surf              = shared_data['surf']
    velx              = shared_data['velx']
    vely              = shared_data['vely']
    dhdt              = shared_data['dhdt']
    smb               = shared_data['smb']
    cond_bed          = shared_data['cond_bed']
    data_mask         = shared_data['data_mask']
    grounded_ice_mask = shared_data['grounded_ice_mask']
    highvel_mask      = shared_data['highvel_mask']
    resolution        = shared_data['resolution']
    V1_p              = shared_data['V1_p']
    sigma_mc          = shared_data['sigma_mc']
    min_block_x       = shared_data['min_block_x']
    max_block_x       = shared_data['max_block_x']
    min_block_y       = shared_data['min_block_y']
    max_block_y       = shared_data['max_block_y']
    sgs_num_neighbors = shared_data['sgs_num_neighbors']
    sgs_search_radius = shared_data['sgs_search_radius']
    rng_seed_base     = shared_data['rng_seed_base']
    
    # 4. Unpack per-chain data
    # unique to each chain
    initial_bed   = per_chain['initial_bed']
    ssc_seed      = per_chain['ssc_seed']
    lsc_seed      = per_chain['lsc_seed']
    chain_id      = per_chain['chain_id']
    tqdm_position = per_chain['tqdm_position']
    n_iter        = per_chain['n_iter']
    output_path   = per_chain['output_path']
    
    # 5. per_bed trend + normal score transformation on GPU
    trend = sp.ndimage.gaussian_filter(initial_bed, sigma = sigma_mc) # TODO adjust this 
    data_for_distribution = (initial_bed - trend).reshape((-1, 1)) # detrend
    
    sklearn_qt = QuantileTransformer(
        n_quantiles = 1000,
        output_distribution="normal",
        subsample   = None,
        random_state = rng_seed_base,
    ).fit(data_for_distribution)
    nst_trans = QuantileTransformer_gpu.NormalScoreTransformGPU(sklearn_qt)
    
    # 6. Build chain on respective GPU (id)
    chain = MCMC_cu.chain_sgs_gpu(
        xx, yy, initial_bed, surf, velx, vely,
        dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution,
    )
    chain.set_update_region(True, highvel_mask)
    chain.set_loss_type(sigma_mc = sigma_mc, massConvInRegion=True)
    chain.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)
    chain.set_normal_transformation(nst_trans, do_transform=True)
    chain.set_trend(trend=trend, detrend_map=True)
    chain.set_variogram(
        'Matern', 
        V1_p[0],
        V1_p[1], 0,
        isotropic=True, 
        vario_smoothness = V1_p[2],
    )
    chain.set_sgs_param(sgs_num_neighbors, sgs_search_radius, sgs_rand_dropout_on=False)
    chain.set_random_generator(rng_seed = ssc_seed)
    
    sys.stdout.close()
    sys.stdout = old_stout
    
    # Read 
    seed_folder = Path(output_path) / f'{str(ssc_seed)[:6]}'
    seed_folder.mkdir(parents=True, exist_ok=True)
        
    exist_chain = list(seed_folder.glob('current_iter.txt'))
    cumulative_iters = 0
    previous_results = None
    files_to_delete = []
    
    if exist_chain:
        cumulative_iters = int(np.loadtxt(exist_chain[0]))
        iter_count = int(cumulative_iters / 1000)
        most_recent_bed = np.load(seed_folder / f'bed_{iter_count}k.npy')
        chain.initial_bed = cp.asarray(most_recent_bed)
        
        # build result dict
        with np.load(seed_folder / f'results_{iter_count}k.npz') as rd:
            previous_results = {
                'loss_mc':         rd['loss_mc'].copy(),
                'loss_data':       rd['loss_data'].copy(),
                'loss':            rd['loss'].copy(),
                'steps':           rd['steps'].copy(),
                'resampled_times': rd['resampled_times'].copy(),
                'blocks_used':     rd['blocks_used'].copy(),
            }
            
        files_to_delete = [
            seed_folder / f'results_{iter_count}k.npz',
            seed_folder / 'current_iter.txt',
        ]
        
        rng_state_json = seed_folder / 'RNGState_chain.json'
        rng_state_txt  = seed_folder / 'RNGState_chain.txt'
        state_file = rng_state_json if rng_state_json.exists() else (
            rng_state_txt if rng_state_txt.exists() else None
        )
        if state_file is not None:
            chain.set_random_generator(rng_seed=ssc_seed + cumulative_iters)

    
    # Run chains
    chain.chain_id      = chain_id
    chain.tqdm_position = tqdm_position
    chain.seed          = ssc_seed
 
    result = chain.run(
        n_iter=n_iter,
        only_save_last_bed=True,
        info_per_iter = 10,
        plot=False,
        progress_bar=False,
    )
    print("RUN DONE!!!!")
    beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = result
    
    # ---- Convert CuPy arrays to NumPy for saving ----
    # chain.run() returns CuPy arrays; np.save/np.concatenate need NumPy
    import cupy as cp
    def _to_numpy(x):
        return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)
    
    beds            = _to_numpy(beds)
    loss_mc         = _to_numpy(loss_mc)
    loss_data       = _to_numpy(loss_data)
    loss            = _to_numpy(loss)
    steps           = _to_numpy(steps)
    resampled_times = _to_numpy(resampled_times)
    blocks_used     = _to_numpy(blocks_used)
    
    # Save RNG serializable - since we can't save CuPy Generator as JSON since it on VRAM 
    try:
        bg = chain.rng.bit_generator
        raw_state = bg.state() if callable(bg.state) else bg.state
 
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bytes):
                return list(obj)
            else:
                try:
                    return obj.get().tolist()
                except Exception:
                    return str(obj)
 
        saveable_state = _make_serializable(raw_state)
        with open(seed_folder / 'RNGState_chain.json', 'w') as f:
            json.dump(saveable_state, f)
            
    except Exception as e:
        print(f"[WARNING] Could not save RNG state for chain {chain_id}: {e}")
        print(f"          Chain results will still be saved, but resume won't restore exact RNG position.")
    
    # <><><><><> :D
    # Update results 
    print("UPDATE RESULTS")
    if previous_results is not None:
        loss_mc         = np.concatenate([previous_results['loss_mc'],    loss_mc])
        loss_data       = np.concatenate([previous_results['loss_data'],  loss_data])
        loss            = np.concatenate([previous_results['loss'],       loss])
        steps           = np.concatenate([previous_results['steps'],      steps])
        resampled_times = previous_results['resampled_times'] + resampled_times
        blocks_used     = np.vstack([previous_results['blocks_used'],     blocks_used])
    
    # Add to total iterations ran (lsc + ssc)
    cumulative_iters += n_iter
    iteration_label = f'{cumulative_iters // 1000}k'
 
    print('STARTING TO SAVE')
    print(f'seed_folder = {seed_folder}')
    print(f'seed_folder exists = {seed_folder.exists()}')

    try:
        np.save(seed_folder / f'bed_{iteration_label}.npy', beds)
        print(f'Saved bed_{iteration_label}.npy')
        np.savez_compressed(
            seed_folder / f'results_{iteration_label}.npz',
            loss_mc=loss_mc, loss_data=loss_data, loss=loss,
            steps=steps, resampled_times=resampled_times, blocks_used=blocks_used,
        )
        print(f'Saved results_{iteration_label}.npz')
        np.savetxt(seed_folder / 'current_iter.txt', [cumulative_iters], fmt='%d')
        print(f'Saved current_iter.txt')
    except Exception as e:
        print(f'SAVE FAILED: {e}')
        import traceback; traceback.print_exc()

    return result
    
if __name__ == '__main__':
    """
    These values are computed once, so we compute them on CPU, and then store them on each GPU

    A constaint is that a personal device might run out of memory on large sites b/c of all of the .copy()
    """
    # FILE PATHS 
    glacier_data_path = Path(r'./data/Supprt_Force.csv')
    seed_file_path    = Path(r'./data/200_seeds.txt')
    output_path       = Path(r'./data/support_force')
    
    # Number of SmallScaleChains per LargeScaleChain
    n_ssc_per_lsc = 10

    sigma_mc = 1.5

    # Total iterations per chain
    n_iter = 10000
    
    lsc_starting_idx = 0
    lsc_ending_idx   = 10
    
    rng_seed = 0
    
    # LOAD DATA
    df = pd.read_csv(glacier_data_path)
 
    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)
 
    resolution = 500
    xx, yy = np.meshgrid(x_uniq, y_uniq)

    # assign values -- use copy() to avoid in-place updates
    # this is on-device memory , not VRAM
    dhdt                 = df['dhdt'].values.reshape(xx.shape).copy()
    smb                  = df['smb'].values.reshape(xx.shape).copy()
    velx                 = df['velx'].values.reshape(xx.shape).copy()
    vely                 = df['vely'].values.reshape(xx.shape).copy()
    bedmap_mask          = df['bedmap_mask'].values.reshape(xx.shape).copy()
    bedmachine_thickness = df['bedmachine_thickness'].values.reshape(xx.shape).copy()
    bedmap_surf          = df['bedmap_surf'].values.reshape(xx.shape).copy()
    highvel_mask         = df['highvel_mask'].values.reshape(xx.shape).copy()
    bedmap_bed           = df['bedmap_bed'].values.reshape(xx.shape).copy()
    
    # BM BED
    bedmachine_bed = bedmap_surf - bedmachine_thickness
    
    # COND MASK
    cond_bed = np.where(
        bedmap_mask == 1, df['bed'].values.reshape(xx.shape), bedmap_bed,
    ).copy()
    df['cond_bed'] = cond_bed.flatten()
    
    data_mask = (~np.isnan(cond_bed)).copy()
    
    # Get seeds
    with open(seed_file_path, 'r') as f:
        rng_seeds = [int(line.strip()) for line in f.readlines()]
        
    lsc_indices = range(lsc_starting_idx, lsc_ending_idx + 1)
    first_lsc_seed = rng_seeds[lsc_indices[0]]
    first_lsc_path = output_path / 'LargeScaleChain' / str(first_lsc_seed)[:6]

    # get LSC bed paths (latest) - just one!
    first_bed_files = sorted(
        first_lsc_path.glob('bed_*.npy'),
        key=lambda f: int(f.stem.split('_')[1].replace('k', '')), # filter only the bed number
    )

    first_bed = np.load(first_bed_files[-1])
    thickness = bedmap_surf - first_bed
    first_bed = np.where(
        (thickness <= 0) & (bedmap_mask == 1), bedmap_surf - 1, first_bed,
    )
    trend_for_vario = sp.ndimage.gaussian_filter(first_bed, sigma=10) # ADJUST THIS
    df['cond_bed_residual'] = df['cond_bed'].values - trend_for_vario.flatten()
    data_for_dist_vario = (first_bed - trend_for_vario).reshape((-1, 1))
    
    sklearn_qt_vario = QuantileTransformer(
        n_quantiles = 1000,
        output_distribution = "normal",
        subsample = None,
        random_state=rng_seed,
    ).fit(data_for_dist_vario)
    
    transformed_data = sklearn_qt_vario.transform(
        df['cond_bed_residual'].values.reshape(-1, 1),
    )
    df['Nbed_residual'] = transformed_data
 
    # ADJUST SAMPLING if take too long
    df_sampled = df.sample(frac = 0.5, random_state=rng_seed)
    df_sampled = df_sampled[df_sampled['cond_bed_residual'].isnull() == False]
    df_sampled = df_sampled[df_sampled['bedmap_mask'] == 1]
    
    coords = df_sampled[['x', 'y']].values
    values = df_sampled['Nbed_residual']
     
    MAX_LAG     = 70000
    N_LAGS_BINS = 70
    V1 = skg.Variogram(
        coords, values, bin_func='even',
        n_lags=N_LAGS_BINS, maxlag=MAX_LAG, normalize=False, model='matern',
    )
    
    V1_p: list = [float(p) for p in V1.parameters]
    
    grounded_ice_mask = (bedmap_mask == 1).copy()
    
    # Cast to dictionary - all numpy based, MCMC_cu will cast to cp.asarray
    
    shared_data = {
        'xx':                np.array(xx, copy=True),
        'yy':                np.array(yy, copy=True),
        'surf':              np.array(bedmap_surf, copy=True),
        'velx':              np.array(velx, copy=True),
        'vely':              np.array(vely, copy=True),
        'dhdt':              np.array(dhdt, copy=True),
        'smb':               np.array(smb, copy=True),
        'cond_bed':          np.array(cond_bed, copy=True),
        'data_mask':         np.array(data_mask, dtype=np.bool_, copy=True),
        'grounded_ice_mask': np.array(grounded_ice_mask, dtype=np.bool_, copy=True),
        'highvel_mask':      np.array(highvel_mask, copy=True),
        'resolution':        int(resolution),
        'V1_p':              V1_p,          # list[float]
        'sigma_mc':          sigma_mc, # ADJUST THIS
        'min_block_x':       5,
        'max_block_x':       20,
        'min_block_y':       5,
        'max_block_y':       20,
        'sgs_num_neighbors': 48,
        'sgs_search_radius': 30e3,
        'rng_seed_base':     int(rng_seed),
    }
    

    # check if shared_data is pickable (unpackable)
    # This will fail if we have a non numpy array or a non-int is inserted
    try:
        pickle.dumps(shared_data)
        print("AYYYY shared_data is picklable")
    except Exception as e:
        print(f"shared_data pickle FAILED :( - {e}")
        for k, v in shared_data.items():
            try:
                pickle.dumps(v)
            except Exception as e2:
                print(f"  Key '{k}' FAILED: {e2}  (type={type(v)})")
        sys.exit(1)
    
    # PER CHAIN DATA - ADJUST THIS
    per_chain_data_all = []
    
    for lsc_idx in lsc_indices:
        lsc_seed = rng_seeds[lsc_idx]
        lsc_path = output_path / 'LargeScaleChain' / str(lsc_seed)[:6]
        
        # Get all beds
        bed_files = sorted(
            lsc_path.glob('bed_*.npy'),
            key=lambda f: int(f.stem.split('_')[1].replace('k', '')),
        )
        if not bed_files:
            print(f"WARNING: No bed files in {lsc_path}, skipping LSC {lsc_idx}")
            continue
            
        latest_bed_file = bed_files[-1]
        # e.g. 'bed_1000k.npz' will be parsed as 
        # bed_1000k -> 1000k -> 1000 * 1000 -> 1 000 000 
        lsc_iter = int(latest_bed_file.stem.split('_')[1].replace('k', '')) * 1000
        one_initial_bed = np.load(latest_bed_file).copy()
        thickness = bedmap_surf - one_initial_bed
        one_initial_bed = np.where(
            (thickness <= 0) & (bedmap_mask == 1), bedmap_surf - 1, one_initial_bed,
        ).copy()
        
        # get seeds
        ssc_seeds_for_lsc = rng_seeds[
            10 + lsc_idx * n_ssc_per_lsc :
            10 + lsc_idx * n_ssc_per_lsc + n_ssc_per_lsc
        ]
        
        for j, ssc_seed in enumerate(ssc_seeds_for_lsc):
            ssc_folder = lsc_path / 'SmallScaleChain' / str(ssc_seed)[:6]
            ssc_folder.mkdir(parents=True, exist_ok=True)
            np.savetxt(ssc_folder / 'init_iter.txt', [lsc_iter], fmt='%d')
 
            per_chain_data_all.append({
                'initial_bed':  np.array(one_initial_bed, copy=True),
                'ssc_seed':     int(ssc_seed),
                'lsc_seed':     int(lsc_seed),
                'gpu_id':       None, # this will get assigned in smallScaleChain_mp()
                'chain_id':     None,
                'tqdm_position': None,
                'n_iter':       int(n_iter),
                'output_path':  str(lsc_path / 'SmallScaleChain'),
            })
    
    n_ssc_total = len(per_chain_data_all)
    print(f"Total SSC chains to run: {n_ssc_total}")
    
    # now that chains data are loaded, lets test that they are unpackable
    if per_chain_data_all:
        try:
            pickle.dumps(per_chain_data_all[0])
            print("per_chain_data is picklable AYYYYY")
        except Exception as e:
            print(f"per_chain_data pickle FAILED :( : {e}")
            for k, v in per_chain_data_all[0].items():
                try:
                    pickle.dumps(v)
                except Exception as e2:
                    print(f"  Key '{k}' FAILED: {e2}  (type={type(v)})")
            sys.exit(1)
 

    
    # Batch runs : one chain per GPU 
    # Once a chain finishes running, we move onto the next chain
    # check $nvidia-smi
    n_gpus = _get_n_gpus()

    if n_gpus == 0:
        print("ERROR: No GPUs detected via nvidia-smi")
        sys.exit(1)
    print(f"GPUs available: {n_gpus}")
    
    all_results = []
 
    for i in range(0, n_ssc_total, n_gpus):
        batch_size = min(n_gpus, n_ssc_total - i)
        batch_pcd  = per_chain_data_all[i : i + batch_size]
 
        for j, pcd in enumerate(batch_pcd):
            pcd['gpu_id']        = j % n_gpus
            pcd['chain_id']      = i + j
            pcd['tqdm_position'] = j + 2
 
        print(f"\n--- Batch {i // n_gpus + 1}: "
              f"chains {i} to {i + batch_size - 1} ---")
 
        result = smallScaleChain_mp(
            batch_size,
            batch_size,
            shared_data,
            batch_pcd,
        )
        all_results.extend(result)
 
    print(f"\nAll {n_ssc_total} chains completed.")



    

    

