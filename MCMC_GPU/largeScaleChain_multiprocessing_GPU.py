import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gstatsMCMC import Topography
from gstatsMCMC import MCMC_gpu
import gstatsim as gs
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
from copy import deepcopy
import time
import multiprocessing as mp
from pathlib import Path
import os
import sys
import scipy as sp
import json
import psutil
import torch


def largeScaleChain_mp(n_chains,n_workers,largeScaleChain,rf,initial_beds,rng_seeds,n_iters,output_path='./Data/output'):
    '''
    function to run multiple large scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    largeScaleChain (MCMC.chain_crf): an existing large scale chain that has already been set-up
    rf (MCMC.RandField): an existing RandField instance that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    rng_seeds (list): a list of int used to initialize the random number generator of each chain
    n_iters (int): a list of number of iterations runned for each chain
    output_path (str): Path to the folder where the user wants to save results

    Returns
    -------
    result: a list of results from all the chains runned.

    '''
    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')
    
    tic = time.time()

    ctx = mp.get_context('spawn')

    params = []

    # Retrive parameters from the existing chain / RandField
    example_chain = largeScaleChain.__dict__ 
    example_RF = rf.__dict__

   # Modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):   
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]
        
        RF_param = deepcopy(example_RF)
        RF_param['rng_seed'] = rng_seeds[i]

        run_param = {} # A dictionary of parameters passed in the run() function
        run_param['n_iter'] = n_iters[i]
        run_param['only_save_last_bed']=True # Some display parameters are fixed.
        run_param['info_per_iter']=1000
        run_param['plot']=False
        run_param['progress_bar']=False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 1 # 1 extra line for header
        run_param['seed'] = rng_seeds[i]
        run_param['output_path'] = str(Path(output_path) / 'LargeScaleChain')

        params.append([chain_param,RF_param,deepcopy(run_param)])

    # Print header and reserve space for progress bars
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Running MCMC chains on {device}...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # Force output into the terminal
    
    # The multiprocessing step
    with ctx.Pool(n_workers) as pool:
        result = pool.starmap(lsc_run_wrapper, params)

    # Print completed output
    print('\n' * 3)
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
    print(f'Completed in {toc-tic:.2f} seconds')
    
    return result

def lsc_run_wrapper(param_chain, param_rf, param_run):
    '''
    A function used to initialize chain by input parameters and run the chains

    Parameters
    ----------
    param_chain (dict): Dictionary containing parameters needed to initialize chain
    param_rf (dict): Dictionary containing parameters needed to initialize random field
    param_run (dict): Dictionary containing parameters needed to run chain

    Returns
    -------
    result (tuple): A tuple containing the results of the run

    '''

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    chain = MCMC_gpu.init_lsc_chain_by_instance(param_chain)
    rf1 = MCMC_gpu.initiate_RF_by_instance(param_rf)

    # Restore stdout after initialization
    sys.stdout.close()
    sys.stdout = old_stdout

    # Setup output path
    output_path = param_run.get('output_path', './Data/LargeScaleChain')
    seed = param_run['seed']
    n_iter = param_run['n_iter']
    seed_folder = Path(output_path) / f'{str(seed)[:6]}'

    # Check for existing bed files (to resume progress)
    # existing_beds = list(seed_folder.glob('bed_*.txt'))
    exist_chain = list(seed_folder.glob('current_iter.txt'))
    cumulative_iters = 0
    previous_results = None
    files_to_delete = []

    # Prepare to merge/concatenate existing files with new results
    #if existing_beds:
    if exist_chain:
        #bed_file = existing_beds[0] # Existing bed file
        
        # Extract iteration count from filename
        #filename = bed_file.stem  # Gets 'bed_100k' from 'bed_100k.txt'
        #iter_str = filename.split('_')[1].replace('k', '')  # Gets '100' from 'bed_100k'
        #iter_count = int(iter_str)
        #cumulative_iters = iter_count * 1000  # Convert back to actual iterations
        cumulative_iters = int(np.loadtxt(exist_chain[0]))
        iter_count = int(cumulative_iters / 1000)
        bed_file = 'bed_'+str(iter_count)+'k.txt'
        
        # Load the most recent bed file
        most_recent_bed = np.load(seed_folder / f'bed_{iter_count}k.npy')
        
        # Update the chain's initial bed
        chain.initial_bed = most_recent_bed
        
        # Load all previous result files
        with np.load(seed_folder / f'results_{iter_count}k.npz') as results_data:
            previous_results = {
                'loss_mc' : results_data['loss_mc'],
                'loss_data' : results_data['loss_data'],
                'loss' : results_data['loss'],
                'steps' : results_data['steps'],
                'resampled_times' : results_data['resampled_times'],
                'blocks_used' : results_data['blocks_used']
            }
        
        # Mark files for deletion
        files_to_delete = [
            seed_folder / f'results_{iter_count}k.npz',
            seed_folder / 'current_iter.txt'
        ]
        
        with open(seed_folder / 'RNGState_RandField.txt', "r") as file: 
            rf1.rng.bit_generator.state = json.load(file)
        with open(seed_folder / 'RNGState_chain.txt', "r") as file: 
            chain.rng.bit_generator.state = json.load(file)

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    chain.seed = param_run.get('seed', 'Unknown')

    # Run the chain
    result = chain.run(
        n_iter=param_run['n_iter'], 
        RF=rf1, 
        only_save_last_bed=param_run['only_save_last_bed'], 
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )
    
    # Unpack results
    beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = result

    # Save the state of the random generator
    with open(seed_folder / 'RNGState_RandField.txt', "w") as file: 
        json.dump(rf1.rng.bit_generator.state, file)
    with open(seed_folder / 'RNGState_chain.txt', "w") as file: 
        json.dump(chain.rng.bit_generator.state,file)

    # Combine with previous results if they exist
    if previous_results is not None:
        # Append new results to previous results
        loss_mc = np.concatenate([previous_results['loss_mc'], loss_mc])
        loss_data = np.concatenate([previous_results['loss_data'], loss_data])
        loss = np.concatenate([previous_results['loss'], loss])
        steps = np.concatenate([previous_results['steps'], steps])
        resampled_times = previous_results['resampled_times'] + resampled_times
        blocks_used = np.vstack([previous_results['blocks_used'], blocks_used])
    
    # Calculate new cumulative iteration count
    cumulative_iters += n_iter
    iteration_label = f'{cumulative_iters // 1000}k'
    
    # Save all outputs with updated iteration label
    np.save(seed_folder / f'bed_{iteration_label}.npy', beds)

    np.savez_compressed(
        seed_folder / f'results_{iteration_label}.npz',
        loss_mc=loss_mc,
        loss_data=loss_data,
        loss=loss,
        steps=steps,
        resampled_times=resampled_times,
        blocks_used=blocks_used
    )
    
    # Delete old files after successfully saving new ones
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            
    np.savetxt(seed_folder / 'current_iter.txt', [cumulative_iters], fmt='%d')

    return result

    
def smallScaleChain_mp(n_chains, n_workers, smallScaleChain, initial_beds, ssc_rng_seeds, lsc_rng_seed, n_iters, output_path='./Data/output'):
    '''
    function to run multiple small scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    smallScaleChain (MCMC.chain_sgs): an existing small scale chain that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    ssc_rng_seeds (list): a list of int used to initialize the random number generator of each chain
    lsc_rng_seed (int): rng seed for the parent lsc that will be used to find where to save results
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''
    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')

    tic = time.time()

    params = []
    # retrive parameters from the existing chain
    example_chain = smallScaleChain.__dict__

    # modify some of the parameters based on the input ssc_rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = ssc_rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]

        run_param = {}
        run_param['n_iter'] = n_iters[i]
        # some display parameters are fixed.
        run_param['only_save_last_bed'] = True
        run_param['info_per_iter'] = 10
        run_param['plot'] = False
        run_param['progress_bar'] = False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 2 # 2 lines for header
        run_param['ssc_seed'] = ssc_rng_seeds[i]
        run_param['lsc_seed'] = lsc_rng_seed
        run_param['output_path'] = str(Path(output_path) / 'LargeScaleChain' / str(lsc_rng_seed)[:6] / 'SmallScaleChain')
        params.append([deepcopy(chain_param), deepcopy(run_param)])

    # Print header and reserve space for progress bars
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Running MCMC chains on {device}...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # force output into the terminal

    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(msc_run_wrapper, params)

    # Print final output
    print('\n' * 3)
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


def msc_run_wrapper(param_chain, param_run):
    '''
    A function used to initialize chain by input parameters and run the chains

    Parameters
    ----------
    param_chain (dict): Dictionary containing parameters needed to initialize chain
    param_rf (dict): Dictionary containing parameters needed to initialize random field
    param_run (dict): Dictionary containing parameters needed to run chain

    Returns
    -------
    result (tuple): A tuple containing the results of the run

    '''

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    chain = MCMC.init_msc_chain_by_instance(param_chain)

    # Restore stdout after initialization
    sys.stdout.close()
    sys.stdout = old_stdout

    # Setup output path
    output_path = param_run.get(
        'output_path', 
        './Data/LargeScaleChain/'+str(param_run['lsc_seed'])[:6]+'/SmallScaleChain'
        )
    seed = param_run['ssc_seed']
    n_iter = param_run['n_iter']
    seed_folder = Path(output_path) / f'{str(seed)[:6]}'

    # Check for existing bed files (to resume progress)
    existing_beds = list(seed_folder.glob('bed_*.txt'))
    cumulative_iters = 0
    previous_results = None
    files_to_delete = []

    # Prepare to merge/concatenate existing files with new reults
    if existing_beds:
        bed_file = existing_beds[0] # Existing bed file

        # Extract iteration count from filename
        filename = bed_file.stem # Gets 'bed_100k' from 'bed_100k.txt'
        iter_str = filename.split('_')[1].replace('k', '')
        iter_count = int(iter_str)
        cumulative_iters = iter_count * 1000 # Convert back to actual iterations

        # Load the most recent bed file
        most_recent_bed = np.load(seed_folder / f'bed_{iter_count}k.npy')

        # Update the chain's intiial bed
        chain.initial_bed = most_recent_bed

        # Load all previous result files
        with np.load(seed_folder / f'results_{iter_count}k.npz') as results_data:
            previous_results = {
                'loss_mc' : results_data['loss_mc'],
                'loss_data' : results_data['loss_data'],
                'loss' : results_data['loss'],
                'steps' : results_data['steps'],
                'resampled_times' : results_data['resampled_times'],
                'blocks_used' : results_data['blocks_used']
            }

        # Mark files for deletion
        files_to_delete = [
            seed_folder / f'results_{iter_count}k.npz',
            seed_folder / 'current_iter.txt'
        ]
        
        with open(seed_folder / 'RNGState_chain.txt', "r") as file: 
            chain.rng.bit_generator.state = json.load(file)

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    #chain.seed = param_run.get('ssc_seed', 'Unkown')

    # Run the chain
    result = chain.run(
        n_iter=param_run['n_iter'], 
        only_save_last_bed=param_run['only_save_last_bed'],
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )
    
    # Unpack results
    beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = result

    with open(seed_folder / 'RNGState_chain.txt', "w") as file: 
        json.dump(chain.rng.bit_generator.state,file)

    # Combine with previous results if they exist
    if previous_results is not None:
        # Append new results to previous results
        loss_mc = np.concatenate([previous_results['loss_mc'], loss_mc])
        loss_data = np.concatenate([previous_results['loss_data'], loss_data])
        loss = np.concatenate([previous_results['loss'], loss])
        steps = np.concatenate([previous_results['steps'], steps])
        resampled_times = previous_results['resampled_times'] + resampled_times
        blocks_used = np.vstack([previous_results['blocks_used'], blocks_used])

    # Calculate new cumulative iteration count
    cumulative_iters += n_iter
    iteration_label = f'{cumulative_iters // 1000}k'

    # Save all outputs with updated iteration label
    np.save(seed_folder / f'bed_{iteration_label}.npy', beds)

    np.savez_compressed(
        seed_folder / f'results_{iteration_label}.npz',
        loss_mc=loss_mc,
        loss_data=loss_data,
        loss=loss,
        steps=steps,
        resampled_times=resampled_times,
        blocks_used=blocks_used
    )

    # Delete old files after successfully saving new ones
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()

    return result

if __name__=='__main__':
    # Set file paths here
    #NOTE use r string literals in case backslashes are used
    glacier_data_path = Path(r'./data/BindSchadler_Macayeal_IceStreams.csv') 
    sgs_bed_path = Path(r'./sgs_beds/sgs_0_bindshadler_macayeal.txt')
    data_weight_path = Path(r'./data/data_weight.txt')
    seed_file_path = Path(r'./data/200_seeds.txt')
    output_path = Path(r'./data/bindshadler_macayeal/')

    config = {
        'resolution':500,
        'T3_xmin_block':50,
        'T3_xmax_block':80,
        'T3_ymin_block':50,
        'T3_ymax_block':80,
        'range_max_x':50e3,
        'range_max_y':50e3,
        'scale_min':50,
        'scale_max':150,
        'sigma':5
    }

    # Multiprocessing params
    n_iter = 5000
    offset_idx = 0 # Which seed to start from (0-9)
    n_chains = 10
    n_workers = psutil.cpu_count(logical=False)-1

    # load compiled bed elevation measurements
    df = pd.read_csv(glacier_data_path)
    
    rng_seed = 0
    
    # create a grid of x and y coordinates
    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)
    
    xmin = np.min(x_uniq)
    xmax = np.max(x_uniq)
    ymin = np.min(y_uniq)
    ymax = np.max(y_uniq)
    
    cols = len(x_uniq)
    rows = len(y_uniq)
    
    resolution = config['resolution']

    
    xx, yy = np.meshgrid(x_uniq, y_uniq)
    
    # load other data
    dhdt = df['dhdt'].values.reshape(xx.shape)
    smb = df['smb'].values.reshape(xx.shape)
    velx = df['velx'].values.reshape(xx.shape)
    vely = df['vely'].values.reshape(xx.shape)
    bedmap_mask = df['bedmap_mask'].values.reshape(xx.shape)
    bedmachine_thickness = df['bedmachine_thickness'].values.reshape(xx.shape)
    bedmap_surf = df['bedmap_surf'].values.reshape(xx.shape)
    highvel_mask = df['highvel_mask'].values.reshape(xx.shape)
    bedmap_bed = df['bedmap_bed'].values.reshape(xx.shape)
    
    bedmachine_bed = bedmap_surf - bedmachine_thickness
    
    # create conditioning data
    # bed elevation measurement in grounded ice region, and bedmachine bed topography elsewhere
    cond_bed = np.where(bedmap_mask == 1, df['bed'].values.reshape(xx.shape), bedmap_bed)
    df['cond_bed'] = cond_bed.flatten()
    
    # create a mask of conditioning data
    data_mask = ~np.isnan(cond_bed)
    
    # normalize the conditioning bed data, saved to df['Nbed']
    data = df['cond_bed'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=1000, output_distribution="normal",random_state=rng_seed,subsample=None).fit(data)
    transformed_data = nst_trans.transform(data)
    df['Nbed'] = transformed_data
    
    # randomly drop out 50% of coordinates. Decrease this value if you have a lot of data and it takes a long time to run
    df_sampled = df.sample(frac=0.15, random_state=rng_seed)
    df_sampled = df_sampled[df_sampled["cond_bed"].isnull() == False]
    df_sampled = df_sampled[df_sampled["bedmap_mask"]==1]
    
    # compute experimental (isotropic) variogram
    coords = df_sampled[['x','y']].values
    values = df_sampled['Nbed']

    maxlag = 80000      # maximum range distance
    n_lags = 60         # num of bins (try decreasing if this is taking too long)

    # compute variogram
    V1 = skg.Variogram(coords, values, bin_func='even', 
                       n_lags=n_lags, maxlag=maxlag, normalize=False, 
                       model='matern')
    
    # extract variogram values
    xdata = V1.bins
    ydata = V1.experimental
    
    # Notice: because we randomly drop out some data, the calculation of V1_p won't always obtain the same result
    # To ensure reproducibility of your work, please use a consistent set of V1_p throughout different chains
    V1_p = V1.parameters
    
    # load bed generated by Sequential Gaussian Simulation
    sgs_bed = np.loadtxt(sgs_bed_path)
    thickness = bedmap_surf - sgs_bed
    sgs_bed = np.where((thickness<=0)&(bedmap_mask==1), bedmap_surf-1, sgs_bed)
    
    grounded_ice_mask = (bedmap_mask == 1)
    
    # initialize a large scale chain to be used as an example to initialize other large scale chain
    largeScaleChain = MCMC_gpu.chain_crf_gpu(xx, yy, sgs_bed, bedmap_surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
    
    largeScaleChain.set_update_region(True,highvel_mask)
    
    mc_res_bm = Topography.get_mass_conservation_residual(bedmachine_bed,bedmap_surf,velx,vely,dhdt,smb,resolution)
    
    # in multiprocessing, we choose to only use mass flux residual loss in squared sum (Gaussian distribution)
    largeScaleChain.set_loss_type(sigma_mc=config['sigma'], massConvInRegion=True)
    
    #range_max and range_min changes topographies features' lateral scale
    #by default, I set range_max to variogram range
    range_max_x = config['range_max_x'] #in terms of meters in lateral dimension, regardless of resolution of the map
    range_max_y = config['range_max_y']
    range_min_x = 10e3
    range_min_y = 10e3
    scale_min = config['scale_min'] #in terms of meters in vertical dimension, how much you want to multiply the perturbation by
    scale_max = config['scale_max']
    nugget_max = 0
    random_field_model = 'Matern' # currently only supporting 'Gaussian' or 'Exponential'
    isotropic = True
    smoothness = V1_p[2]
    
    # initialize a RandField instance to be used for all large scale chains
    rf1 = MCMC_gpu.RandField(range_min_x, range_max_x, range_min_y, range_max_y, scale_min, scale_max, nugget_max, random_field_model, isotropic, smoothness = smoothness)
    
    min_block_x = config['T3_xmin_block']
    max_block_x = config['T3_xmax_block']
    min_block_y = config['T3_ymin_block']
    max_block_y = config['T3_ymax_block']

    rf1.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)
    
    logis_func_L = 2
    logis_func_x0 = 0
    logis_func_k = 6
    logis_func_offset = 1
    max_dist = V1_p[0] # set to the distance between two points on the map where the correlation vanish / is minimal
    # this controls how fast the perturbation magnitude decay to zero when close to a radar data
    # a large max_dist give slower decay
    rf1.set_weight_param(logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution)
    
    # spectral synthesis can generate the random field significantly faster
    rf1.set_generation_method(spectral=True)
    
    ## The function set_crf_data_weight calculate the weight for conditioning to the radar measurements
    ## Calculate this and re-save data weight again if you change weight parameters in 'set_weight_param()'
    # largeScaleChain.set_crf_data_weight(rf1)
    # largeScaleChain.set_update_type('CRF_weight')
    # np.savetxt('data_weight_denman.txt', largeScaleChain.crf_data_weight)
    
    # load and set the data weight for the chain
    largeScaleChain.crf_data_weight = np.loadtxt(data_weight_path)
    largeScaleChain.set_update_type('CRF_weight')
    
    largeScaleChain.set_random_generator(rng_seed = rng_seed)
    
    initial_beds = []
    for i in range(n_chains):
        #sgs_bed = np.loadtxt(r'sgs_beds/sgs_'+str(i)+'_bindshadler_macayeal.txt')
        sgs_bed = np.loadtxt(r'sgs_bed_denman.txt')
        initial_beds.append(sgs_bed)
    #initial_beds = np.array([sgs_bed] * n_chains) # np.repeat(sgs_bed, n_chains)
    
    with open(Path(seed_file_path), 'r') as f:
        lines = f.readlines()
    
    rng_seeds = []
    for line in lines:
        rng_seeds.append(int(line.strip()))
        
    # Create output directory for all results
    for i in range(0, n_chains):
        #print(i, rng_seeds[i])
        ls_seed = rng_seeds[i]
        ls_seed_folder = output_path / 'LargeScaleChain' / f'{str(ls_seed)[:6]}'
        ls_seed_folder.mkdir(parents=True, exist_ok=True)

# =============================================================================
#         ss_chain_folder = ls_seed_folder / 'SmallScaleChain'
#         ss_chain_folder.mkdir(parents=True, exist_ok=True)
# 
#         # For each large scale chain, create 20 small scale chain folders
#         for j in range(i*20, i*20 + 20):
#             #print('\t', j,  rng_seeds[j])
#             ss_seed = rng_seeds[j]
#             ss_seed_folder = ss_chain_folder / f'{str(ss_seed)[:6]}'
#             ss_seed_folder.mkdir(parents=True, exist_ok=True)
# =============================================================================
    
    # Use the offset to select the appropriate seeds for the large scale chain
    selected_rng_seeds = rng_seeds[offset_idx:min(offset_idx + n_chains, len(rng_seeds[:-n_chains]))]

    iteration_batches = []  
    if n_iter < 100000:
        iteration_batches = [n_iter]
    else:
        iteration_batches = [n_iter - 10000*9] + [10000]*9
    
    for iter_per_batch in iteration_batches:
        n_iters = [iter_per_batch]*n_chains

        result = largeScaleChain_mp(n_chains, n_workers, largeScaleChain, rf1, initial_beds, selected_rng_seeds, n_iters, output_path)