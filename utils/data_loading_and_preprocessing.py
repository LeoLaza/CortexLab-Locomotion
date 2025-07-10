"""
Data loading and preprocessing functions.

This module handles:
- Loading ONE system data (experiments, spikes, behavior)
- DLC tracking data processing
- Rotary encoder data loading
- Time synchronization and binning
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy.io import loadmat
from pinkrigs_tools.dataset.query import load_data
from scipy.ndimage import gaussian_filter1d

def load_ONE(exp_kwargs):
    """
    Load the ONE folder of a specified session.

    Parameters:
    -----------
    exp_kwargs : dict
        Experiment parameters (subject, expDate)
        
    Returns:
    --------
    ONE : DataFrame containing ONE folder items
    """

    ONE = load_data(
    data_name_dict= 'all-default',
    **exp_kwargs
    )
        
    return ONE


def get_experiment_identifiers(ONE=None, exp_kwargs=None, dlc_frame_count=None, cam_fps=60):
    """
    Extract experiment path from ONE data.
    """

    if ONE is None:
        if exp_kwargs is None:
            raise ValueError("Either ONE or exp_kwargs must be provided")
        ONE = load_ONE

    exp_indices = ONE.index[(ONE.expDef == 'spontaneousActivity') & 
                           (ONE.rigName == 'poppy-stim')]


    if len(exp_indices) == 1:
        exp_idx = exp_indices[0]
        #exp_num = ONE.loc[idx, "expNum"]
        exp_path = ONE.loc[exp_idx, 'expFolder']


        return  exp_path, exp_idx
    
    else:
        dlc_duration = dlc_frame_count / cam_fps
        
        diffs = []

        for idx in exp_indices:
            exp_duration = float(ONE.loc[idx, 'expDuration'])
            diff = abs(exp_duration - dlc_duration)
            diffs.append(diff)
            

        exp_idx = exp_indices[np.argmin(diffs)]
        #exp_num = ONE.loc[idx, 'expNumber']
        exp_path = ONE.loc[exp_idx, 'expFolder']

        return exp_path, exp_idx


def get_cam_timestamps(exp_kwargs=None,ONE=None, exp_idx=None, dlc_frame_count=None):

    if exp_kwargs is None:
            raise ValueError("Exp_kwargs must be provided")
    
    if exp_idx is None:
        if ONE is None:
            ONE = load_ONE(exp_kwargs)
        _, exp_idx = get_experiment_identifiers(
            ONE=ONE, 
            exp_kwargs=exp_kwargs, 
            dlc_frame_count=dlc_frame_count
        )
        
    cam_dict = {'topCam': {'camera': ['times', 'ROIMotionEnergy']}}

    cam_data = load_data(data_name_dict=cam_dict,**exp_kwargs)
    poppy_cam_data = cam_data[cam_data['rigName'] == "poppy-stim"]
    
    
    timestamps = poppy_cam_data.loc[exp_idx, 'topCam']['camera'].times
    
    

    exp_onset= np.where(timestamps >=0)[0][0]
    timestamps[:exp_onset] = np.nan
    cam_timestamps = timestamps.flatten()
    
    return exp_onset, cam_timestamps

def create_time_bins(cam_timestamps, exp_onset, rotary_timestamps, target_freq=10):
   
    bin_width = 1.0 / target_freq

    end_time = cam_timestamps[-1]
    
    # If rotary data exists and ends earlier, use that instead
    if rotary_timestamps is not None and rotary_timestamps[-1] < end_time:
        end_time = rotary_timestamps[-1]
    
    bin_edges = np.arange(cam_timestamps[exp_onset], end_time, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers, bin_width

def temporally_align_variable(variable, bin_centers, timestamps):
    """Align any data stream, only interpolate within valid range"""
    
    # Find valid timestamp range  
    min_time = np.nanmin(timestamps)
    max_time = np.nanmax(timestamps) 
    
    # Only interpolate for bin_centers within timestamp range
    valid_bins = (bin_centers >= min_time) & (bin_centers <= max_time)
    
    aligned_variable = np.full_like(bin_centers, np.nan)
    
    if np.any(valid_bins):
        valid_mask = ~np.isnan(timestamps)
        aligned_variable[valid_bins] = np.interp(
            bin_centers[valid_bins], 
            timestamps[valid_mask], 
            variable[valid_mask]
        )
    
    return aligned_variable

def get_dlc_df (subject_id, date):
    """
    Load DeepLabCut tracking data.
    
    Parameters:
    -----------
    subject_id : str
    date : str
    exp_onset : int
        Index of camera frame when experiment starts
        
    Returns:
    --------
    dlc_df : DataFrame
        DLC tracking data
    scorer : Index
        DLC scorer information
    """
    dlc_df = pd.read_hdf(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}DLC_resnet50_downsampled_trialJul11shuffle1_150000_filtered.h5')
    dlc_df =dlc_df.droplevel('scorer', axis=1)

    return dlc_df

def preprocess_dlc_data(dlc_df, quality_thresh = 0.90, selected_bodyparts = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3'], max_distance=10, max_gap=30):

    bodypart_positions = dlc_df.loc[:, (selected_bodyparts, slice(None))]
    likelihood_values = bodypart_positions.xs('likelihood', level='coords', axis=1)

    # exclude values below quality threshold
    quality_filter = likelihood_values <= quality_thresh
    processed_x = bodypart_positions.xs('x', level='coords', axis=1)  
    processed_y = bodypart_positions.xs('y', level='coords', axis=1) 
    processed_x[quality_filter] = np.nan
    processed_y[quality_filter] = np.nan

    for bodypart in selected_bodyparts:
        x_diff = processed_x[bodypart].diff()  
        y_diff = processed_y[bodypart].diff()
        euclidean_dist = np.sqrt(x_diff**2 + y_diff**2)
    
        # Find positions where movement exceeds threshold
        false_positive = euclidean_dist > max_distance
    
        # Apply position filter to this bodypart
        processed_x.loc[false_positive, bodypart] = np.nan
        processed_y.loc[false_positive, bodypart] = np.nan

    for bodypart in selected_bodyparts:
        processed_x[bodypart] = processed_x[bodypart].interpolate(method='linear')
        processed_y[bodypart] = processed_y[bodypart].interpolate(method='linear')

        
    return processed_x, processed_y


def load_probes(exp_kwargs=None, ONE=None, exp_idx=None, dlc_frame_count=None):

    if exp_kwargs is None:
            raise ValueError("Exp_kwargs must be provided")
    
    if exp_idx is None:
        if ONE is None:
            ONE = load_ONE(exp_kwargs)
        _, exp_idx = get_experiment_identifiers(
            ONE=ONE, 
            exp_kwargs=exp_kwargs, 
            dlc_frame_count=dlc_frame_count
        )

    ephys_dict = {'spikes':'all','clusters':'all'}
    probe_dict = {'probe0':ephys_dict,'probe1':ephys_dict}
    spike_data = load_data(data_name_dict=probe_dict,**exp_kwargs)
    poppy_spike_data = spike_data[spike_data['rigName'] == "poppy-stim"]
    
    try:
        spikes_0 = poppy_spike_data.loc[exp_idx, 'probe0'].spikes.times  
        clusters_0 = poppy_spike_data.loc[exp_idx, 'probe0'].spikes.clusters  
        
    except (KeyError, IndexError, AttributeError):
        spikes_0 = None
        clusters_0 = None
        print('No probe0 data found')
    
    try:
        spikes_1 = poppy_spike_data.loc[exp_idx, 'probe1'].spikes.times  
        clusters_1 = poppy_spike_data.loc[exp_idx, 'probe1'].spikes.clusters 
    except (KeyError, IndexError, AttributeError):
        spikes_1 = None
        clusters_1 = None
        print('No probe1 data found')
    
    return spikes_0, clusters_0, spikes_1, clusters_1


def get_rotary_position(exp_kwargs=None, exp_folder=None, dlc_frame_count=None):

    if exp_folder is None:
        if exp_kwargs is None:
            raise ValueError("Need to provide either exp_folder or exp_kwargs")
        else:
            ONE = load_ONE(exp_kwargs)
            exp_folder, _ = get_experiment_identifiers(
                ONE=ONE, 
                exp_kwargs=exp_kwargs, 
                dlc_frame_count=dlc_frame_count
            )

    try:
        TICKS_PER_CYCLE = 1024
        rotary = np.load(os.path.join(exp_folder, 'rotaryEncoder.raw.npy'), allow_pickle=True)
        rotary = rotary.flatten()
        rotary[rotary > 2**31] = rotary[rotary > 2**31] - 2**32
        rotary_position = 360* rotary / (TICKS_PER_CYCLE*4)
        rotary_position = np.unwrap(rotary_position * np.pi/180) * 180/np.pi
            
        timeline_file = glob.glob(os.path.join(exp_folder, f'*_Timeline.mat'))[0]   
        time = loadmat(timeline_file)
        rotary_timestamps = time['Timeline']['rawDAQTimestamps'].item()[0, :]
       

        return rotary_timestamps, rotary_position
            

    except Exception as e:
        print(f"Error accessing {exp_folder}: {e}")
            
        return None, None
    
    
def get_spike_hist(spike_times, spike_clusters,time_bins):
    spike_counts, _, _ = np.histogram2d(
        spike_clusters, 
        spike_times,
        bins=[np.arange(0, len(np.unique(spike_clusters))+1), time_bins],
        )
    return spike_counts

def normalize_spike_counts(spike_counts):
    """
    Z-score normalize spike counts across time for each neuron.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Raw spike count matrix
        
    Returns:
    --------
    normalized_counts : array, shape (n_neurons, n_time_bins)
        Z-score normalized spike counts
    """
    spike_counts_norm = (spike_counts - np.mean(spike_counts, axis=1, keepdims=True)) / np.std(spike_counts, axis=1, keepdims=True)
    return spike_counts_norm

### ideally do this filtering only where it is needed
def filter_spike_counts(spike_counts, arena_mask, wheel_mask):
     """
    Remove neurons with zero variance in either behavioral context.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Spike count matrix
    arena_mask : array, shape (n_time_bins,)
        Boolean mask for arena periods
    wheel_mask : array, shape (n_time_bins,)  
        Boolean mask for wheel periods
        
    Returns:
    --------
    filtered_counts : array, shape (n_valid_neurons, n_time_bins)
        Spike counts with zero-variance neurons removed
    """
     arena_var = np.var(spike_counts[:, arena_mask], axis=1)
     wheel_var = np.var(spike_counts[:, wheel_mask], axis=1)

     zero_var = (arena_var == 0) | (wheel_var == 0)
     spike_counts = spike_counts[~zero_var, :]

     return spike_counts

