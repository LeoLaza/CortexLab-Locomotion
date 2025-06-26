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


def get_experiment_path(ONE, dlc_frame_count, cam_fps=60):
    """
    Extract experiment path from ONE data.
    """

    exp_indices = ONE.index[ONE.expDef.isin(['spontaneousActivity'])]

    print(len(exp_indices))
    print(exp_indices)
    if len(exp_indices) == 1:
        exp_path = ONE.loc[exp_indices[0], 'expFolder']
        return exp_path
    
    else:
        dlc_duration = dlc_frame_count / cam_fps
        
        diffs = []

        for idx in exp_indices:
            exp_duration = float(ONE.loc[idx, 'expDuration'])
            diff = abs(exp_duration - dlc_duration)
            diffs.append(diff)
            
        print(diffs)
        likely_index = np.argmin(diffs)
        print(likely_index)
        
        exp_path = ONE.loc[exp_indices[likely_index], 'expFolder']
        return exp_path
    


def get_cam_timestamps(exp_kwargs, rigName='poppy-stim'):
    """
    Load camera timestamps from ONE system.
    
    Parameters:
    -----------
    exp_kwargs : dict
        Experiment parameters (subject, expDate)
    rigName : str
        Rig identifier 
        
    Returns:
    --------
    exp_onset : int
        Index of camera frame when experiment starts
    cam_timestamps : array
        Timestamps of camera recording
    """
    data_name_dict = {'topCam':{'camera':['times','ROIMotionEnergy']}}
    recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
    stim_recordings = recordings[recordings['rigName'] == rigName]
    timestamps = stim_recordings['topCam'].iloc[0]['camera'].times
    exp_onset= np.where(timestamps >=0)[0][0]
    timestamps[:exp_onset] = np.nan
    cam_timestamps = timestamps.flatten()
    
    return exp_onset, cam_timestamps

def create_time_bins(cam_timestamps, exp_onset, rotary_timestamps, target_freq=10):
    """
    Create time bins for temporal alignment.
    
    Parameters:
    -----------
    cam_timestamps : array
        Index of camera frame when experiment starts
    exp_onset: int
        Index of camera frame when experiment starts
    bin_size : float
        Bin size in seconds
        
    Returns:
    --------
    bin_edges : array
    bin_centers : array
    bin_width : float
    """
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
        processed_x[bodypart] = processed_x[bodypart].interpolate(method='linear', limit=max_gap)
        processed_y[bodypart] = processed_y[bodypart].interpolate(method='linear', limit=max_gap)
        #total_nans_x = processed_x[bodypart].isna().sum()
        #total_nans_y = processed_y[bodypart].isna().sum()
        #print(f"{bodypart}: {total_nans_x} NaN frames for x ({total_nans_x/len(processed_x)*100:.1f}%)")
        #print(f"{bodypart}: {total_nans_y} NaN frames for y ({total_nans_y/len(processed_x)*100:.1f}%)")

        
    return processed_x, processed_y


def load_probes(exp_kwargs, rigName='poppy-stim'):
    """
    Load spike data from neuropixel probes.
    
    Parameters:
    -----------
    exp_kwargs : dict
        Experiment parameters
    rigName : str
        Rig identifier
        
    Returns:
    --------
    spikes_0 : array or None
        Spike times from probe 0
    clusters_0 : array or None
        Cluster IDs from probe 0
    spikes_1 : array or None
        Spike times from probe 1
    clusters_1 : array or None
        Cluster IDs from probe 1
    """

    ephys_dict = {'spikes':'all','clusters':'all'}
    
    data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict}
    
    recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
    stim_recordings = recordings[recordings['rigName'] == rigName]
    
    try:
        spikes_0 = stim_recordings['probe0'].iloc[0]['spikes']['times'] 
        clusters_0 = stim_recordings['probe0'].iloc[0]['spikes']['clusters']  
    except (KeyError, IndexError, AttributeError):
        spikes_0 = None
        clusters_0 = None
        print('No probe0 data found')
    
    try:
        spikes_1 = stim_recordings['probe1'].iloc[0]['spikes']['times']  
        clusters_1 = stim_recordings['probe1'].iloc[0]['spikes']['clusters']  
    except (KeyError, IndexError, AttributeError):
        spikes_1 = None
        clusters_1 = None
        print('No probe1 data found')
    
    return spikes_0, clusters_0, spikes_1, clusters_1


def get_rotary_position(exp_folder):
    """
    Load and process rotary encoder data.
    
    Parameters:
    -----------
    exp_folder : str
        Experiment folder path
    bin_centers : array
        Time bin centers for interpolation
        
    Returns:
    --------
    rotary_timestamps : array or None
        Rotary encoder timestamps
    rotary_position : array or None
        Processed wheel position data
    """
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