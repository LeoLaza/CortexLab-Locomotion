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

def load_ONE(subject_id, date):
    """
    Load specified components from ONE system.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    date : str
        Session date
        
    Returns:
    --------
    data : DataFrame
        ONE system data
    """

    data = load_data(
    subject=subject_id,
    expDate=date,
    data_name_dict= 'all-default',
    )
        
    return data


def get_experiment_path(data):
    """
    Extract experiment folder path from ONE data.
    
    Parameters:
    -----------
    data : DataFrame
        ONE system data
        
    Returns:
    --------
    exp_folder : str
        Experiment folder path
    exp_num : int
        Experiment number
    """

    exp_idx = data.index[data.expDef.isin(['spontaneousActivity'])][0]
    exp_folder = data.loc[exp_idx, 'expFolder']
    exp_num = data.loc[exp_idx, 'expNum']
    return exp_folder, exp_num

def get_timestamps(exp_kwargs, rigName='poppy-stim'):
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
    start_time : int
        Index of first valid timestamp
    timestamps : array
        Camera timestamps
    """
    data_name_dict = {'topCam':{'camera':['times','ROIMotionEnergy']}}
    recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
    stim_recordings = recordings[recordings['rigName'] == rigName]
    timestamps = stim_recordings['topCam'].iloc[0]['camera'].times
    start_time = np.where(timestamps >=0)[0][0]
    timestamps[:start_time] = np.nan
    timestamps = timestamps.flatten()
    
    return start_time, timestamps

def create_time_bins(timestamps, start_time, bin_size=0.1):
    """
    Create time bins for neural data analysis.
    
    Parameters:
    -----------
    timestamps : array
        Camera timestamps
    start_time : int
        Starting index
    bin_size : float
        Bin size in seconds
        
    Returns:
    --------
    time_bins : array
        Time bin edges
    bin_centers : array
        Time bin centers
    bin_width : float
        Time bin width
    """
   
    time_bins = np.arange(timestamps[start_time], timestamps[-1], bin_size)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    return time_bins, bin_centers, bin_width


def get_DLC_data (subject_id, date, start_time):
    """
    Load DeepLabCut tracking data.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    date : str
        Session date
    start_time : int
        Starting frame index
        
    Returns:
    --------
    dlc_df : DataFrame
        DLC tracking data
    scorer : Index
        DLC scorer information
    """
    dlc_df = pd.read_hdf(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}DLC_resnet50_downsampled_trialJul11shuffle1_150000_filtered.h5')
    dlc_df = dlc_df.iloc[start_time:]
    scorer = dlc_df.columns.get_level_values('scorer')

    return dlc_df, scorer


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


def get_rotary_metadata(exp_folder, bin_centers):
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
            
        timeline_file = glob.glob(os.path.join(exp_folder, f'*_Timeline.mat'))[0]   
        time = loadmat(timeline_file)
        rotary_timestamps = time['Timeline']['rawDAQTimestamps'].item()[0, :]
        rotary_position = 360* rotary / (TICKS_PER_CYCLE*4)
        unwrapped_rotary_position = np.unwrap(rotary_position * np.pi/180) * 180/np.pi 
        rotary_position = np.interp(bin_centers,rotary_timestamps, unwrapped_rotary_position)

        return rotary_timestamps, rotary_position
            

    except Exception as e:
        print(f"Error accessing {exp_folder}: {e}")
            
        return None, None