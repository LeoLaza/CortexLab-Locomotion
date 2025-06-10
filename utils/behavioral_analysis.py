"""
Behavioral analysis functions for locomotion and context detection.

This module handles:
- Velocity calculations from position data
- Context classification (arena vs wheel)
- Locomotion bout detection
- Temporal smoothing of behavioral states
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd

def calculate_median_position(dlc_df, scorer, BODYPARTS = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3']):
    """
    Calculate robust median position from multiple DLC bodyparts.
    
    Parameters:
    -----------
    dlc_df : DataFrame
        DLC tracking data
    scorer : Index
        DLC scorer information
    BODYPARTS : list
        List of bodypart names to use
        
    Returns:
    --------
    x : Series
        Median x position
    y : Series
        Median y position
    """
    
    bodypart_positions = dlc_df.loc[:, (scorer, BODYPARTS, slice(None))]
    

    
    likelihood_values = bodypart_positions.xs('likelihood', level='coords', axis=1)
    low_filter = likelihood_values <= 0.95
    strong_x = bodypart_positions.xs('x', level='coords', axis=1)  # Define first
    strong_y = bodypart_positions.xs('y', level='coords', axis=1) 
    strong_x[low_filter] = np.nan
    strong_y[low_filter] = np.nan
    strong_x = strong_x.interpolate(method='linear', axis=0)
    strong_y = strong_y.interpolate(method='linear', axis=0)
    x = strong_x.median(axis=1)
    y = strong_y.median(axis=1)
    
    return  x, y

def bin_median_positions(x, y,timestamps, start_time, bin_centers):
    """
    Bin position data to match neural data timing.
    
    Parameters:
    -----------
    x, y : Series
        Position coordinates
    timestamps : array
        Camera timestamps
    start_time : int
        Starting time index
    bin_centers : array
        Time bin centers
        
    Returns:
    --------
    binned_x : array
        Binned x positions
    binned_y : array
        Binned y positions
    """ 
    binned_x = np.interp(bin_centers, np.linspace(timestamps[start_time], timestamps[-1], len(x)), x)
    binned_y = np.interp(bin_centers, np.linspace(timestamps[start_time], timestamps[-1], len(y)), y)
    return binned_x, binned_y


def calculate_velocity(binned_x, binned_y, bin_width):
    """
    Calculate velocity from position data with outlier removal and smoothing.
    
    Parameters:
    -----------
    binned_x, binned_y : array
        Position coordinates over time
    bin_width : float
        Time bin width in seconds
        
    Returns:
    --------
    velocity : array
        Smoothed velocity in cm/s
    """
    
    distances = np.sqrt(np.diff(binned_x)**2 + np.diff(binned_y)**2)
    max_distance = np.percentile(distances, 99)  
    distances[distances > max_distance] = np.nan
    valid_indices = np.where(~np.isnan(distances))[0]
    distances = np.interp(
    np.arange(len(distances)),  
    valid_indices,              
    distances[valid_indices]    
)
    velocity_pix = distances / bin_width
    conversion_factor = 0.07 
    velocity = velocity_pix * conversion_factor
    velocity = gaussian_filter1d(velocity, 3)
    velocity = np.concatenate(([velocity[0] if len(velocity) > 0 else 0], velocity)) 
    return velocity




def calculate_wheel_velocity(rotary_position, bin_width, wheel_diameter=10):
    """
    Calculate wheel running velocity from rotary encoder data.
    
    Parameters:
    -----------
    rotary_position : array
        Wheel position in degrees over time
    bin_width : float
        Time bin width in seconds
    wheel_diameter : float
        Wheel diameter in cm
        
    Returns:
    --------
    wheel_velocity : array
        Wheel running velocity in cm/s
    """
    wheel_circumference = np.pi * wheel_diameter
    linear_distance_cm = np.diff(rotary_position) * wheel_circumference / 360 
    wheel_velocity = np.abs(linear_distance_cm / bin_width) 
    wheel_velocity = gaussian_filter1d(wheel_velocity, 3)
    wheel_velocity = np.concatenate(([wheel_velocity[0] if len(wheel_velocity) > 0 else 0], wheel_velocity))
    return wheel_velocity

def get_position_masks (binned_x, binned_y, center_x, center_y, radius, subject_id):
    """
    Classify position data into arena vs wheel contexts.
    
    Parameters:
    -----------
    binned_x, binned_y : array
        Position coordinates
    center_x, center_y : float
        Wheel center coordinates
    radius : float
        Wheel radius in pixels
    subject_id : str
        Subject identifier for subject-specific handling
        
    Returns:
    --------
    arena_mask : array, bool
        True for arena periods
    wheel_mask : array, bool  
        True for wheel periods
    """
    distances = np.sqrt((binned_x - center_x) ** 2 + (binned_y - center_y) ** 2)

    wheel_mask = distances <= radius
    wheel_mask = temporal_buffer(wheel_mask)

    
    right_corner =(
        (binned_x > center_x ) &  
        (binned_y > center_y)
    )

    left_corner = (
        (binned_x < center_x ) &  
        (binned_y < center_y)
    )   

    if not subject_id == 'GB012':
        arena_mask = ~wheel_mask & (~right_corner)

    else:
        arena_mask = ~wheel_mask & (~left_corner)

    return arena_mask, wheel_mask


def temporal_buffer(mask, buffer_size=30):
    """
    Smooth context transitions by buffering brief context switches.
    
    Parameters:
    -----------
    mask : array, bool
        Original context mask
    buffer_size : int
        Minimum duration (in time bins) for context switches
        
    Returns:
    --------
    smoothed_mask : array, bool
        Temporally buffered mask
    """
    transition_indices = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
    
    for i in range(len(transition_indices) - 1):
        if transition_indices[i+1] - transition_indices[i] < buffer_size:
            if transition_indices[i] > 0:
                mask[transition_indices[i]:transition_indices[i+1]] = mask[transition_indices[i]-1]
    
    return mask

def get_speed_masks(velocity, wheel_velocity, arena_mask, wheel_mask):
    """
    Detect running periods in each context using velocity thresholds and bout detection.
    
    Parameters:
    -----------
    velocity : array
        Arena velocity in cm/s
    wheel_velocity : array
        Wheel velocity in cm/s
    arena_mask : array, bool
        Arena context mask
    wheel_mask : array, bool
        Wheel context mask
        
    Returns:
    --------
    oa_running : array, bool
        True during arena running periods
    wh_running : array, bool  
        True during wheel running periods
    """

    wh_running = (wheel_velocity > 1) &  ~arena_mask
    arena_velocity = velocity.copy()
    arena_velocity[~arena_mask] = 0  
    
    arena_onsets, arena_offsets = detect_bouts(
        arena_velocity,
        onset_threshold=2.0,
        offset_threshold=0.5,
        min_bout_duration=20
    )
    
    # Create arena running mask from bouts
    oa_running = np.zeros_like(velocity, dtype=bool)
    for onset, offset in zip(arena_onsets, arena_offsets):
        oa_running[onset:offset] = True
    
    oa_running = oa_running & arena_mask  
    
    return oa_running, wh_running #arena_onsets, arena_offsets




def detect_bouts(velocity, 
                 onset_threshold=2.0,      
                 offset_threshold=0.5,     
                 min_bout_duration=20,    
                 min_stable_offset=10):

    """
    Detect locomotion bouts using velocity thresholds with stable offset detection.
    
    Parameters:
    -----------
    velocity : array
        Velocity signal in cm/s
    onset_threshold : float
        Velocity threshold for bout onset
    offset_threshold : float
        Velocity threshold for bout offset
    min_bout_duration : int
        Minimum bout duration in time bins
    min_stable_offset : int
        Required duration below offset threshold
        
    Returns:
    --------
    onsets : array
        Bout onset indices
    offsets : array
        Bout offset indices
    """    
    
    onsets = np.where((velocity[:-1] < onset_threshold) & 
                      (velocity[1:] >= onset_threshold))[0] + 1
    
    if len(onsets) == 0:
        return np.array([]), np.array([])
    
    offsets = []
    for onset in onsets:

        offset_found = False
        for i in range(onset + min_bout_duration, len(velocity)):
            if i + min_stable_offset > len(velocity):
                offsets.append(len(velocity) - 1)
                offset_found = True
                break
            
            if np.all(velocity[i:i+min_stable_offset] < offset_threshold):
                offsets.append(i)
                offset_found = True
                break
        
        if not offset_found:
            offsets.append(len(velocity) - 1)
    
    offsets = np.array(offsets)
    
    bout_durations = offsets - onsets
    valid_bouts = bout_durations >= min_bout_duration
    
    return onsets[valid_bouts], offsets[valid_bouts]
