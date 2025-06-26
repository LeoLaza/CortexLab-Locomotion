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


    
def calculate_median_position(x,y):
    x = x.median(axis=1)
    y = y.median(axis=1)
    return  x, y


def calculate_oa_speed(x, y, bin_width):
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
    
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    max_distance = np.percentile(distances, 99) # think about a more rigorpus approach to filtering what velocities would be unrealistic
    distances[distances > max_distance] = np.nan
    valid_indices = np.where(~np.isnan(distances))[0]
    distances = np.interp(
    np.arange(len(distances)),  
    valid_indices,              
    distances[valid_indices]    
)
    oa_speed_pix = distances / bin_width
    conversion_factor = 0.07 
    oa_speed = oa_speed_pix * conversion_factor
    oa_speed = gaussian_filter1d(oa_speed, 6)
    oa_speed = np.concatenate(([oa_speed[0] if len(oa_speed) > 0 else 0], oa_speed)) 
    return oa_speed




def calculate_wh_speed(rotary_position, bin_width, wh_diameter=15):
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
    wh_circumference = np.pi * wh_diameter
    linear_distance_cm = np.diff(rotary_position) * wh_circumference / 360 
    wh_speed = np.abs(linear_distance_cm / bin_width) 
    wh_speed = gaussian_filter1d(wh_speed, 6)
    wh_speed = np.concatenate(([wh_speed[0] if len(wh_speed) > 0 else 0], wh_speed))
    return wh_speed

def get_position_masks (x, y, center_x, center_y, radius, subject_id):
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
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    wh_pos = distances <= radius
    wh_pos = temporal_buffer(wh_pos)

    
    right_corner =(
        (x > center_x ) &  
        (y > center_y)
    )

    left_corner = (
        (x < center_x ) &  
        (y < center_y)
    )   

    if not subject_id == 'GB012':
        oa_pos = ~wh_pos & (~right_corner)

    else:
        oa_pos = ~wh_pos & (~left_corner)

    return oa_pos, wh_pos


def temporal_buffer(pos_mask, buffer_size=10):
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
    transition_indices = np.where(np.diff(pos_mask.astype(int)) != 0)[0] + 1
    
    for i in range(len(transition_indices) - 1):
        if transition_indices[i+1] - transition_indices[i] < buffer_size:
            if transition_indices[i] > 0:
                pos_mask[transition_indices[i]:transition_indices[i+1]] = pos_mask[transition_indices[i]-1]
    
    return pos_mask

def get_speed_masks(oa_speed, wh_speed, oa_pos, wh_pos, oa_thresh = 2.0, wh_thresh=0.8):
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

    oa_speed = oa_speed.copy()
    wh_speed = wh_speed.copy()

    oa_speed[~oa_pos] = 0  
    
    oa_onsets, oa_offsets = detect_bouts(
        oa_speed,
        onset_threshold=oa_thresh,
        offset_threshold=0.5,
        min_bout_duration=20,
        min_stable_offset=10
    )

    # Create arena running mask from bouts
    oa_running = np.zeros_like(oa_speed, dtype=bool)
    for onset, offset in zip(oa_onsets, oa_offsets):
        oa_running[onset:offset] = True

     # Wheel: simple threshold + find transitions
    wh_speed[~wh_pos] = 0
    wh_running = (wh_speed >= wh_thresh) & wh_pos
    
    # Find wheel onsets and offsets from boolean mask
    wh_diff = np.diff(wh_running.astype(int))
    wh_onsets = np.where(wh_diff == 1)[0] + 1  # Transition from False to True
    wh_offsets = np.where(wh_diff == -1)[0] + 1  # Transition from True to False
    
    # Handle edge cases
    if len(wh_onsets) > 0 and len(wh_offsets) == 0:
        wh_offsets = np.array([len(wh_running) - 1])  # Running until end
    elif len(wh_onsets) == 0 and len(wh_offsets) > 0:
        wh_onsets = np.array([0])  # Running from start
    elif wh_running[0]:  # Starts running
        wh_onsets = np.concatenate([[0], wh_onsets])
    if wh_running[-1]:  # Ends running
        wh_offsets = np.concatenate([wh_offsets, [len(wh_running) - 1]])

    
    oa_running = oa_running & oa_pos 
    wh_running = wh_running & wh_pos
        
    
    return oa_running, wh_running, oa_onsets, oa_offsets, wh_onsets, wh_offsets




def detect_bouts(speed, 
                 onset_threshold=None,      
                 offset_threshold=None,     
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
    
    onsets = np.where((speed[:-1] < onset_threshold) & 
                      (speed[1:] >= onset_threshold))[0] + 1
    
    if len(onsets) == 0:
        return np.array([]), np.array([])
    
    offsets = []
    for onset in onsets:

        offset_found = False
        for i in range(onset + min_bout_duration, len(speed)):
            if i + min_stable_offset > len(speed):
                offsets.append(len(speed) - 1)
                offset_found = True
                break
            
            if np.all(speed[i:i+min_stable_offset] < offset_threshold):
                offsets.append(i)
                offset_found = True
                break
        
        if not offset_found:
            offsets.append(len(speed) - 1)
    
    offsets = np.array(offsets)
    
    bout_durations = offsets - onsets
    valid_bouts = bout_durations >= min_bout_duration
    
    return onsets[valid_bouts], offsets[valid_bouts]
