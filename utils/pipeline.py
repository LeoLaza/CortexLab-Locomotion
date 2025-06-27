"""
High-level pipeline functions for complete session analysis.

This module handles:
- Complete single session processing workflows
- Multi-session analysis coordination
- Integration of all analysis modules
"""

import numpy as np
from types import SimpleNamespace as Bunch
from .behavioral_analysis import *
from .neural_processing import *
from .correlation_analysis import *
from .statistical_testing import *
from .roi_detection import *
from .data_io import *


def load_session_data(subject_id, date, target_freq=None):

    exp_kwargs = {
        'subject': subject_id, 
        'expDate': date
    }

    # load data
    ONE = load_ONE(exp_kwargs)
    dlc_df = get_dlc_df(subject_id, date)
    exp_path, exp_idx = get_experiment_identifiers(ONE, dlc_frame_count= len(dlc_df), cam_fps=60)
    exp_onset, cam_timestamps = get_cam_timestamps(exp_kwargs, None, exp_idx)
    bodypart_x, bodypart_y = preprocess_dlc_data(dlc_df)
    x, y = calculate_median_position(bodypart_x, bodypart_y)
    rotary_timestamps, rotary_position = get_rotary_position(None, exp_path)
    
    # temporally align variables
    bin_edges, bin_centers, bin_width = create_time_bins(cam_timestamps, exp_onset, rotary_timestamps, target_freq=target_freq)
    rotary_position = temporally_align_variable(rotary_position, bin_centers, rotary_timestamps)
    x = temporally_align_variable(x, bin_centers, cam_timestamps)
    y = temporally_align_variable(y, bin_centers, cam_timestamps)


    # define context
    frame, roi_x, roi_y, radius = get_ROI(subject_id, date)
    #plot_ROI(frame, roi_x, roi_y, radius, subject_id, date)
    
    # compute speed in either context 
    oa_speed = calculate_oa_speed(x, y, bin_width)
    wh_speed = calculate_wh_speed(rotary_position, bin_width)


    # identify bouts of running in either context
    oa_pos, wh_pos = get_position_masks(x, y, roi_x, roi_y, radius, subject_id)
    oa_running, wh_running, oa_onsets, oa_offsets, wh_onsets, wh_offsets = get_speed_masks(oa_speed, wh_speed, oa_pos, wh_pos, oa_thresh = 2.0, wh_thresh=0.8)
    
        
    # generate spike histogram 
    spikes_0, clusters_0, spikes_1, clusters_1 = load_probes(exp_kwargs, None, exp_idx)
    
    try: 
        if spikes_1 is not None and clusters_1 is not None:
            
            if spikes_0 is not None and clusters_0 is not None:
                offset = np.max(clusters_0) + 1
                clusters_1 = clusters_1 + offset
                
                # Combine both probes
                spikes = np.concatenate([spikes_0, spikes_1])
                clusters = np.concatenate([clusters_0, clusters_1])
            else:
                # Only probe 1
                spikes = spikes_1
                clusters = clusters_1
        else:
            # Only probe 0
            spikes = spikes_0
            clusters = clusters_0

        # Create histogram as usual
        spike_counts = get_spike_hist(spikes, clusters, bin_edges)
        spike_counts = filter_spike_counts(spike_counts,oa_pos, wh_pos)

    
    except Exception as e: 
        spike_counts= None
    


    session = Bunch(

        # Metadata
        subject_id=subject_id,
        date=date,
        bin_edges=bin_edges,
        exp_onset=exp_onset,
        exp_idx = exp_idx,

            # Behavioral data
        bodypart_x=bodypart_x,
        bodypart_y=bodypart_y,
        x=x,
        y=y,
        oa_speed=oa_speed,
        wh_speed=wh_speed,
        oa_pos=oa_pos,
        wh_pos=wh_pos,
        oa_onsets=oa_onsets,
        oa_offsets=oa_offsets,
        wh_onsets=wh_onsets,
        wh_offsets=wh_offsets,
        oa_running=oa_running,
        wh_running=wh_running,

        # ROI Data
        roi_x=roi_x,
        roi_y=roi_y,
        radius=radius,

        # spikes
        spike_counts =spike_counts,
    )

    
    
    return session
    

def analyze_single_session(session):
    """Analyze a single session's correlations and stability"""
    
    if session.spike_counts is None:
        return session
    
    # Compute correlations
    r_oa, r_wh = get_correlations(
        session.spike_counts, 
        session.oa_speed, 
        session.wh_speed, 
        session.oa_pos, 
        session.wh_pos
    )

    # Cross-context correlation
    valid = ~(np.isnan(r_oa) | np.isnan(r_wh))
    r_oa_wh = np.corrcoef(r_oa[valid], r_wh[valid])[0, 1] if np.sum(valid) > 1 else np.nan
    
    # Stability
    r_oa_first_half, r_oa_second_half, r_wh_first_half, r_wh_second_half, oa_stability, wh_stability = cross_validate_correlations(
        session.spike_counts,
        session.oa_pos,
        session.wh_pos,
        session.oa_speed,
        session.wh_speed, 
    )

    # Add results directly to session 
    session.r_oa = r_oa
    session.r_wh = r_wh
    session.r_oa_wh = r_oa_wh
    session.r_oa_first_half=r_oa_first_half
    session.r_oa_second_half=r_oa_second_half
    session. r_wh_first_half= r_wh_first_half
    session. r_wh_second_half= r_wh_second_half
    session.oa_stability = oa_stability
    session.wh_stability = wh_stability
    
    return session


def analyze_multiple_sessions(session_list, target_freq=10):
    """Load all sessions, then compute p-values"""
    # Load all sessions
    
    all_sessions= []
    for subject_id, date in session_list:
        try:
            session_data = load_session_data(subject_id, date, target_freq=target_freq)
            
            if session_data.spike_counts is not None:
                analyzed_session = analyze_single_session(session_data)
            
            all_sessions.append(analyzed_session)
            print(f"Loaded and analyzed: {subject_id} - {date}")

        except Exception as e:
            print(f"Failed: {subject_id} - {date}: {e}")
           
    
    # Compute p-values for each session
    complete_sessions = [s for s in all_sessions if s.spike_counts is not None]
    for i, session in enumerate(complete_sessions):
        null_sessions = all_sessions[:i] + all_sessions[i+1:] 
        if len(null_sessions) > 0:
            null_arena, null_wheel = compute_null_distributions_for_session(session, null_sessions)
            compute_p_values_from_null(session, null_arena, null_wheel)
            print(f"Computed p-values for session {i+1}/{len(complete_sessions)}")

    categorise_neurons(complete_sessions)
    
    return complete_sessions


def compute_null_distributions_for_session(session, null_sessions):
    """
    Compute null distributions for correlation analysis using behavioral data from other sessions.
    
    Parameters:
    -----------
    session_data : dict
        Current session data containing neural and behavioral data
    other_sessions : list
        List of other session data dictionaries for null generation
        
    Returns:
    --------
    null_arena : array, shape (n_null_sessions, n_neurons)
        Null arena correlations
    null_wheel : array, shape (n_null_sessions, n_neurons)
        Null wheel correlations
    """
    spike_counts = session.spike_counts
    n_neurons = spike_counts.shape[0]
    n_null = len(null_sessions)
    
    null_arena = np.zeros((n_null, n_neurons))
    null_wheel = np.zeros((n_null, n_neurons))
    
    for i, null_session in enumerate(null_sessions):
        try:
            current_length = spike_counts.shape[1]
            null_length = len(null_session.oa_speed)
            min_length = min(current_length, null_length)
            
            null_arena[i], null_wheel[i] = get_correlations(
                spike_counts[:, :min_length],
                null_session.oa_speed[:min_length],
                null_session.wh_speed[:min_length],
                null_session.oa_running[:min_length],
                null_session.wh_running[:min_length],
            )
        except Exception as e:
            print(f"Error computing null for session {i}: {e}")
            null_arena[i] = np.nan
            null_wheel[i] = np.nan
    
    return null_arena, null_wheel

def compute_p_values_from_null(session, null_arena, null_wheel):
    """
    Compute p-values for correlations using null distributions.
    
    Parameters:
    -----------
    session_data : dict
        Session data with correlation results
    null_arena : array, shape (n_null, n_neurons)
        Null arena correlations
    null_wheel : array, shape (n_null, n_neurons)
        Null wheel correlations
        
    Returns:
    --------
    session_data : dict
        Session data with added p-values
    """
    
    n_neurons = len(session.r_oa)
    n_null = null_arena.shape[0]
    
    p_vals_oa = np.zeros(n_neurons)
    p_vals_wh = np.zeros(n_neurons)
    
    for n in range(n_neurons):
        # One-tailed test based on sign
        if session.r_oa[n] >= 0:
            p_vals_oa[n] = np.sum(null_arena[:, n] >= session.r_oa[n]) / n_null
        else:
            p_vals_oa[n] = np.sum(null_arena[:, n] <= session.r_oa[n]) / n_null
            
        if session.r_wh[n] >= 0:
            p_vals_wh[n] = np.sum(null_wheel[:, n] >= session.r_wh[n]) / n_null
        else:
            p_vals_wh[n] = np.sum(null_wheel[:, n] <= session.r_wh[n]) / n_null
    
    session.p_vals_oa = p_vals_oa
    session.p_vals_wh = p_vals_wh
    
    return session

def add_p_values_to_session(session, other_sessions):
    """Add p-values to an already analyzed session"""
    null_arena, null_wheel = compute_null_distributions_for_session(session, other_sessions)
    session = compute_p_values_from_null(session, null_arena, null_wheel)
    return session

def categorise_neurons(all_sessions, alpha=0.05):
    """
    Classify neurons based on statistical significance in each context.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries with p-values
    alpha : float
        Significance threshold for FDR correction
        
    Returns:
    --------
    all_sessions : list
        Sessions with added neuron category classifications
    """

    for session in all_sessions:
        if not hasattr(session, 'r_oa') or not hasattr(session, 'p_vals_oa'):
            print(f"Skipping session {session.subject_id} - {session.date}: not fully analyzed")
            continue
            
        oa_sig,_ = fdrcorrection(session.p_vals_oa, alpha=alpha)
        wh_sig,_ = fdrcorrection(session.p_vals_wh, alpha=alpha)

        session.context_invariant = oa_sig & wh_sig & (np.sign(session.r_oa) == np.sign(session.r_wh))
        session.arena_only = oa_sig & ~wh_sig
        session.wheel_only = ~oa_sig & wh_sig
        session.context_switching = oa_sig & wh_sig & (np.sign(session.r_oa) != np.sign(session.r_wh)) 
        session.non_encoding = ~oa_sig & ~wh_sig

        
        
    return all_sessions

def get_reliability_stability(session):

        r_oa_1= session.r_oa_first_half
        r_oa_2 = session.r_oa_second_half
        r_wh_1= session.r_wh_first_half
        r_wh_2= session.r_wh_second_half
        r_oa_1_2 = session.oa_stability
        r_wh_1_2 = session.wh_stability


        r_oa_1_wh_2 = np.corrcoef(r_oa_1, r_wh_2) [0,1]
        r_oa_2_wh_1 = np.corrcoef(r_oa_2, r_wh_1) [0,1]


        session.reliability = np.sqrt(r_oa_1_2* r_wh_1_2) 

        
        z1 = np.arctanh(r_oa_1_wh_2)
        z2 = np.arctanh(r_oa_2_wh_1)

        z_mean = (z1 + z2 ) / 2
        session.stability = np.tanh(z_mean)

