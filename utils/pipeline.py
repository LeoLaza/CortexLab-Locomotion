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

def load_and_process_session(subject_id, date, target_freq=None):    
    
    exp_kwargs = {
        'subject': subject_id, 
        'expDate': date
    }


    # load data
    ONE = load_ONE(exp_kwargs)
    dlc_df = get_dlc_df(subject_id, date)
    exp_path = get_experiment_path(ONE, dlc_frame_count= len(dlc_df), cam_fps=60)
    exp_onset, cam_timestamps = get_cam_timestamps(exp_kwargs, rigName='poppy-stim')
    bodypart_x, bodypart_y = preprocess_dlc_data(dlc_df)
    x, y = calculate_median_position(bodypart_x, bodypart_y)
    rotary_timestamps, rotary_position = get_rotary_position(exp_path)
    
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

    print(np.where(np.isnan(wh_speed)))

    
    
    # identify bouts of running in either context
    oa_pos, wh_pos = get_position_masks(x, y, roi_x, roi_y, radius, subject_id)
    oa_running, wh_running, oa_onsets, oa_offsets, wh_onsets, wh_offsets = get_speed_masks(oa_speed, wh_speed, oa_pos, wh_pos, oa_thresh = 2.0, wh_thresh=0.8)
    

    # generate spike histogram 
    spikes_0, clusters_0, spikes_1, clusters_1 = load_probes(exp_kwargs)
    
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
    spike_counts = filter_spike_counts(spike_counts, oa_pos, wh_pos)

    session = Bunch(
        # Metadata
        subject_id=subject_id,
        date=date,
        bin_edges=bin_edges,
        exp_onset=exp_onset,
        
        # Neural data
        spike_counts=spike_counts,
        
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
        radius=radius
    )
    
    return session
    

def analyze_single_session(session):
    """Analyze a single session's correlations and stability"""
    
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
    all_sessions = []
    for subject_id, date in session_list:
        try:
            session_data = load_and_process_session(subject_id, date, target_freq=target_freq)
            session_data = analyze_single_session(session_data)
            all_sessions.append(session_data)
            print(f"Loaded and analyzed: {subject_id} - {date}")
        except Exception as e:
            print(f"Failed: {subject_id} - {date}: {e}")
    
    # Compute p-values for each session
    for i, session in enumerate(all_sessions):
        other_sessions = all_sessions[:i] + all_sessions[i+1:]
        if len(other_sessions) > 0:
            null_arena, null_wheel = compute_null_distributions_for_session(session, other_sessions)
            compute_p_values_from_null(session, null_arena, null_wheel)
            print(f"Computed p-values for session {i+1}/{len(all_sessions)}")

    categorise_neurons(all_sessions)
    
    return all_sessions