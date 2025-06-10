"""
High-level pipeline functions for complete session analysis.

This module handles:
- Complete single session processing workflows
- Multi-session analysis coordination
- Integration of all analysis modules
"""

import numpy as np
from .behavioral_analysis import *
from .neural_processing import *
from .correlation_analysis import *
from .statistical_testing import *
from .roi_detection import *
from .data_io import *

def load_and_process_session(subject_id, date):    
    # load data
    ONE = load_ONE(subject_id, date)
    exp_folder, exp_num = get_experiment_path(ONE)
    exp_kwargs = {
        'subject': subject_id, 
        'expDate': date
    }
    start_time, timestamps = get_timestamps(exp_kwargs, rigName='poppy-stim')
    time_bins, bin_centers, bin_width = create_time_bins(timestamps, start_time)
    dlc_df, scorer = get_DLC_data(subject_id, date, start_time)
    rotary_timestamps, rotary_position = get_rotary_metadata(exp_folder, bin_centers)

    # define context
    frame, center_x, center_y, radius = get_ROI(subject_id, date)
    plot_ROI(frame, center_x, center_y, radius, subject_id, date)

    # compute velocity
    x, y = calculate_median_position(dlc_df, scorer)
    binned_x, binned_y = bin_median_positions(x, y, timestamps, start_time, bin_centers)
    velocity = calculate_velocity(binned_x, binned_y, bin_width)
    wheel_velocity = calculate_wheel_velocity(rotary_position, bin_width)
    arena_mask, wheel_mask = get_position_masks(binned_x, binned_y, center_x, center_y, radius, subject_id)
    oa_running, wh_running = get_speed_masks(velocity, wheel_velocity, arena_mask, wheel_mask)
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
    spike_counts = get_spike_hist(spikes, clusters, time_bins)
    spike_counts = filter_spike_counts(spike_counts, arena_mask, wheel_mask)

    session_data = {
        "metadata": {
            "subject_id": subject_id,
            "date":  date,
            "time_bins": time_bins
        },

        "neural_data": {
            "spike_counts": spike_counts,
        },

        "behavioral_data": {
            "binned_x": binned_x,
            "binned_y": binned_y, 
            "velocity": velocity,
            "wheel_velocity": wheel_velocity,
            "arena_mask": oa_running,
            "wheel_mask": wh_running,
        },
    }
    
    return session_data

def analyze_single_session(session_data):
    """Analyze a single session's correlations and stability"""
    spike_counts = session_data["neural_data"]["spike_counts"]
    behavioral_data = session_data["behavioral_data"]
    velocity = behavioral_data["velocity"]
    wheel_velocity = behavioral_data["wheel_velocity"]
    arena_mask = behavioral_data["arena_mask"]
    wheel_mask = behavioral_data["wheel_mask"]
    time_bins = session_data["metadata"]["time_bins"]

    # Compute correlations
    arena_corrs, wheel_corrs = get_correlations(spike_counts, velocity, wheel_velocity, arena_mask, wheel_mask)
    
    # Cross-context correlation
    valid = ~(np.isnan(arena_corrs) | np.isnan(wheel_corrs))
    cross_context_corr = np.corrcoef(arena_corrs[valid], wheel_corrs[valid])[0, 1] if np.sum(valid) > 1 else np.nan
    
    # Stability
    _, _, _, _, oa_stability, wh_stability = cross_validate_correlations(
        spike_counts, arena_mask, wheel_mask, velocity, wheel_velocity, time_bins
    )

    # Add results to session data
    session_data["correlations"] = {
        "arena": arena_corrs,
        "wheel": wheel_corrs,
        "cross_context": cross_context_corr
    }
    
    session_data["stability"] = {
        "arena": oa_stability,
        "wheel": wh_stability
    }
    
    return session_data


def analyze_multiple_sessions(session_list):
    """Load all sessions, then compute p-values"""
    # Load all sessions
    all_sessions = []
    for subject_id, date in session_list:
        try:
            session_data = load_and_process_session(subject_id, date)
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
            session = compute_p_values_from_null(session, null_arena, null_wheel)
            print(f"Computed p-values for session {i+1}/{len(all_sessions)}")
    
    return all_sessions