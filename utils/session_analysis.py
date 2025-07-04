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
from .correlation_analysis import *
from .data_loading_and_preprocessing import *
from .decoding_analysis import *



def load_session_data(subject_id, date, target_freq=10):

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
    _, roi_x, roi_y, radius = get_ROI(subject_id, date)
    oa_pos, wh_pos = get_position_masks(x, y, roi_x, roi_y, radius, subject_id)
    
    # compute speed in either context 
    oa_speed = calculate_oa_speed(x, y, bin_width)
    wh_speed = calculate_wh_speed(rotary_position, bin_width)
    

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
    


    metadata = Bunch(

        subject_id=subject_id,
        date=date,
        bin_width = bin_width,
    )

    behavior = Bunch(
        speed_arena=oa_speed,
        speed_wheel =wh_speed,
        mask_arena=oa_pos,
        mask_wheel=wh_pos,
    )

    
    
    return metadata, behavior, spike_counts
    

def perform_correlation_analyses(behavior, spike_counts):

    # think of validation for behavior  
    if spike_counts is None:
        print("No spike counts were provided")
        return None

    speed_arena= behavior.speed_arena
    speed_wheel= behavior.speed_wheel
    mask_arena= behavior.mask_arena 
    mask_wheel= behavior.mask_wheel

    corr_arena, corr_wheel = get_speed_correlations(
        spike_counts, 
        speed_arena, 
        speed_wheel, 
        mask_arena, 
        mask_wheel
    )

    # Cross-context correlation
    valid = ~(np.isnan(corr_arena) | np.isnan(corr_wheel))
    corr_cross_context = np.corrcoef(corr_arena[valid], corr_wheel[valid])[0, 1] if np.sum(valid) > 1 else np.nan
    
    # Stability
    corr_arena_half1, corr_arena_half2, corr_wheel_half1, corr_wheel_half2, reliability_arena, reliability_wheel = get_split_half_correlations(
        spike_counts,
        speed_arena, 
        speed_wheel, 
        mask_arena, 
        mask_wheel 
    )

    reliability, stability = get_reliability_stability(
        corr_arena_half1, 
        corr_arena_half2, 
        corr_wheel_half1, 
        corr_wheel_half2, 
        reliability_arena, 
        reliability_wheel )

    correlation_results = Bunch(
  
        arena=corr_arena,
        wheel=corr_wheel,
        cross_context=corr_cross_context,
        
        
        arena_half1=corr_arena_half1,
        arena_half2=corr_arena_half2,
        wheel_half1=corr_wheel_half1,
        wheel_half2=corr_wheel_half2,
        
        
        reliability_arena=reliability_arena,
        reliability_wheel=reliability_wheel,
        reliability= reliability,
        stability=stability
    )
        
        
    return correlation_results


def perform_decoding_analyses(behavior, spike_counts, leaveout=None, alpha=None):

    if spike_counts is None:
        print("No spike counts were provided")
        return None
    
    
    spike_counts = normalize_spike_counts(spike_counts)
    
    speed_arena= behavior.speed_arena
    speed_wheel= behavior.speed_wheel
    mask_arena= behavior.mask_arena 
    mask_wheel= behavior.mask_wheel


    # split data into train and test sets
    train_data, test_data = split_for_decoding(spike_counts, speed_arena, speed_wheel, mask_arena, mask_wheel)

    # train models
    arena_model = train_model(train_data.spike_counts_arena, train_data.speed_arena, alpha=alpha)
    wheel_model = train_model(train_data.spike_counts_wheel, train_data.speed_wheel, alpha=alpha)
    
    # compare model weights
    weights_arena = arena_model.coef_
    weights_wheel = wheel_model.coef_
    corr_weights = np.corrcoef(weights_arena, weights_wheel)[0,1]
    cosine_similarity_weights = np.dot(weights_arena, weights_wheel) / (np.linalg.norm(weights_arena) * np.linalg.norm(weights_wheel))
 
    # test model performance within context
    r2_arena = arena_model.score(test_data.spike_counts_arena, test_data.speed_arena)
    r2_wheel = wheel_model.score(test_data.spike_counts_wheel, test_data.speed_wheel)

    # test model performance cross-context  
    r2_arena_to_wheel = arena_model.score(test_data.spike_counts_wheel, test_data.speed_wheel)
    r2_wheel_to_arena = wheel_model.score(test_data.spike_counts_arena, test_data.speed_arena)

    leaveout_results = None

    if leaveout:
        leaveout_results = compute_leaveout_analysis(train_data, test_data, weights_arena, weights_wheel, alpha=alpha)
        
    decoding_results = Bunch(
    models=Bunch(
        arena=arena_model,
        wheel=wheel_model
    ),
    performance=Bunch(
        within_context=Bunch(arena=r2_arena, wheel=r2_wheel),
        cross_context=Bunch(arena_to_wheel=r2_arena_to_wheel, 
                           wheel_to_arena=r2_wheel_to_arena)
    ),
    weights=Bunch(
        correlation=corr_weights,
        cosine_similarity=cosine_similarity_weights
    ),
    leaveout=leaveout_results 
)

    return decoding_results


def analyze_single_session(subject_id, date, correlation=True, decoding=True, leaveout=True):
    # Load data
    metadata, behavior, spike_counts = load_session_data(subject_id, date)

    correlation_results = None
    decoding_results = None
    
    # Run analyses
    if correlation:
        correlation_results = perform_correlation_analyses(behavior, spike_counts)

    if decoding:
        decoding_results = perform_decoding_analyses(behavior, spike_counts, 
                                                leaveout=leaveout, alpha=1.0)
    
    return Bunch(
        metadata=metadata,
        spike_counts=spike_counts,
        behavior=behavior,  
        correlations=correlation_results,
        decoding=decoding_results,
    )

def analyze_all_sessions(session_list, correlation=True, decoding=True, leaveout= True):
    return [analyze_single_session(subject_id, date, correlation=correlation, decoding=decoding, leaveout=leaveout) for subject_id, date in session_list]






