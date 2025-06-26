"""
Statistical testing and validation functions for neural-behavioral analysis.

This module handles:
- Cross-validation of neural-behavioral correlations
- Null distribution generation for significance testing
- P-value computation and multiple comparisons correction
- Neuron classification based on statistical significance
"""

import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from utils.correlation_analysis import get_correlations 

def cross_validate_correlations(spike_counts, oa_mask, wh_mask, oa_speed, wh_speed, split = 2):
    """
    Cross-validate neural-velocity correlations using temporal splits.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons)
        Neural spike count matrix
    arena_mask : array, bool
        Arena context mask
    wheel_mask : array, bool
        Wheel context mask
    velocity : array
        Arena velocity
    wheel_velocity : array
        Wheel velocity
    split : int
        Number of temporal splits (2 = split half)
        
    Returns:
    --------
    train_corr_arena : array
        Training correlations in arena
    test_corr_arena : array
        Test correlations in arena
    train_corr_wheel : array
        Training correlations on wheel
    test_corr_wheel : array
        Test correlations on wheel
    oa_stability : float
        Arena correlation stability (train-test correlation)
    wh_stability : float
        Wheel correlation stability (train-test correlation)
    """
    midpoint =  spike_counts.shape[1] // split

    oa_first_half_idx = np.where(oa_mask[:midpoint])[0]
    oa_second_half_idx = np.where(oa_mask[midpoint:])[0] + midpoint
    wh_first_half_idx = np.where(wh_mask[:midpoint])[0]
    wh_second_half_idx = np.where(wh_mask[midpoint:])[0] + midpoint

    oa_train_var = np.var(spike_counts[:, oa_first_half_idx], axis=1)
    oa_test_var = np.var(spike_counts[:, oa_second_half_idx], axis=1)
    wh_train_var = np.var(spike_counts[:, wh_first_half_idx], axis=1)
    wh_test_var = np.var(spike_counts[:, wh_second_half_idx], axis=1)
    
    # Remove neurons with zero variance in any split
    good_neurons = ~((oa_train_var == 0) | (oa_test_var == 0) | 
                     (wh_train_var == 0) | (wh_test_var == 0))
    
    spike_counts = spike_counts[good_neurons, :]
    n_neurons = spike_counts.shape[0]

    r_oa_first_half = np.zeros(n_neurons)
    r_oa_second_half = np.zeros(n_neurons)
    r_wh_first_half = np.zeros(n_neurons)
    r_wh_second_half = np.zeros(n_neurons)

    for i in range(n_neurons):
        try:
            r_oa_first_half[i] = np.corrcoef(spike_counts[i,oa_first_half_idx], oa_speed[oa_first_half_idx])[0,1]
            r_oa_second_half[i] = np.corrcoef (spike_counts[i,oa_second_half_idx ], oa_speed[oa_second_half_idx ])[0,1]
            r_wh_first_half[i] = np.corrcoef(spike_counts[i,wh_first_half_idx], wh_speed[wh_first_half_idx])[0,1]
            r_wh_second_half[i] = np.corrcoef(spike_counts[i,wh_second_half_idx ], wh_speed[wh_second_half_idx])[0,1]

        except Exception as e:
            pass

    
    oa_stability = np.corrcoef(r_oa_first_half, r_oa_second_half)[0,1]
    wh_stability = np.corrcoef(r_wh_first_half, r_wh_second_half)[0,1]

    return r_oa_first_half, r_oa_second_half, r_wh_first_half, r_wh_second_half, oa_stability, wh_stability

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
            
        oa_sig,_ = fdrcorrection(session.p_vals_oa, alpha=0.05)
        wh_sig,_ = fdrcorrection(session.p_vals_wh, alpha=0.05)

        session.context_invariant = oa_sig & wh_sig & (np.sign(session.r_oa) == np.sign(session.r_wh))
        session.arena_only = oa_sig & ~wh_sig
        session.wheel_only = ~oa_sig & wh_sig
        session.context_switching = oa_sig & wh_sig & (np.sign(session.r_oa) != np.sign(session.r_wh)) 
        session.non_encoding = ~oa_sig & ~wh_sig

        
        
    return all_sessions
