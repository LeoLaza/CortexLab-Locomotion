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

def cross_validate_correlations(spike_counts, arena_mask,wheel_mask, velocity, wheel_velocity, time_bins, split = 2):
    """
    Cross-validate neural-velocity correlations using temporal splits.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Neural spike count matrix
    arena_mask : array, bool
        Arena context mask
    wheel_mask : array, bool
        Wheel context mask
    velocity : array
        Arena velocity
    wheel_velocity : array
        Wheel velocity
    time_bins : array
        Time bin edges
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

    oa_train_idx = np.where(arena_mask[:midpoint])[0]
    oa_test_idx = np.where(arena_mask[midpoint:])[0] + midpoint
    wh_train_idx = np.where(wheel_mask[:midpoint])[0]
    wh_test_idx = np.where(wheel_mask[midpoint:])[0] + midpoint

    oa_train_var = np.var(spike_counts[:, oa_train_idx], axis=1)
    oa_test_var = np.var(spike_counts[:, oa_test_idx], axis=1)
    wh_train_var = np.var(spike_counts[:, wh_train_idx], axis=1)
    wh_test_var = np.var(spike_counts[:, wh_test_idx], axis=1)
    
    # Remove neurons with zero variance in any split
    good_neurons = ~((oa_train_var == 0) | (oa_test_var == 0) | 
                     (wh_train_var == 0) | (wh_test_var == 0))
    
    spike_counts = spike_counts[good_neurons, :]
    n_neurons = spike_counts.shape[0]

    train_corr_arena = np.zeros(n_neurons)
    test_corr_arena = np.zeros(n_neurons)
    train_corr_wheel = np.zeros(n_neurons)
    test_corr_wheel = np.zeros(n_neurons)

    for i in range(n_neurons):
        try:
            train_corr_arena[i] = np.corrcoef(spike_counts[i,oa_train_idx], velocity[oa_train_idx])[0,1]
            test_corr_arena[i] = np.corrcoef (spike_counts[i,oa_test_idx],velocity [oa_test_idx])[0,1]
            train_corr_wheel[i] = np.corrcoef(spike_counts[i,wh_train_idx], wheel_velocity[wh_train_idx])[0,1]
            test_corr_wheel[i] = np.corrcoef(spike_counts[i, wh_test_idx], wheel_velocity[wh_test_idx])[0,1]

        except Exception as e:
            pass

    
    oa_stability = np.corrcoef(train_corr_arena, test_corr_arena)[0,1]
    wh_stability = np.corrcoef(train_corr_wheel, test_corr_wheel)[0,1]

    return train_corr_arena, test_corr_arena, train_corr_wheel, test_corr_wheel, oa_stability, wh_stability

def compute_null_distributions_for_session(session_data, other_sessions):
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
    spike_counts = session_data["neural_data"]["spike_counts"]
    n_neurons = spike_counts.shape[0]
    n_null = len(other_sessions)
    
    null_arena = np.zeros((n_null, n_neurons))
    null_wheel = np.zeros((n_null, n_neurons))
    
    for i, other_session in enumerate(other_sessions):
        try:
            current_length = spike_counts.shape[1]
            null_length = len(other_session["behavioral_data"]["velocity"])
            min_length = min(current_length, null_length)
            
            null_arena[i], null_wheel[i] = get_correlations(
                spike_counts[:, :min_length],
                other_session["behavioral_data"]["velocity"][:min_length],
                other_session["behavioral_data"]["wheel_velocity"][:min_length],
                other_session["behavioral_data"]["arena_mask"][:min_length],
                other_session["behavioral_data"]["wheel_mask"][:min_length],
                filter=False
            )
        except Exception as e:
            print(f"Error computing null for session {i}: {e}")
            null_arena[i] = np.nan
            null_wheel[i] = np.nan
    
    return null_arena, null_wheel

def compute_p_values_from_null(session_data, null_arena, null_wheel):
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
    arena_corrs = session_data["correlations"]["arena"]
    wheel_corrs = session_data["correlations"]["wheel"]
    n_neurons = len(arena_corrs)
    n_null = null_arena.shape[0]
    
    p_values_arena = np.zeros(n_neurons)
    p_values_wheel = np.zeros(n_neurons)
    
    for n in range(n_neurons):
        # One-tailed test based on sign
        if arena_corrs[n] >= 0:
            p_values_arena[n] = np.sum(null_arena[:, n] >= arena_corrs[n]) / n_null
        else:
            p_values_arena[n] = np.sum(null_arena[:, n] <= arena_corrs[n]) / n_null
            
        if wheel_corrs[n] >= 0:
            p_values_wheel[n] = np.sum(null_wheel[:, n] >= wheel_corrs[n]) / n_null
        else:
            p_values_wheel[n] = np.sum(null_wheel[:, n] <= wheel_corrs[n]) / n_null
    
    session_data["p_values"] = {
        "arena": p_values_arena,
        "wheel": p_values_wheel
    }
    
    return session_data

def add_p_values_to_session(session_data, other_sessions):
    """Add p-values to an already analyzed session"""
    null_arena, null_wheel = compute_null_distributions_for_session(session_data, other_sessions)
    session_data = compute_p_values_from_null(session_data, null_arena, null_wheel)
    return session_data

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
        arena_corrs =session["correlations"]["arena"]
        wheel_corrs =session["correlations"]["wheel"]
        pvals_oa = session["p_values"]["arena"]
        pvals_wh = session["p_values"]["wheel"]
            

        arena_sig,_ = fdrcorrection(pvals_oa, alpha=0.05)
        wheel_sig,_ = fdrcorrection(pvals_wh, alpha=0.05)

        categories =  {

            "context_invariant": arena_sig & wheel_sig & (np.sign(arena_corrs) == np.sign(wheel_corrs)),

            "arena_only": arena_sig & ~wheel_sig,

            "wheel_only": ~arena_sig & wheel_sig,

            "context_switching": arena_sig & wheel_sig & (np.sign(arena_corrs) != np.sign(wheel_corrs)),

            "non_encoding": ~arena_sig & ~wheel_sig,
        }

        session["categories"]= categories
        
    return all_sessions
