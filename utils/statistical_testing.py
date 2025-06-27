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

