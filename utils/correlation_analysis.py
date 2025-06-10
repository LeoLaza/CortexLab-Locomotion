"""
Neural-behavioral correlation analysis functions.

This module handles:
- Computing correlations between neural activity and locomotion
- Cross-context correlation comparisons
"""




import numpy as np

def get_correlations(spike_counts_filtered, velocity, wheel_velocity, arena_mask, wheel_mask, filter=True):
    """
    Compute correlations between neural activity and velocity in each context.
    
    Parameters:
    -----------
    spike_counts_filtered : array, shape (n_neurons, n_time_bins)
        Preprocessed spike count matrix
    velocity : array
        Arena velocity in cm/s
    wheel_velocity : array
        Wheel velocity in cm/s
    arena_mask : array, bool
        Boolean mask for arena periods
    wheel_mask : array, bool
        Boolean mask for wheel periods
    filter : bool
        Whether to filter out neurons with NaN correlations
        
    Returns:
    --------
    arena_corrs : array
        Neural-velocity correlations during arena periods
    wheel_corrs : array
        Neural-velocity correlations during wheel periods
    """
    arena_corrs = np.zeros(spike_counts_filtered.shape[0])
    wheel_corrs = np.zeros(spike_counts_filtered.shape[0])

    for i in range(spike_counts_filtered.shape[0]):
    # Free running correlation
        arena_corrs[i] = np.corrcoef(velocity[arena_mask], spike_counts_filtered[i, arena_mask])[0, 1]
        
        # Wheel running correlation
        wheel_corrs[i] = np.corrcoef(wheel_velocity[wheel_mask], spike_counts_filtered[i, wheel_mask])[0, 1]

    if filter:
        corrs_nan =  np.isnan(wheel_corrs) | np.isnan(arena_corrs)
        wheel_corrs = wheel_corrs[~corrs_nan]
        arena_corrs = arena_corrs[~corrs_nan]

    return arena_corrs, wheel_corrs

def get_cross_context_correlations(arena_corrs, wheel_corrs):
    """
    Compute correlation between neural responses across contexts.
    
    Parameters:
    -----------
    arena_corrs : array
        Neural-velocity correlations in arena
    wheel_corrs : array
        Neural-velocity correlations on wheel
        
    Returns:
    --------
    cross_context_corr : float
        Correlation between arena and wheel neural responses
    """

    valid = ~(np.isnan(wheel_corrs) | np.isnan(arena_corrs))
    cross_context_corr = np.corrcoef(arena_corrs[valid], wheel_corrs[valid])[0,1]

    return cross_context_corr