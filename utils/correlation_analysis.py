"""
Neural-behavioral correlation analysis functions.

This module handles:
- Computing correlations between neural activity and locomotion
- Cross-context correlation comparisons
"""




import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

def get_correlations(spike_counts, oa_speed, wh_speed, oa_mask, wh_mask, filter=True):
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
    r_oa = np.zeros(spike_counts.shape[0])
    r_wh = np.zeros(spike_counts.shape[0])

    for i in range(spike_counts.shape[0]):
            # Free running correlation
            r_oa[i] = np.corrcoef(oa_speed[oa_mask], spike_counts[i, oa_mask])[0, 1]
        
            # Wheel running correlation
            r_wh[i] = np.corrcoef(wh_speed[wh_mask], spike_counts[i, wh_mask])[0, 1]


    if filter:
        r_nan =  np.isnan(r_oa) | np.isnan(r_wh)
        r_oa = r_oa[~r_nan]
        r_wh = r_wh[~r_nan]
        

    return r_oa, r_wh

def get_cross_context_correlations(r_oa, r_wh):
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

    valid = ~(np.isnan(r_oa) | np.isnan(r_wh))
    r_oa_wh = np.corrcoef(r_oa[valid], r_wh[valid])[0,1]

    return r_oa_wh

def run_PCA(spike_counts):
    binsxneurons = spike_counts.T
    binsxneurons = zscore(binsxneurons, axis=0)
    pca = PCA(n_components = 10)
    pctrajectories = pca.fit_transform(binsxneurons)

    return pctrajectories