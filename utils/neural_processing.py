import numpy as np


def get_spike_hist(spike_times, spike_clusters,time_bins):
    """
    Convert spike times and clusters into binned spike count matrix.
    
    Parameters:
    -----------
    spike_times : array
        Timestamps of individual spikes
    spike_clusters : array  
        Cluster ID for each spike
    time_bins : array
        Time bin edges for histogramming
        
    Returns:
    --------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Spike count matrix
    """
    spike_counts, _, _ = np.histogram2d(
        spike_clusters, 
        spike_times,
        bins=[np.arange(0, len(np.unique(spike_clusters))+1), time_bins],
        )
    return spike_counts

def normalize_spike_counts(spike_counts):
    """
    Z-score normalize spike counts across time for each neuron.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Raw spike count matrix
        
    Returns:
    --------
    normalized_counts : array, shape (n_neurons, n_time_bins)
        Z-score normalized spike counts
    """
    spike_counts_norm = (spike_counts - np.mean(spike_counts, axis=1, keepdims=True)) / np.std(spike_counts, axis=1, keepdims=True)
    return spike_counts_norm


def filter_spike_counts(spike_counts, arena_mask, wheel_mask):
     """
    Remove neurons with zero variance in either behavioral context.
    
    Parameters:
    -----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Spike count matrix
    arena_mask : array, shape (n_time_bins,)
        Boolean mask for arena periods
    wheel_mask : array, shape (n_time_bins,)  
        Boolean mask for wheel periods
        
    Returns:
    --------
    filtered_counts : array, shape (n_valid_neurons, n_time_bins)
        Spike counts with zero-variance neurons removed
    """
     arena_var = np.var(spike_counts[:, arena_mask], axis=1)
     wheel_var = np.var(spike_counts[:, wheel_mask], axis=1)

     zero_var = (arena_var == 0) | (wheel_var == 0)
     spike_counts = spike_counts[~zero_var, :]

     return spike_counts