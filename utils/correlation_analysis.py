
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

def get_speed_correlations(spike_counts, oa_speed, wh_speed, oa_mask, wh_mask):

    # why does this not work if I use filtered_spike_counts, INVESTIGATE
    n_neurons = spike_counts.shape[0]

    
    r_oa = np.full(n_neurons, np.nan)
    r_wh = np.full(n_neurons, np.nan)

    oa_var = np.var(spike_counts[:, oa_mask], axis=1)
    wh_var = np.var(spike_counts[:, wh_mask], axis=1)

  
    
    # Remove neurons with zero variance in any split
    good_neurons = ~(oa_var == 0) & ~(wh_var == 0) 
    good_indices = np.where(good_neurons)[0]

   

    for i in good_indices:
            # Free running correlation
            r_oa[i] = np.corrcoef(oa_speed[oa_mask], spike_counts[i, oa_mask])[0, 1]
        
            # Wheel running correlation
            r_wh[i] = np.corrcoef(wh_speed[wh_mask], spike_counts[i, wh_mask])[0, 1]


    return r_oa, r_wh

def get_cross_context_correlations(r_oa, r_wh):

    valid = ~(np.isnan(r_oa) | np.isnan(r_wh))
    r_oa_wh = np.corrcoef(r_oa[valid], r_wh[valid])[0,1]

    return r_oa_wh



def get_split_half_correlations(spike_counts, speed_arena, speed_wheel, mask_arena, mask_wheel, split = 2):
    
    midpoint =  spike_counts.shape[1] // split

    arena_half1_idx = np.where(mask_arena[:midpoint])[0]
    arena_half2_idx = np.where(mask_arena[midpoint:])[0] + midpoint
    wheel_half1_idx = np.where(mask_wheel[:midpoint])[0]
    wheel_half2_idx = np.where(mask_wheel[midpoint:])[0] + midpoint

    arena_half1_var = np.var(spike_counts[:, arena_half1_idx], axis=1)
    arena_half2_var = np.var(spike_counts[:, arena_half2_idx ], axis=1)
    wheel_half1_var = np.var(spike_counts[:, wheel_half1_idx], axis=1)
    wheel_half2_var = np.var(spike_counts[:, wheel_half2_idx], axis=1)
    
    # Remove neurons with zero variance in any split
    good_neurons = ~((arena_half1_var == 0) | (arena_half2_var == 0) | 
                     (wheel_half1_var == 0) | (wheel_half2_var  == 0))
    
    spike_counts = spike_counts[good_neurons, :]
    n_neurons = spike_counts.shape[0]

    corr_arena_half1 = np.zeros(n_neurons)
    corr_arena_half2 = np.zeros(n_neurons)
    corr_wheel_half1 = np.zeros(n_neurons)
    corr_wheel_half2 = np.zeros(n_neurons)

    for i in range(n_neurons):
        try:
            corr_arena_half1[i] = np.corrcoef(spike_counts[i,arena_half1_idx], speed_arena[arena_half1_idx])[0,1]
            corr_arena_half2[i] = np.corrcoef (spike_counts[i,arena_half2_idx], speed_arena[arena_half2_idx])[0,1]
            corr_wheel_half1[i] = np.corrcoef(spike_counts[i,wheel_half1_idx], speed_wheel[wheel_half1_idx])[0,1]
            corr_wheel_half2[i] = np.corrcoef(spike_counts[i,wheel_half2_idx ], speed_wheel[wheel_half2_idx])[0,1]

        except Exception as e:
            pass

    
    reliability_arena = np.corrcoef(corr_arena_half1, corr_arena_half2)[0,1]
    reliability_wheel = np.corrcoef(corr_wheel_half1, corr_wheel_half2)[0,1]

    return corr_arena_half1, corr_arena_half2, corr_wheel_half1, corr_wheel_half2, reliability_arena, reliability_wheel

def get_reliability_stability(corr_arena_half1, corr_arena_half2, corr_wheel_half1, corr_wheel_half2, reliability_arena, reliability_wheel):

        reliability = np.sqrt(reliability_arena * reliability_wheel) 


        corr_arena1_wheel2 = np.corrcoef(corr_arena_half1, corr_wheel_half2) [0,1]
        corr_wheel1_arena2 = np.corrcoef(corr_wheel_half1, corr_arena_half2) [0,1]

        # Fisher Transform and Compute Mean 
        z1 = np.arctanh(corr_arena1_wheel2)
        z2 = np.arctanh(corr_wheel1_arena2)
        z_mean = (z1 + z2 ) / 2

        # reverse transform mean
        stability = np.tanh(z_mean)

        return reliability, stability 
