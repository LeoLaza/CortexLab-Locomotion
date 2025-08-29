
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
from types import SimpleNamespace as Bunch

def get_speed_correlations(spike_counts, oa_speed, wh_speed, oa_mask, wh_mask):


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

def compute_speed_tuning(spike_counts, speed, mask, speed_bins, dt=0.1):
    context_speed = speed[mask]
    context_spikes = spike_counts[:, mask]
    
    bin_centers = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    n_neurons = spike_counts.shape[0]
    n_bins = len(bin_centers)
    
    # initialize
    firing_rates = np.zeros((n_neurons, n_bins))
    firing_sem = np.zeros((n_neurons, n_bins))
    
    # digitize speed into speed bins
    speed_indices = np.digitize(context_speed, speed_bins) - 1
    
    # calculate firing rates
    for i in range(n_bins):
        idx = speed_indices == i
        if np.sum(idx) > 30:  # at least 3 seconds of data 
            rates_in_bin = context_spikes[:, idx] / dt
            firing_rates[:, i] = np.mean(rates_in_bin, axis=1)
            firing_sem[:, i] = np.std(rates_in_bin, axis=1) / np.sqrt(np.sum(idx))
        else:
            firing_rates[:, i] = np.nan
            firing_sem[:, i] = np.nan
    
    return bin_centers, firing_rates, firing_sem


def run_permutation_test(all_session_results, alpha=0.05):


    for i, experimental_session in enumerate(all_session_results):
            
            # only compute p-values for sessions with neural recordings
            if experimental_session.spike_counts is None:
                continue
                
            null_arena = []
            null_wheel = []

            for j, control in enumerate(all_session_results):
                if i == j:  # Skip self
                    continue
            
                # Handle length mismatches between experimental spikes and control behavior
                spikes_length = experimental_session.spike_counts.shape[1]
                speed_length = len(control.behavior.speed_arena)
                min_length = min(spikes_length, speed_length)

                
                control_corr_arena, control_corr_wheel = get_speed_correlations(
                    experimental_session.spike_counts[:, :min_length],
                    control.behavior.speed_arena[:min_length],
                    control.behavior.speed_wheel[:min_length],
                    control.behavior.mask_arena[:min_length],
                    control.behavior.mask_wheel[:min_length]
            )

                null_arena.append(control_corr_arena)
                null_wheel.append(control_corr_wheel)
        
            null_arena = np.array(null_arena)
            null_wheel = np.array(null_wheel)
    
            n_neurons = experimental_session.spike_counts.shape[0]
            n_null = null_arena.shape[0]
    
            p_vals_arena = np.zeros(n_neurons)
            p_vals_wheel = np.zeros(n_neurons)
            
            for n in range(n_neurons):
                # One-tailed test based on sign
                obs_arena = experimental_session.correlations.arena[n]
                if obs_arena >= 0:
                    p_vals_arena[n] = np.sum(null_arena[:, n] >= obs_arena) / n_null
                else:
                    p_vals_arena[n] = np.sum(null_arena[:, n] <= obs_arena) / n_null

                obs_wheel = experimental_session.correlations.wheel[n]
                if obs_wheel >= 0:
                    p_vals_wheel[n] = np.sum(null_wheel[:, n] >= obs_wheel) / n_null
                else:
                    p_vals_wheel[n] = np.sum(null_wheel[:, n] <= obs_wheel) / n_null
            
            sig_arena, p_vals_arena_corrected = fdrcorrection(p_vals_arena, alpha)
            sig_wheel, p_vals_wheel_corrected = fdrcorrection(p_vals_wheel, alpha)

            experimental_session.permutation = Bunch(
            n_comparisons=n_null,
            alpha=alpha,
            p_vals=Bunch(arena=p_vals_arena, wheel=p_vals_wheel),
            significant=Bunch(arena=sig_arena, wheel=sig_wheel)
        )




def categorise_neurons(corr_arena, corr_wheel, sig_arena, sig_wheel):


        return Bunch (
             context_invariant= sig_arena & sig_wheel & (np.sign(corr_arena) == np.sign(corr_wheel)),
             arena_only= sig_arena & ~sig_wheel,
             wheel_only= ~sig_arena & sig_wheel,
             context_switching= sig_arena & sig_wheel & (np.sign(corr_arena) != np.sign(corr_wheel)),
             non_encoding = ~sig_arena & ~sig_wheel
        )
    
