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
from utils.correlation_analysis import get_speed_correlations
from types import SimpleNamespace as Bunch

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
                    control.speed_arena[:min_length],
                    control.speed_wheel[:min_length],
                    control.mask_arena[:min_length],
                    control.mask_wheel[:min_length]
            )

                null_arena.append(control_corr_arena)
                null_wheel.append(control_corr_wheel)
        
            null_arena = np.array(null_arena)
            null_wheel = np.array(null_wheel)
    
            n_neurons = len(experimental_session.spike_counts[0])
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
                if p_vals_wheel[n] >= 0:
                    p_vals_wheel[n] = np.sum(null_wheel[:, n] >= obs_wheel) / n_null
                else:
                    p_vals_wheel[n] = np.sum(null_wheel[:, n] <= obs_wheel[n]) / n_null
            
            sig_arena, p_vals_arena_corrected = fdrcorrection(p_vals_arena, alpha)
            sig_wheel, p__vals_wheel_corrected = fdrcorrection(p_vals_wheel, alpha)

            permutation_results = Bunch (
                n_comparisons = n_null,
                alpha= alpha,
                p_vals= Bunch(arena=p_vals_arena, wheel=p_vals_wheel),
                significant = Bunch(arena=sig_arena, wheel=sig_wheel)
            )
            
            return permutation_results



def categorise_neurons(corr_arena, corr_wheel, sig_arena, sig_wheel):


        return Bunch (
             context_invariant= sig_arena & sig_wheel & (np.sign(corr_arena) == np.sign(corr_wheel)),
             arena_only= sig_arena & ~sig_wheel,
             wheel_only= ~sig_arena & sig_wheel,
             contect_switching= sig_arena & sig_wheel & (np.sign(corr_arena) != np.sign(corr_wheel)),
             non_encoding = ~sig_arena & ~sig_wheel
        )
    

        
        
    