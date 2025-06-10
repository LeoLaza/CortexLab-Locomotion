"""
Visualization functions for neural-behavioral analysis.

This module handles:
- Neural activity visualizations (heatmaps, correlation plots)
- Behavioral analysis plots (position, velocity, context detection)
- Cross-validation and statistical test visualizations
- Multi-session comparison plots
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.neural_processing import normalize_spike_counts
from utils.statistical_testing import cross_validate_correlations

def plot_correlation_histogram(corrs):
    """
    Plot histogram of neural-behavioral correlations.
    
    Parameters:
    -----------
    corrs : array
        Correlation coefficients to plot
    """
    plt.figure(figsize=(12, 6))
    plt.hist(corrs, bins= 100, alpha=0.7)
    plt.xlabel('Correlation Coefficient', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks( fontsize=16)
    plt.yticks( fontsize=16)
    plt.show()

def plot_sorted_spike_counts(sorting_argument, velocity, wheel_velocity, wheel_mask, arena_mask, spike_counts, w_start=0, w_end=1500):
     
    """
    Plot spike counts sorted by correlation strength with velocity traces.
    
    Parameters:
    -----------
    sorting_argument : array
        Values to sort neurons by (e.g., correlation coefficients)
    velocity : array
        Arena velocity
    wheel_velocity : array
        Wheel velocity
    wheel_mask : array, bool
        Wheel context mask
    arena_mask : array, bool
        Arena context mask
    spike_counts : array
        Neural spike count matrix
    w_start, w_end : int
        Time window for plotting
    """
     
    # Sort neurons by sorting argument
    sorted_idx = np.argsort(sorting_argument)[::-1]
    spike_counts = normalize_spike_counts(spike_counts)
    sorted_spike_counts = spike_counts[sorted_idx, :]
    
    # Prepare velocity data for plotting
    velocity[~arena_mask] = 0
    wheel_velocity[~wheel_mask] = 0
    
    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), 
                                       gridspec_kw={'height_ratios': [1, 1, 3]},
                                       sharex=True)
    
    # Plot velocities
    ax1.plot(velocity[w_start:w_end], color='green', alpha=0.8)
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
   #ax1.set_ylabel('Free Running Velocity (cm/s)', color='green')

    ax2.plot(wheel_velocity[w_start:w_end], color='purple', alpha=0.8)
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax2.set_ylabel('Wheel Running Velocity (cm/s)', color='purple')

    ax3.matshow(sorted_spike_counts[:, w_start:w_end], aspect= 'auto', cmap='gray_r', vmin=0, vmax=np.percentile(spike_counts, 90))
    ax3.set_yticks([])
    plt.tight_layout()
    plt.show()
    

def plot_wheel_arena_corr(arena_corrs, wheel_corrs):
    """
    Plot arena vs wheel correlations with identity line.
    
    Parameters:
    -----------
    arena_corrs : array
        Arena correlation coefficients
    wheel_corrs : array
        Wheel correlation coefficients
    """
    id_corr = np.corrcoef(arena_corrs, wheel_corrs)[0, 1]
    print(f'Correlation between free and wheel running neurons: {id_corr}')

    plt.figure(figsize=(10, 8))
    plt.scatter(arena_corrs, wheel_corrs, alpha=0.5)
    plt.xlabel('Open Arena Correlation Coefficient', fontsize=18)
    plt.ylabel('Wheel Running Correlation Coefficient', fontsize=18)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xticks(np.arange(-0.5, 0.51, 0.25), fontsize=16)
    plt.yticks(np.arange(-0.5, 0.51, 0.25), fontsize=16)
    plt.show()


def plot_cross_validation(train_corr_arena, test_corr_arena, train_corr_wheel, test_corr_wheel, oa_stability, wh_stability):
    """
    Plot cross-validation results for correlation stability.
    
    Parameters:
    -----------
    train_corr_arena, test_corr_arena : array
        Training and test arena correlations
    train_corr_wheel, test_corr_wheel : array
        Training and test wheel correlations
    oa_stability, wh_stability : float
        Stability coefficients
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train_corr_arena, test_corr_arena, alpha=0.5)
    
       
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xticks(np.arange(-1, 1.1, 0.4), fontsize=16)
    plt.yticks(np.arange(-1, 1.1, 0.4), fontsize=16)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f'correlation train/test arena r={oa_stability}')
       

    plt.subplot(1, 2, 2)
    plt.scatter(train_corr_wheel, test_corr_wheel, alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xticks(np.arange(-1, 1.1, 0.4), fontsize=16)
    plt.yticks(np.arange(-1, 1.1, 0.4), fontsize=16)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.title(f'correlation train/test wheel r={wh_stability}')
    plt.show()

def plot_masked_positions(binned_x, binned_y, arena_mask, wheel_mask):
    """
    Plot position data colored by behavioral context.
    
    Parameters:
    -----------
    binned_x, binned_y : array
        Position coordinates
    arena_mask, wheel_mask : array, bool
        Context masks
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(binned_x[arena_mask], binned_y[arena_mask], c='green', s=2)
    plt.scatter(binned_x[wheel_mask], binned_y[wheel_mask], c='purple', s=2)
    plt.show

def plot_all_correlation_distributions(all_sessions, n_sessions):
    """
    Plot correlation distributions across multiple sessions.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries
    n_sessions : int
        Number of sessions
    """
    x_positions_arena = []
    x_positions_wheel = []
    arena_data = []
    wheel_data = []
    labels = []

    for i, session in enumerate(all_sessions):
        arena_corrs = session["correlations"]["arena"]
        wheel_corrs = session["correlations"]["wheel"]
        
        arena_corrs = arena_corrs[~np.isnan(arena_corrs)]
        wheel_corrs = wheel_corrs[~np.isnan(wheel_corrs)]
        
        x_positions_arena.append(i*2)
        x_positions_wheel.append(i*2 + 0.8)
        
        # Store data
        arena_data.append(arena_corrs)
        wheel_data.append(wheel_corrs)
        
        # Create label
        subject = session["metadata"]["subject_id"]
        date = session["metadata"]["date"]
        labels.append(f"{subject}\n{date}")

    fig, ax1 = plt.subplots(figsize=(2*n_sessions, 8))
    violin_arena = ax1.violinplot(arena_data,positions = x_positions_arena, widths=0.7)
    

    for pc in violin_arena['bodies']:
        pc.set_facecolor('green')
        pc.set_alpha(0.7)
    

    ax1.set_xticks([i*2 + 0.8 for i in range(n_sessions)])
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize = 18)
    ax1.set_ylabel('Correlation', fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_title('Distribution of Neural Correlations in Arena by Session', fontsize=24)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(2*n_sessions, 8))
    violin_wheel = ax2.violinplot(wheel_data, positions = x_positions_wheel, widths=0.7)

    for pc in violin_wheel['bodies']:
        pc.set_facecolor('purple')
        pc.set_alpha(0.7)

    ax2.set_xticks([i*2 + 0.8 for i in range(n_sessions)])
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize = 18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_ylabel('Correlation', fontsize = 18)
    ax2.set_title('Distribution of Neural Correlations on Wheel by Session', fontsize=24)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)


    plt.tight_layout()
    plt.show()


def plot_all_stability(all_sessions, n_sessions):
    """
    Plot correlation stability across sessions.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries
    n_sessions : int
        Number of sessions
    """

    fig, ax = plt.subplots(figsize=(max(8, n_sessions), 8))

    x_positions= np.arange(n_sessions)
    arena_stability = []
    wheel_stability = []
    labels= []
    

    for i, session in enumerate(all_sessions):
        arena_stability.append(session["stability"]["arena"])
        wheel_stability.append(session["stability"]["wheel"])

        subject = session["metadata"]["subject_id"]
        date = session["metadata"]["date"]
        labels.append(f"{subject}\n{date}")

    ax.scatter(x_positions, arena_stability, color ='green', alpha =0.7, label = 'Arena', s=45)
    ax.scatter(x_positions, wheel_stability, color ='purple', alpha =0.7, label = 'Wheel', s=45)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Stability', fontsize = 18)
    ax.set_title('Neural Stability Comparison Across Sessions', fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_all_cross_context_corrs(all_sessions, n_sessions):
    """
    Plot cross-context correlations across sessions.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries
    n_sessions : int
        Number of sessions
    """

    fig, ax = plt.subplots(figsize=(max(8, n_sessions), 8))

    x_positions= np.arange(n_sessions)
    cross_context = []
    labels= []
    

    for i, session in enumerate(all_sessions):
        cross_context.append(session["correlations"]["cross_context"])

        subject = session["metadata"]["subject_id"]
        date = session["metadata"]["date"]
        labels.append(f"{subject}\n{date}")

    ax.scatter(x_positions, cross_context, color ='red', alpha =0.7, label = 'Arena', s=45)
    

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize = 18)
    ax.set_ylabel('Cross-Context Correlation', fontsize = 18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title('Cross-Context Correlation Comparison Across Sesssion', fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_categories(all_sessions, n_sessions):
    """
    Plot neuron category proportions across sessions.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries
    n_sessions : int
        Number of sessions
    """

    fig, ax = plt.subplots(figsize=(max(8, n_sessions), 8))

    x_positions= np.arange(n_sessions)
    labels= []

    category_names = ["context_invariant", "arena_only", "wheel_only", 
                     "context_switching", "non_encoding"]
    
    colors = ['purple', 'pink', 'red', 'orange', 'yellow']
    
    category_counts = {name: [] for name in category_names}

    for i, session in enumerate(all_sessions):
        
        categories = session["categories"] 
        
        total_neurons = len(categories["context_invariant"])

        for cat_name in category_names:
            count = np.sum(categories[cat_name])
            proportion = count / total_neurons if total_neurons > 0 else 0
            category_counts[cat_name].append(proportion)

        subject = session["metadata"]["subject_id"]
        date = session["metadata"]["date"]
        labels.append(f"{subject}\n{date}")

    bottom = np.zeros(n_sessions)
    
    for cat_name, color in zip(category_names, colors):
        proportions = category_counts[cat_name]
        ax.bar(x_positions, proportions, bottom=bottom, 
               label=cat_name.replace('_', ' ').title(), 
               color=color, alpha=0.7, width=0.6)
        bottom += proportions

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Proportion of Neurons', fontsize=18)
    ax.set_title('Neuron Category Proportions Across Sessions', fontsize=20)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_single_session(session_data, w_start=0, w_end=1200):
    """
    Create comprehensive plots for a single session.
    
    Parameters:
    -----------
    session_data : dict
        Session data dictionary
    w_start, w_end : int
        Time window for plotting
    """

    time_bins = session_data["metadata"]["time_bins"]
    spike_counts = session_data["neural_data"]["spike_counts"]
    binned_x = session_data["behavioral_data"]["binned_x"]
    binned_y = session_data["behavioral_data"]["binned_y"]
    velocity = session_data["behavioral_data"]["velocity"]
    wheel_velocity = session_data["behavioral_data"]["wheel_velocity"]
    arena_mask = session_data["behavioral_data"]["arena_mask"]
    wheel_mask = session_data["behavioral_data"]["wheel_mask"]
    arena_corrs = session_data["correlations"]["arena"]
    wheel_corrs = session_data["correlations"]["wheel"]
    plt.hist(velocity[velocity > 0], bins=100)
    plt.xlabel('Velocity')
    plt.ylabel('Count')
    plt.axvline(1.0, color='g', label='Threshold')
    plt.legend()
    plot_masked_positions(binned_x, binned_y, arena_mask, wheel_mask)
    plot_masked_positions(binned_x[w_start:w_end], binned_y[w_start:w_end], arena_mask[w_start:w_end], wheel_mask[w_start:w_end])
    plot_sorted_spike_counts(arena_corrs, velocity, wheel_velocity, wheel_mask, arena_mask, spike_counts, w_start, w_end)
    plot_sorted_spike_counts(wheel_corrs, velocity, wheel_velocity, wheel_mask, arena_mask, spike_counts, w_start, w_end)

    plot_correlation_histogram(arena_corrs)
    plot_correlation_histogram(wheel_corrs)

    plot_wheel_arena_corr(arena_corrs, wheel_corrs)

    train_corr_arena, test_corr_arena, train_corr_wheel, test_corr_wheel, oa_stability, wh_stability = cross_validate_correlations(spike_counts, arena_mask,wheel_mask, velocity, wheel_velocity, time_bins, split = 2)
    plot_cross_validation(train_corr_arena, test_corr_arena, train_corr_wheel, test_corr_wheel, oa_stability, wh_stability)


def plot_all_sessions(all_sessions):
    """
    Create comprehensive plots for multiple sessions.
    
    Parameters:
    -----------
    all_sessions : list
        List of session data dictionaries
    """

    n_sessions= len(all_sessions)

    plot_all_correlation_distributions(all_sessions, n_sessions)
    plot_all_cross_context_corrs(all_sessions, n_sessions)
    plot_all_stability(all_sessions, n_sessions)
    plot_categories(all_sessions, n_sessions)