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
from utils.data_io import preprocess_dlc_data
from utils.behavioral_analysis import calculate_median_position

def plot_correlation_histogram(r):
    """
    Plot histogram of neural-behavioral correlations.
    
    Parameters:
    -----------
    corrs : array
        Correlation coefficients to plot
    """
    plt.figure(figsize=(12, 6))
    plt.hist(r, bins= 100, alpha=0.7)
    plt.xlabel('Correlation Coefficient', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks( fontsize=16)
    plt.yticks( fontsize=16)
    plt.show()

def plot_sorted_spike_counts(sorting_argument, oa_speed, wh_speed, wh_mask, oa_mask, spike_counts, w_start=0, w_end=1500):
     
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
    sorted_spike_counts = spike_counts[sorted_idx, :]

    # Prepare velocity data for plotting
    oa_speed[~oa_mask] = 0
    wh_speed[~wh_mask] = 0

    n_neurons = len(sorted_spike_counts)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), 
                                       gridspec_kw={'height_ratios': [1, 1, 3]},
                                       sharex=True)

    
    # Plot velocities
    ax1.plot(oa_speed[w_start:w_end], color='green', alpha=0.8)
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
   #ax1.set_ylabel('Free Running Velocity (cm/s)', color='green')

    ax2.plot(wh_speed[w_start:w_end], color='purple', alpha=0.8)
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax2.set_ylabel('Wheel Running Velocity (cm/s)', color='purple')


    ax3.matshow(sorted_spike_counts[:, w_start:w_end], aspect= "auto", cmap='gray_r', vmin=0, vmax=np.percentile(spike_counts, 90), interpolation="none")
    ax3.set_yticks([])
    ax3.set_ylim(n_neurons - 0.5, -0.5)
    
    plt.tight_layout()
    plt.show()
    

def plot_wheel_arena_corr(r_oa, r_wh):
    """
    Plot arena vs wheel correlations with identity line.
    
    Parameters:
    -----------
    arena_corrs : array
        Arena correlation coefficients
    wheel_corrs : array
        Wheel correlation coefficients
    """
    id_corr = np.corrcoef(r_oa, r_wh)[0, 1]
    print(f'Correlation between free and wheel running neurons: {id_corr}')

    plt.figure(figsize=(10, 8))
    plt.scatter(r_oa, r_wh, alpha=0.5)
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

def plot_masked_positions(x, y, arena_mask, wheel_mask):
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
    plt.scatter(x[arena_mask], y[arena_mask], c='green', s=2)
    plt.scatter(x[wheel_mask], y[wheel_mask], c='purple', s=2)
    plt.show

def plot_correlation_distributions(all_sessions, n_sessions, axes=None):

    n_sessions = len(all_sessions)
    standalone = axes is None

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6

    if standalone:
        fig, axes = plt.subplots(2, 1, figsize=(2*n_sessions + 2, 10))
        show_plot = True

        if n_sessions == 1:
            axes = axes.reshape(2, 1)

    arena_data = [session.r_oa[~np.isnan(session.r_oa)] for session in all_sessions]
    wheel_data = [session.r_wh[~np.isnan(session.r_wh)] for session in all_sessions]


    positions = np.arange(n_sessions) * 1.5

    
    violin_arena = axes[0].violinplot(arena_data,positions = positions, widths=0.7)
    for pc in violin_arena['bodies']:
            pc.set_facecolor('green')
            pc.set_alpha(0.7)
    
    axes[0].set_xlim(-0.75, positions[-1] + 0.75 if n_sessions > 1 else 0.75)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_xticks([])  
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    violin_wheel = axes[1].violinplot(wheel_data, positions=positions, widths=0.7)
    for pc in violin_wheel['bodies']:
            pc.set_facecolor('purple')
            pc.set_alpha(0.7)
            
    axes[1].set_xlim(-0.75, positions[-1] + 0.75 if n_sessions > 1 else 0.75)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_xticks(positions)  # Keep tick for label placement
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

        
    if standalone:
        axes[1].set_xticklabels([f'{s.subject_id}\n{s.date}' for s in all_sessions], 
                               rotation=45, ha='right', fontsize=18)
    
    else:
        axes[1].set_xticklabels([])

    if standalone:
        plt.suptitle('Neural Correlation Distributions', fontsize=18)
        plt.tight_layout()
        plt.show()
    
    return axes

   
def plot_all_stability(all_sessions, axes=None):
    """Plot stability scatter plots for all sessions"""
    n_sessions = len(all_sessions)
    standalone = axes is None 

    if standalone:
        fig, axes = plt.subplots(2, n_sessions, figsize=(6*n_sessions, 8))
        if n_sessions == 1:
            axes = axes.reshape(2, 1)

    for i, session in enumerate(all_sessions):
        # Arena stability plot (top row)
        axes[0, i].scatter(session.r_oa_first_half, session.r_oa_second_half, alpha=0.5, color='green')
        axes[0, i].axhline(0, color='gray', linestyle='--')
        axes[0, i].axvline(0, color='gray', linestyle='--')
        axes[0, i].set_xlim(-1, 1)
        axes[0, i].set_ylim(-1, 1)
        axes[0, i].set_xticks(np.arange(-1, 1.1, 0.4))  
        axes[0, i].set_yticks(np.arange(-1, 1.1, 0.4))  
        
        # Wheel stability
        axes[1, i].scatter(session.r_wh_first_half, session.r_wh_second_half, alpha=0.5, color='purple')
        axes[1, i].axhline(0, color='gray', linestyle='--')
        axes[1, i].axvline(0, color='gray', linestyle='--')
        axes[1, i].set_xlim(-1, 1)
        axes[1, i].set_ylim(-1, 1)
        axes[1, i].set_xticks(np.arange(-1, 1.1, 0.4))  
        axes[1, i].set_yticks(np.arange(-1, 1.1, 0.4))  
        
        axes[0, i].set_title(f'Arena r={session.oa_stability:.3f}', fontsize=35)
        axes[1, i].set_title(f'Wheel r={session.wh_stability:.3f}', fontsize=35)
        
        if i == 0:
            axes[0, i].set_ylabel('2nd Half', fontsize=18)
            axes[1, i].set_ylabel('2nd Half', fontsize=18)
        axes[0, i].set_xlabel('1st Half', fontsize=18)
        axes[1, i].set_xlabel('1st Half', fontsize=18)
    
        if standalone:
            
            axes[1, i].text(0, -1.4, f'{session.subject_id}\n{session.date}', 
                          ha='center', va='top', fontsize=18, transform=axes[1, i].transData)
    if standalone:
        plt.suptitle('Neural Stability: Train vs Test Correlations', fontsize=18)
        plt.tight_layout()
        plt.show()
        
    return axes

   


def plot_all_wheel_arena_corr(all_sessions, axes=None):
    """Plot cross-correlation between arena and wheel for all sessions"""
    n_sessions = len(all_sessions)
    standalone = axes is None
    
    if standalone:
        fig, axes = plt.subplots(1, n_sessions, figsize=(6*n_sessions, 8), 
                                sharey=True, sharex=True)
        if n_sessions == 1:
            axes = [axes]
        
    elif n_sessions == 1 and not isinstance(axes, list):
        axes = [axes]
    # Plot each session
    for i, session in enumerate(all_sessions):
        ax = axes[i]
        arena_corrs = session.r_oa
        wheel_corrs = session.r_wh
        
        # Plot
        ax.scatter(arena_corrs, wheel_corrs, alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(np.arange(-0.5, 0.51, 0.25))
        ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
        ax.set_title(f'r = {session.r_oa_wh:.3f}', fontsize=35)
    
        if i == 0:
            ax.set_ylabel('Wheel Correlation', fontsize=18)
        ax.set_xlabel('Arena Correlation', fontsize=18)
        
        
        if standalone:
            ax.text(0, -0.7, f'{session.subject_id}\n{session.date}', 
                   ha='center', va='top', fontsize=18, transform=ax.transData)
    
    if standalone:
        plt.suptitle('Cross-Context Correlations', fontsize=18)
        plt.tight_layout()
        plt.show()
        
    return axes
    


def plot_categories(all_sessions, n_sessions, axes=None):

    n_sessions= len(all_sessions)
    standalone= axes is None
 
    if standalone:
        fig, axes = plt.subplots(1, n_sessions, figsize=(6*n_sessions, 8))  
        if n_sessions == 1:  
            axes = [axes]
        show_plot = True
    elif n_sessions == 1 and not isinstance(axes, list):
        axes = [axes]

    category_names = ["context_invariant", "arena_only", "wheel_only", 
                     "context_switching", "non_encoding"]
    
    colors = ['purple', 'pink', 'red', 'orange', 'yellow']
    
    
    for i, (session, ax) in enumerate(zip(all_sessions, axes)):
        total_neurons = len(getattr(session, category_names[0]))
        proportions = []
        
        for cat_name in category_names:
            category_data = getattr(session, cat_name)
            count = np.sum(category_data)
            proportion = count / total_neurons if total_neurons > 0 else 0
            proportions.append(proportion)
        
        
        wedges, texts, autotexts = ax.pie(proportions, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
            autotext.set_weight('bold')
        
        
        ax.set_title('Neuron Categories', fontsize=18)

    
    if standalone:
            ax.text(0, -1.5, f"{session.subject_id}\n{session.date}", 
                   ha='center', va='top', fontsize=18, transform=ax.transData)

    if axes:
        axes[-1].legend(wedges, [name.replace('_', ' ').title() for name in category_names],
                       loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)

    if standalone:
        plt.suptitle('Neuron Category Proportions', fontsize=18)
        plt.tight_layout()
        plt.show()
        
    return axes
        
        
    
  

    

def plot_single_session(session, w_start=0, w_end=1200):
    """
    Create comprehensive plots for a single session.
    
    Parameters:
    -----------
    session_data : dict
        Session data dictionary
    w_start, w_end : int
        Time window for plotting
    """

    spike_counts = session.spike_counts
    x = session.x
    y = session.y
    oa_speed= session.oa_speed
    wh_speed= session.wh_speed
    oa_pos = session.oa_pos
    wh_pos= session.wh_pos
    r_oa= session.r_oa
    r_wh = session.r_wh
    r_oa_first_half =session.r_oa_first_half
    r_oa_second_half =session.r_oa_second_half
    r_wh_first_half =session.r_wh_first_half 
    r_wh_second_half= session.r_wh_second_half
    oa_stability=session.oa_stability
    wh_stability=session.wh_stability 

    plt.hist(oa_speed[oa_speed > 0], bins=100)
    plt.xlabel('Velocity')
    plt.ylabel('Count')
    plt.legend()
    plot_masked_positions(x, y, oa_pos, wh_pos)
    #plot_masked_positions(x[w_start:w_end], y[w_start:w_end], oa_pos[w_start:w_end], wh_pos[w_start:w_end])
    plot_sorted_spike_counts(r_oa, oa_speed, wh_speed, wh_pos, oa_pos, spike_counts, w_start, w_end)
    plot_sorted_spike_counts(r_wh, oa_speed, wh_speed, wh_pos, oa_pos, spike_counts, w_start, w_end)
    

    plot_correlation_histogram(r_oa)
    plot_correlation_histogram(r_wh)

    plot_wheel_arena_corr(r_oa, r_wh)

    
    plot_cross_validation(r_oa_first_half, r_oa_second_half, r_wh_first_half, r_wh_second_half, oa_stability, wh_stability)



def plot_all_sessions(all_sessions):
 
    n_sessions = len(all_sessions)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(6*n_sessions, 60))
    
    # Create subplot grid
    gs = fig.add_gridspec(6, n_sessions, height_ratios=[3, 3, 1, 1, 1, 1], 
                         hspace=0.3, wspace=0.3)
    
    ax_arena_violin = fig.add_subplot(gs[0, :])
    ax_wheel_violin = fig.add_subplot(gs[1, :])
    axes_arena_stability = [fig.add_subplot(gs[2, i]) for i in range(n_sessions)]
    axes_wheel_stability = [fig.add_subplot(gs[3, i]) for i in range(n_sessions)]
    axes_cross = [fig.add_subplot(gs[4, i]) for i in range(n_sessions)]
    axes_categories = [fig.add_subplot(gs[5, i]) for i in range(n_sessions)]
    
    # Plot using existing functions - they won't add session labels
    plot_correlation_distributions(all_sessions, n_sessions, axes=[ax_arena_violin, ax_wheel_violin])
    plot_all_stability(all_sessions, axes=np.array([axes_arena_stability, axes_wheel_stability]))
    plot_all_wheel_arena_corr(all_sessions, axes=axes_cross)
    plot_categories(all_sessions, n_sessions, axes=axes_categories)
    
    # ADDED: Add session labels only at the bottom
    for i, session in enumerate(all_sessions):
        axes_categories[i].text(0, -1.5, f'{session.subject_id}\n{session.date}', 
                               ha='center', va='top', fontsize=35, fontweight='bold',
                               transform=axes_categories[i].transData)
    
    plt.suptitle('Comprehensive Neural Analysis Across Sessions', fontsize=20, y=0.995)
    plt.tight_layout()
    plt.show()
    
    return fig
    

def plot_PCA(pctrajectories, arena_mask, wheel_mask):
     
    plt.figure(figsize=(15, 5))

    plt.subplot(1,2,1)

    plt.scatter(pctrajectories[arena_mask, 0],pctrajectories[arena_mask, 1], label="Arena", s=1, cmap="Reds", zorder=2)
    plt.scatter(pctrajectories[wheel_mask, 0],pctrajectories[wheel_mask, 1], label="Wheel", s=1, cmap="Blues",zorder=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()


    plt.subplot(1,2,2)

    plt.scatter(pctrajectories[arena_mask, 0],pctrajectories[arena_mask, 1],label="Arena", s=1, cmap="Reds")
    plt.scatter(pctrajectories[wheel_mask, 0],pctrajectories[wheel_mask, 1], label="Wheel", s=1, cmap="Blues")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

def plot_likelihood_distributions(likelihood, quality_thresh, selected_bodyparts):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, bodypart in enumerate(selected_bodyparts):
        bodypart_likelihood = likelihood[bodypart]
        bodypart_likelihood.hist(bins=50, alpha=0.7, ax=axes[i])
        axes[i].set_title(f'{bodypart} - Likelihood Distribution')
        axes[i].set_xlabel('Likelihood')
        axes[i].set_ylabel('Frequency')
        percentile = (bodypart_likelihood < quality_thresh).mean() * 100
        axes[i].axvline(quality_thresh, color='red', linestyle='--', 
                    label=f'{quality_thresh} threshold\n({percentile:.1f}th percentile)')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_position_changes(position_changes, selected_bodyparts, max_distance):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, bodypart in enumerate(selected_bodyparts):
        position_changes[bodypart].hist(bins=50, alpha=0.7, ax=axes[i])
        axes[i].set_title(f'{bodypart} - Position Changes')
        axes[i].set_xlabel('Euclidean Distance (pixels)')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(0,20)
        axes[i].axvline(max_distance, color='red', linestyle='--', label=f'Distance threshold')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_dlc_pre_post(raw_position_changes, processed_position_changes, raw_median_change, processed_median_change, selected_bodyparts, w_start=0, w_end=7200):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    
    for bodypart in selected_bodyparts:
        axes[0,0].plot(raw_position_changes[bodypart][w_start:w_end], alpha=0.6, label=bodypart)
    axes[0,0].set_title('Raw - Individual Bodypart Changes')
    axes[0,0].set_ylabel('Distance (pixels)')
    axes[0,0].set_ylim(0, 30)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    for bodypart in selected_bodyparts:
        axes[0,1].plot(processed_position_changes[bodypart][w_start:w_end], alpha=0.6, label=bodypart)
    axes[0,1].set_title('Processed - Individual Bodypart Changes')
    axes[0,1].set_ylabel('Distance (pixels)')
    axes[0,1].set_ylim(0, 30)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    

    axes[1,0].plot(raw_median_change[w_start:w_end], 'r-', linewidth=2)
    axes[1,0].set_title('Raw - Median Position Changes')
    axes[1,0].set_xlabel('Frame')
    axes[1,0].set_ylabel('Distance (pixels)')
    axes[1,0].set_ylim(0, 20)
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(processed_median_change[w_start:w_end], 'b-', linewidth=2)
    axes[1,1].set_title('Processed - Median Position Changes')
    axes[1,1].set_xlabel('Frame')
    axes[1,1].set_ylabel('Distance (pixels)')
    axes[1,1].set_ylim(0, 20)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_dlc_analyses(dlc_df, quality_thresh = 0.90, selected_bodyparts = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3'], max_distance = 10, w_start =0, w_end=7200):

    
    bodypart_pos= dlc_df.loc[:, (selected_bodyparts, slice(None))]
    likelihood = bodypart_pos.xs('likelihood', level='coords', axis=1)
    plot_likelihood_distributions(likelihood, quality_thresh, selected_bodyparts)
    
    raw_position_changes = {}
    raw_x = bodypart_pos.xs('x', level='coords', axis=1)  
    raw_y = bodypart_pos.xs('y', level='coords', axis=1) 

    for bodypart in selected_bodyparts:
        x_diff = raw_x[bodypart].diff()
        y_diff = raw_y[bodypart].diff()
        raw_euclidean_dist = np.sqrt(x_diff**2 + y_diff**2)
        raw_position_changes[bodypart] = raw_euclidean_dist

    raw_median_x = raw_x.median(axis=1)
    raw_median_y = raw_y.median(axis=1)
    raw_median_change = np.sqrt(raw_median_x.diff()**2 + raw_median_y.diff()**2)

    plot_position_changes(raw_position_changes, selected_bodyparts, max_distance)


    processed_x, processed_y = preprocess_dlc_data(dlc_df, quality_thresh=quality_thresh, selected_bodyparts=selected_bodyparts, max_distance=15)
    processed_x_median, processed_y_median = calculate_median_position(processed_x, processed_y)  
    
    processed_position_changes = {}
    for bodypart in selected_bodyparts:
        x_diff = processed_x[bodypart].diff()
        y_diff = processed_y[bodypart].diff()
        processed_position_changes[bodypart] = np.sqrt(x_diff**2 + y_diff**2)
    
    processed_median_change = np.sqrt(processed_x_median.diff()**2 + processed_y_median.diff()**2)
    
    plot_dlc_pre_post(raw_position_changes, processed_position_changes, raw_median_change, processed_median_change, selected_bodyparts, w_start, w_end)
    

def plot_reliability_stability(all_sessions):
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)

    reliability_values = []
    stability_values = []
    brain_regions = []

    for session in all_sessions:
        
        reliability_values.append(session.reliability)
        stability_values.append(session.stability)
        
    
        if session.subject_id[:2] == 'EB':
            brain_regions.append('Hippocampus')
        else:
            brain_regions.append('Secondary Motor Cortex')


    reliability_values = np.array(reliability_values)
    stability_values = np.array(stability_values)
    brain_regions = np.array(brain_regions)


    colors = {
        'Hippocampus':"#132B97",  
        'Secondary Motor Cortex':"#690B0B" 
    }  
    for region in ['Hippocampus', 'Secondary Motor Cortex']:
        mask = brain_regions == region
        
        ax.scatter(reliability_values[mask], 
                stability_values[mask], 
                c=colors[region],
                s=100,  
                alpha=0.7,  
                edgecolors='none',  
                linewidth=0.5,
                label=region,
                zorder=3)

    ax.plot([0, 1], [0, 1], 'k-', linewidth=1.5, alpha=0.5, zorder=1)

    ax.set_xlim(-0.10, 1.05)
    ax.set_ylim(-0.10, 1.05)

    ax.set_xlabel('reliability within context', fontsize=16)
    ax.set_ylabel('stability between contexts', fontsize=16)

    # Add legend
    legend = ax.legend(loc='upper left', 
                    frameon=True, 
                    fancybox=False,
                    edgecolor='black',
                    fontsize=12)
    legend.get_frame().set_linewidth(0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)


    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    

