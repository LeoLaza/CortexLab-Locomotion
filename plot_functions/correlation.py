
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_raster_pos_neg(spike_counts, speed, mask, correlations, w_start, w_end, n_neurons=100, color='#141414'):
    
    """
    Plot mouse speed and raster of neural activity for positively and negatively correlated neurons
    for selected context (arena or wheel).

    Parameters:
    spike_counts : numpy array - spike counts for each neuron (rows) and frame (columns)
    speed : numpy array - speed of mouse for each frame
    mask : boolean array - identifies context
    correlations : numpy array - correlation with speed for each neuron
    w_start : int - start index for window to plot
    w_end : int - end index for window to plot
    n_neurons : int - number of neurons to display for positive and negative correlations
    color : str - color for significant correlations, for arena use '#195A2C', for wheel use '#7D0C81'
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), 
                                       gridspec_kw={'height_ratios': [0.5, 1, 1]},
                                       sharex=True)

    # set speed to Nan when not in context
    speed = speed.copy()
    speed[~mask] = 0

    # get neurons with highest positive and negative correlations
    pos_idx = np.where(correlations > 0)[0]
    pos_sort = pos_idx[np.argsort(correlations[pos_idx])[::-1]][:n_neurons]
    neg_idx = np.where(correlations < 0)[0]
    neg_sort = neg_idx[np.argsort(correlations[neg_idx])][:n_neurons]
        

   # determine vmax for each subplot, adjust percentile as needed for better visualization
    spike_sets = [
            spike_counts[pos_sort, w_start:w_end],
            spike_counts[neg_sort, w_start:w_end]
        ]
        
   
    vmax_values = []
    for spikes in spike_sets:
        if len(spikes) > 0:
            non_zero_spikes = spikes[spikes > 0]
            if len(non_zero_spikes) > 0:
            # use percentile of non-zero values only
                vmax = np.percentile(non_zero_spikes, 30)  #ADJUST FOR VISUALISATION
            else:
                vmax = 1
        vmax_values.append(vmax)
    else:
        vmax_values.append(1)
        
    # plot speed
    axes[0].plot(speed[w_start:w_end], color=color, linewidth=1.5)
    axes[0].set_ylabel('speed\n(cm/s)', fontsize=16)
    axes[0].set_yticks(np.arange(0,50.1,50))
    axes[0].tick_params(axis='y', labelsize=16)

    max_speed = np.nanmax(speed[w_start:w_end])
    if np.isnan(max_speed) or max_speed == 0:
        y_max = 50
    else:
        y_max = math.ceil(int(max_speed) // 5 + 1) * 5

    axes[0].set_ylim(0, y_max)
    axes[0].set_yticks([0, y_max])  # Only 0 and the maximum
        
        
    # remove spines of speed axis
    for ax in axes[:1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        
    # plot rasters for positive and negative correlations
    labels = [' (+)', ' (-)']
    for i, (ax, spikes, label, vmax) in enumerate(zip(axes[1:], spike_sets, labels, vmax_values)):
        if len(spikes) > 0:
            # use individual vmax for each subplot
            ax.matshow(spikes, aspect='auto', cmap='Grays', vmin=0, vmax=vmax, interpolation='none')
            if i == len(spike_sets) - 1:
                ax.set_ylabel(f'{len(spikes)} neurons', fontsize=16, loc="bottom")
            
            # axis formatting
            ax.text(-0.07, 0.5, label, transform=ax.transAxes, 
                    rotation=90, va='center', ha='center', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # transform window size to minutes fo x-label
        window_minutes = (w_end - w_start) / 10 / 60
        axes[-1].set_xlabel(f'{window_minutes:.0f} min', fontsize=16, loc='left')

    
    plt.tight_layout()
    plt.show()




def plot_arena_reliability(corr_arena_half1, corr_arena_half2):

    # plot first half vs second half correlations between neural activity and arena speed
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(corr_arena_half1, corr_arena_half2, alpha=0.7, color='#195A2C', zorder=2, s=50)

    # axis formatting
    ax.set_xlabel('correlation arena half 1', fontsize=20)
    ax.set_ylabel('correlation arena half 2', fontsize=20)
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=20)

    # remove spines
    ax.spines[['right', 'top']].set_visible(False)

    # optional display reliability as title
    reliability = np.corrcoef(corr_arena_half1, corr_arena_half2)[0,1]
    ax.set_title(f'reliability: {reliability:.2f}', fontsize=20)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()




def plot_wheel_reliability(corr_wheel_haf1, corr_wheel_half2):

    # plot first half vs second half correlations between neural activity and wheel speed
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(corr_wheel_haf1, corr_wheel_half2, alpha=0.7, color='#7D0C81', zorder=2, s=50)

    # axis formatting
    ax.set_xlabel('correlation wheel half 1', fontsize=20)
    ax.set_ylabel('correlation wheel half 2', fontsize=20)
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=20)

    # remove spines
    ax.spines[['right', 'top']].set_visible(False)

    # optional display reliability as title
    reliability = np.corrcoef(corr_wheel_haf1, corr_wheel_half2)[0,1]
    ax.set_title(f'reliability: {reliability:.2f}', fontsize=20)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()




def plot_arena_half1_vs_wheel_half2(corr_arena_half1, corr_wheel_half2):

    # plot first half correlations between neural activity and arena speed vs wheel speed
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(corr_arena_half1, corr_wheel_half2, alpha=0.7, color= '#141414' , zorder=2, s=50)

    # axis formatting
    ax.set_xlabel('correlation arena half 1', fontsize=20, color='#195A2C')
    ax.set_ylabel('correlation wheel half 2', fontsize=20, color='#7D0C81')
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=20)

    # remove spines
    ax.spines[['right', 'top']].set_visible(False)

    # optional display correlation as title
    r_ar1__wh2 = np.corrcoef(corr_arena_half1, corr_wheel_half2)[0,1]
    ax.set_title(f'cross-context correlation: {r_ar1__wh2:.2f}', fontsize=20)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()




def plot_arena_half2_vs_wheel_half1(corr_arena_half2, corr_wheel_half1):

    # plot second half correlations between neural activity and arena speed vs wheel speed
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(corr_arena_half2, corr_wheel_half1, alpha=0.7, color= '#141414' , zorder=2, s=50)

    # axis formatting
    ax.set_xlabel('correlation arena half 2', fontsize=20, color='#195A2C')
    ax.set_ylabel('correlation wheel half 1', fontsize=20, color='#7D0C81')
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=20)

    # remove spines
    ax.spines[['right', 'top']].set_visible(False)

    # optional display correlation as title
    r_ar2__wh1 = np.corrcoef(corr_arena_half2, corr_wheel_half1)[0,1]
    ax.set_title(f'cross-context correlation: {r_ar2__wh1:.2f}', fontsize=20)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()




def plot_correlation_histogram(correlations,  p_vals, color='#141414'):
    
    """
    Plot histogram of correlations with significance indicated by color.
    
    Parameters:
    
    correlations : numpy array - correlation with speed for each neuron
    p_vals : numpy array - permutation test p-value for each neuron
    color : str - color for significant correlations, for arena use '#195A2C', for wheel use '#7D0C81'
    """
        
    # plot histogram of correlations with significance indicated by color 
    fig, ax = plt.subplots(figsize=(4, 8)) 
    sig_pos_mask = np.where((correlations >0) & (p_vals < 0.05))
    sig_neg_mask = np.where((correlations < 0) & (p_vals < 0.05))
    neutral_mask = np.where( p_vals > 0.05)
    ax.hist(correlations[sig_pos_mask], alpha=0.7, color=color, bins=40, orientation='horizontal')
    ax.hist(correlations[sig_neg_mask], alpha=0.7, color= color, bins=40, orientation= 'horizontal')
    ax.hist(correlations[neutral_mask], alpha=0.7, color='grey', bins=40, orientation='horizontal')

       
    # axis formatting
    ax.set_ylabel('Pearson correlation', fontsize=16)
    ax.set_xlabel('number of neurons', fontsize=16)
    ax.set_yticks(np.arange(-0.5, 0.505, 0.5))
    ax.set_xticks(np.arange(0,100.05, 50))
    ax.tick_params(axis='both', labelsize=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()




def plot_reliability_stability(all_session_results):

    """
    For each session plot summary statistic for reliability versus stability.
    
    Parameters:

    all_session_results : list - contains result object for each session with metadata and correlation results
    """
    
    # gather reliability and stability for each session distinguished by brain region

    reliability_values = []
    stability_values = []
    brain_regions = []
    subject_ids = []

    for result in all_session_results:
        if result.spike_counts is None:
            continue
        
        reliability_values.append(result.correlations.reliability)
        stability_values.append(result.correlations.stability)
        subject_ids.append(result.metadata.subject_id) 

        # classify brain region based on subject ID
        if result.metadata.subject_id == 'EB036':
            brain_regions.append('hippocampus')
        elif result.metadata.subject_id == 'EB037':
            brain_regions.append('striatum')
        else:
            brain_regions.append('secondary motor cortex')

    reliability_values = np.array(reliability_values)
    stability_values = np.array(stability_values)
    brain_regions = np.array(brain_regions)
    subject_ids = np.array(subject_ids)

    # define colors for each region/mouse
    colors = {
        'hippocampus': "#A5CB5D",  
        'striatum': "#E37A2A",
        'MOs(1)': "#660D0D",  
        'MOs(2)': "#9F3A3A",    
        'MOs(3)': "#D26E6E"   
    }
    
    # map MOs subject IDs to labels
    mos_mapping = {
        'AV043': 'MOs(1)',
        'GB011': 'MOs(2)', 
        'GB012': 'MOs(3)'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # plot hippocampus
    mask = brain_regions == 'hippocampus'
    if np.any(mask):
        ax.scatter(reliability_values[mask], 
                stability_values[mask], 
                c=colors['hippocampus'],
                s=150,  
                alpha=0.7,  
                edgecolors='none',  
                linewidth=0.5,
                label='hippocampus',
                zorder=3)
    
    # plot striatum
    mask = brain_regions == 'striatum'
    if np.any(mask):
        ax.scatter(reliability_values[mask], 
                stability_values[mask], 
                c=colors['striatum'],
                s=150,  
                alpha=0.7,  
                edgecolors='none',  
                linewidth=0.5,
                label='striatum',
                zorder=3)
    
    # plot each MOs mouse separately 
    for subject_id, mos_label in mos_mapping.items():
        mask = (brain_regions == 'secondary motor cortex') & (subject_ids == subject_id)
        if np.any(mask):
            ax.scatter(reliability_values[mask], 
                    stability_values[mask], 
                    c=colors[mos_label],
                    s=150,  
                    alpha=0.7,  
                    edgecolors='none',  
                    linewidth=0.5,
                    label=mos_label,
                    zorder=3)
            

    # plot unity line
    ax.plot([0, 1], [0, 1], 'k-', linewidth=1, alpha=0.5, zorder=1, ls='--')

    # axis formatting
    ax.set_xlim(-0.10, 1.05)
    ax.set_ylim(-0.10, 1.05)
    ax.set_xlabel('reliability', fontsize=20)
    ax.set_ylabel('stability', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))

    # add legend
    legend = ax.legend(loc='upper left', 
                    frameon=False, 
                    fancybox=False,
                    edgecolor='black',
                    fontsize=18)
    legend.get_frame().set_linewidth(0.5)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()


def plot_speed_tuning(centers_arena, tuning_arena, sem_arena, centers_wheel, tuning_wheel, sem_wheel,
                      corr_arena, corr_wheel, sig_arena, sig_wheel, speed_arena, speed_wheel,
                        category):
    
    # define neuron category masks
    arena_only_mask = sig_arena & ~sig_wheel
    wheel_only_mask = ~sig_arena & sig_wheel
    context_invariant_mask = sig_arena & sig_wheel & (np.sign(corr_arena) == np.sign(corr_wheel))
    context_switching_mask = sig_arena & sig_wheel & (np.sign(corr_arena) != np.sign(corr_wheel))
    non_encoding_mask = ~sig_arena & ~sig_wheel
    
    # calculate combined metric for some categories
    combined_metric = np.abs(corr_arena) + np.abs(corr_wheel)
    
    # select mask, metric, and percentiles based on category
    if category == 'arena-only':
        selected_mask = arena_only_mask
        selected_indices = np.where(selected_mask)[0]
        metric = np.abs(corr_arena[selected_mask])
        percentiles = [100, 75]
        
    elif category == 'wheel-only':
        selected_mask = wheel_only_mask
        selected_indices = np.where(selected_mask)[0]
        metric = np.abs(corr_wheel[selected_mask])
        percentiles = [100, 75]
        
    elif category == 'context-invariant':
        selected_mask = context_invariant_mask
        selected_indices = np.where(selected_mask)[0]
        metric = combined_metric[selected_mask]
        percentiles = [100, 75]
        
    elif category == 'context-switching':
        selected_mask = context_switching_mask
        selected_indices = np.where(selected_mask)[0]
        metric = combined_metric[selected_mask]
        percentiles = [100, 75]
        
    elif category == 'non-encoding':
        selected_mask = non_encoding_mask
        selected_indices = np.where(selected_mask)[0]
        metric = combined_metric[selected_mask]
        percentiles = [0, 25]
        
    else:
        raise ValueError(f"Unknown category: {category}")
    
    # check if we have neurons in this category
    if len(selected_indices) == 0:
        print(f"No neurons found in category: {category}")
        return None

    
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6))

    for idx, pct in enumerate(percentiles):
        ax = axes[idx]
        
        # find the percentile value among selected neurons
        if pct == 100:
            percentile_value = np.max(metric)
        elif pct == 0:
            percentile_value = np.min(metric)
        else:
            percentile_value = np.percentile(metric, pct)
        
        # find which selected neuron is closest to this percentile
        idx_in_selected = np.argmin(np.abs(metric - percentile_value))
        neuron_idx = selected_indices[idx_in_selected]
        
        # get max rate for this neuron
        max_rate = max(np.nanmax(tuning_arena[neuron_idx, :]), 
                    np.nanmax(tuning_wheel[neuron_idx, :]))
        
        # plot arena with shaded error region 
        ax.fill_between(centers_arena, 
                        tuning_arena[neuron_idx, :] - sem_arena[neuron_idx, :],
                        tuning_arena[neuron_idx, :] + sem_arena[neuron_idx, :],
                        color='#195A2C', alpha=0.3)
        ax.plot(centers_arena, tuning_arena[neuron_idx, :], 
                'o', color='#195A2C', markersize=3)
        
        # plot wheel with shaded error region
        ax.fill_between(centers_wheel, 
                        tuning_wheel[neuron_idx, :] - sem_wheel[neuron_idx, :],
                        tuning_wheel[neuron_idx, :] + sem_wheel[neuron_idx, :],
                        color='#7D0C81', alpha=0.3)
        ax.plot(centers_wheel, tuning_wheel[neuron_idx, :], 
                'o', color='#7D0C81', markersize=3)
        
        # axis formatting
        if idx == 1:  # Only add xlabel on second (bottom) plot
            ax.set_xlabel('running speed (cm/s)', fontsize=14)
        ax.set_ylabel('firing rate (Hz)', fontsize=14)
        max_rate_rounded = round(max_rate, 2)
        ax.set_yticks([0, max_rate_rounded])
        ax.set_yticklabels(['0', f'{max_rate_rounded:.2f}'])
        ax.set_xlim(-0.5, max(np.max(speed_arena), np.max(speed_wheel) + 0.5))
        ax.set_ylim(0, max_rate * 1.2)  # add 30% padding above max for error bars 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=12)
    
 


