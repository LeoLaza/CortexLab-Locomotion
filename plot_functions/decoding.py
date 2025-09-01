from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_decoding_predictions(measured_speed_arena, arena_pred_arena, wheel_pred_arena, 
                             measured_speed_wheel, wheel_pred_wheel, arena_pred_wheel, 
                             w_start, w_end):
    
    # calculate correlations
    corr_arena_arena = np.corrcoef(measured_speed_arena[w_start:w_end], 
                                arena_pred_arena[w_start:w_end])[0,1]
    corr_wheel_arena = np.corrcoef(measured_speed_arena[w_start:w_end], 
                                wheel_pred_arena[w_start:w_end])[0,1]
    corr_wheel_wheel = np.corrcoef(measured_speed_wheel[w_start:w_end], 
                                wheel_pred_wheel[w_start:w_end])[0,1]
    corr_arena_wheel = np.corrcoef(measured_speed_wheel[w_start:w_end], 
                                arena_pred_wheel[w_start:w_end])[0,1]
                

    fig, axes = plt.subplots(3, 2, figsize=(8, 3))
        
    # ARENA CONTEXT (left column)
    # observed arena
    axes[0, 0].plot(measured_speed_arena[w_start:w_end], color="#195A2C", alpha=0.8) 
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_xticks([])
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['bottom'].set_visible(False)
    axes[0, 0].text(-0.13, 0.5, 'observed', transform=axes[0, 0].transAxes, fontsize=12,  
             color="#141414", horizontalalignment='right', verticalalignment='center')
    
    # Arena model predicting arena
    axes[1, 0].plot(arena_pred_arena[w_start:w_end], color="#195A2C", alpha=0.8)
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_xticks([])
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].spines['bottom'].set_visible(False)
    axes[1, 0].text(-0.13, 0.5, 'arena\nmodel', transform=axes[1, 0].transAxes, fontsize=12,  
             color="#141414", horizontalalignment='right', verticalalignment='center')
    axes[1, 0].text(0.98, 1.20, f'r={corr_arena_arena:.2f}', transform=axes[1, 0].transAxes, 
             fontsize=10, horizontalalignment='right', verticalalignment='top')
  
    # Wheel model predicting arena
    axes[2, 0].plot(wheel_pred_arena[w_start:w_end], color="#195A2C", alpha=0.8)
    axes[2, 0].set_xticklabels([])
    axes[2, 0].set_xticks([])
    axes[2, 0].spines['top'].set_visible(False)
    axes[2, 0].spines['right'].set_visible(False)
    axes[2, 0].text(-0.13, 0.5, 'wheel\nmodel', transform=axes[2, 0].transAxes, fontsize=12,  
             color="#141414", horizontalalignment='right', verticalalignment='center')
    axes[2, 0].text(0.98, 1.20, f'r={corr_wheel_arena:.2f}', transform=axes[2, 0].transAxes, 
             fontsize=10, horizontalalignment='right', verticalalignment='top')

    # WHEEL CONTEXT (right column)
    # observed wheel
    axes[0, 1].plot(measured_speed_wheel[w_start:w_end], color="#7D0C81", alpha=0.8)
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_xticks([])
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].spines['bottom'].set_visible(False)
    
    # arena model predicting wheel
    axes[1, 1].plot(arena_pred_wheel[w_start:w_end], color="#7D0C81", alpha=0.8)  
    axes[1, 1].set_xticklabels([])
    axes[1, 1].set_xticks([])
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].spines['bottom'].set_visible(False)
    axes[1, 1].text(0.98, 1.20, f'r={corr_arena_wheel:.2f}', transform=axes[1, 1].transAxes, 
             fontsize=10, horizontalalignment='right', verticalalignment='top')

    # wheel model predicting wheel  
    axes[2, 1].plot(wheel_pred_wheel[w_start:w_end], color="#7D0C81", alpha=0.8) 
    axes[2, 1].set_xticklabels([])
    axes[2, 1].set_xticks([])
    axes[2, 1].spines['top'].set_visible(False)
    axes[2, 1].spines['right'].set_visible(False)
    axes[2, 1].text(0.98, 1.20, f'r={corr_wheel_wheel:.2f}', transform= axes[2, 1].transAxes, 
             fontsize=10, horizontalalignment='right', verticalalignment='top')   
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()
   

    
def plot_decoding_performance_comparison(all_session_results):
    """
    Plot comparison of decoding performance within and across contexts
    for all sessions for both arena and wheel.

    Parameters:
    results : list
        comprises results objects for each session containing decoding performance data

    """

    # define colors for different brain regions and subjects
    colors = {
        'hippocampus': "#A5CB5D",  
        'striatum': "#E37A2A",
        'MOs(1)': "#660D0D",  
        'MOs(2)': "#9F3A3A",  
        'MOs(3)': "#D26E6E"  
    }

    # mapping of subject IDs in secondary motor cortex to mouse labels
    mos_mapping = {
        'AV043': 'MOs(1)',
        'GB011': 'MOs(2)', 
        'GB012': 'MOs(3)'
    }


    # create side by side figure with two subplots (left: arena, right: wheel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # gather arena decoding performance for each session distinguished by brain region
    corr_arena_list = []
    corr_wheel_to_arena_list = []
    brain_regions_arena = []
    subject_ids_arena = [] 

    for session in all_session_results:
        if session.decoding is None:
            continue
        
        test_data = session.decoding.test_data
        predictions = session.decoding.prediction
        
        corr_arena = np.corrcoef(test_data.speed_arena, predictions.arena_to_arena)[0, 1]
        corr_wheel_to_arena = np.corrcoef(test_data.speed_arena, predictions.wheel_to_arena)[0, 1]
        
        corr_arena_list.append(corr_arena)
        corr_wheel_to_arena_list.append(corr_wheel_to_arena)
        subject_ids_arena.append(session.metadata.subject_id) 
        
        # assign brain region based on subject ID
        if session.metadata.subject_id == 'EB036':
            brain_regions_arena.append('hippocampus')
        elif session.metadata.subject_id == 'EB037':
            brain_regions_arena.append('striatum')
        else:
            brain_regions_arena.append('secondary motor cortex')

    # convert correlation lists to numpy arrays
    corr_arena = np.array(corr_arena_list)
    corr_wheel_to_arena = np.array(corr_wheel_to_arena_list)

    # plot individual sessions
    for i in range(len(corr_arena)):
        region = brain_regions_arena[i]
        subject = subject_ids_arena[i]
        
        if region == 'hippocampus':
            color = colors['hippocampus']
        elif region == 'striatum':
            color = colors['striatum']
        else: 
            mos_label = mos_mapping.get(subject, 'MOs(1)') 
            color = colors[mos_label]

        jitter = np.random.uniform(-0.05, 0.05, 2)
        
        ax1.plot([0 + jitter[0], 1 + jitter[1]], [corr_arena[i], corr_wheel_to_arena[i]], 
                '--', color=color, alpha=0.3, linewidth=1) 
        
        ax1.scatter([0 + jitter[0], 1 + jitter[1]], [corr_arena[i], corr_wheel_to_arena[i]], 
                color=color, alpha=0.7, s=70, zorder=3, edgecolors='none', linewidth=0.5) 

    # calculate and plot means
    mean_arena = np.mean(corr_arena)
    mean_wheel_to_arena = np.mean(corr_wheel_to_arena)
    ax1.plot([-0.2, 0.2], [mean_arena, mean_arena], color='#195A2C', linewidth=3, zorder=5, ls='dotted')
    ax1.plot([0.8, 1.2], [mean_wheel_to_arena, mean_wheel_to_arena], color='#7D0C81', linewidth=3, zorder=5, ls='dotted')

    # formatting for ax1
    ax1.axhline(y=0, color="black", linestyle='dotted', linewidth=1, alpha=0.5)
    ax1.set_xlim(-0.3, 1.5)
    ax1.set_ylim(-0.51, 1)
    ax1.set_yticks(np.arange(-0.5, 1.1, 0.5))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['trained\narena', 'trained\nwheel'], fontsize=18)
    xcolors = ["#195A2C", "#7D0C81"]
    for xtick, xcolor in zip(ax1.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax1.set_ylabel('correlation (pred. vs obsv.)', fontsize=18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='y', labelsize=16)


    # gather wheel decoding performance for each session distinguished by brain region
    corr_wheel_list = []
    corr_arena_to_wheel_list = []
    brain_regions_wheel = []
    subject_ids_wheel = []

    for session in all_session_results:
        if session.decoding is None:
            continue
        
        test_data = session.decoding.test_data
        predictions = session.decoding.prediction
        
        corr_wheel = np.corrcoef(test_data.speed_wheel, predictions.wheel_to_wheel)[0, 1]
        corr_arena_to_wheel = np.corrcoef(test_data.speed_wheel, predictions.arena_to_wheel)[0, 1]
        
        corr_wheel_list.append(corr_wheel)
        corr_arena_to_wheel_list.append(corr_arena_to_wheel)
        subject_ids_wheel.append(session.metadata.subject_id) 
        
        if session.metadata.subject_id == 'EB036':
            brain_regions_wheel.append('hippocampus')
        elif session.metadata.subject_id == 'EB037':
            brain_regions_wheel.append('striatum')
        else:
            brain_regions_wheel.append('secondary motor cortex')

    corr_wheel = np.array(corr_wheel_list)
    corr_arena_to_wheel = np.array(corr_arena_to_wheel_list)

    # pot individual sessions for ax2
    for i in range(len(corr_wheel)):
        region = brain_regions_wheel[i]
        subject = subject_ids_wheel[i]
        
        if region == 'hippocampus':
            color = colors['hippocampus']
        elif region == 'striatum':
            color = colors['striatum']
        else: # It's secondary motor cortex
            mos_label = mos_mapping.get(subject, 'MOs(1)')
            color = colors[mos_label]

        jitter = np.random.uniform(-0.05, 0.05, 2)
        
        ax2.plot([1 + jitter[1], 0 + jitter[0]], [corr_wheel[i], corr_arena_to_wheel[i]], 
                '--', color=color, alpha=0.3, linewidth=1)
        
        ax2.scatter([1 + jitter[1], 0 + jitter[0]], [corr_wheel[i], corr_arena_to_wheel[i]], 
                color=color, alpha=0.7, s=70, zorder=3, edgecolors='none', linewidth=0.5)

    # calculate and plot means
    mean_wheel = np.mean(corr_wheel)
    mean_arena_to_wheel = np.mean(corr_arena_to_wheel)
    ax2.plot([0.8, 1.2], [mean_wheel, mean_wheel], color='#7D0C81', linewidth=3, zorder=5, ls='dotted')
    ax2.plot([-0.2, 0.2], [mean_arena_to_wheel, mean_arena_to_wheel], color='#195A2C', linewidth=3, zorder=5, ls='dotted')

    # formatting for ax2
    ax2.axhline(y=0, color="black", linestyle='dotted', linewidth=1, alpha=0.5)
    ax2.set_xlim(-0.3, 1.5)
    ax2.set_ylim(-0.51, 1)
    ax2.set_yticks([])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['trained\narena', 'trained\nwheel'], fontsize=18)
    for xtick, xcolor in zip(ax2.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='y', labelsize=14)


    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 


def plot_weight_correlation(weights_arena, weights_wheel):

    # plot correlation of weights 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(weights_arena, weights_wheel, alpha=0.7, color="#141414", zorder=2, s=80)

    # axis formatting 
    ax.set_xlabel('weights arena', fontsize=20, color="#195A2C")
    ax.set_ylabel('weights wheel', fontsize=20, color="#7D0C81")
    ax.axhline(0, color='k', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.axvline(0, color='k', linewidth=1, alpha=0.5, zorder=1, ls='--')
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xticks(np.arange(-0.15, 0.151, 0.15))  
    ax.set_yticks(np.arange(-0.15, 0.151, 0.15))
    ax.tick_params(axis='x', pad=10) 
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='both', labelsize=20)

    # remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

   