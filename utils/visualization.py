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
from utils.data_loading_and_preprocessing import preprocess_dlc_data
import cv2



##### BEHAVIOR ######


# 1. DLC Preprocessing (APPENDIX)

def plot_bodypart_likelihoods(dlc_df, quality_thresh = 0.9 , selected_bodyparts= ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3']):

    bodypart_pos= dlc_df.loc[:, (selected_bodyparts, slice(None))]
    likelihood = bodypart_pos.xs('likelihood', level='coords', axis=1)

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

def distribution_bodypart_pos_change(position_changes, selected_bodyparts, max_distance):
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

def pos_change_processing(raw_position_changes, processed_position_changes, raw_median_change, processed_median_change, selected_bodyparts =['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3'], w_start=0, w_end=7200):
    
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





def calculate_bodypart_position_change(dlc_df, selected_bodyparts = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3'], quality_thresh= None, preprocessing=False):
    
    if preprocessing:
        x,y = preprocess_dlc_data(dlc_df, quality_thresh=quality_thresh, selected_bodyparts=selected_bodyparts, max_distance=15)

    else:
        bodypart_pos= dlc_df.loc[:, (selected_bodyparts, slice(None))]
        x = bodypart_pos.xs('x', level='coords', axis=1)  
        y= bodypart_pos.xs('y', level='coords', axis=1)

    position_changes = {}
    for bodypart in selected_bodyparts:
        x_diff = x[bodypart].diff()
        y_diff = y[bodypart].diff()
        euclidean_dist = np.sqrt(x_diff**2 + y_diff**2)
        position_changes[bodypart] = euclidean_dist

    x_median = x.median(axis=1)
    y_median = y.median(axis=1)
    median_change = np.sqrt(x_median.diff()**2 + y_median.diff()**2)

    return position_changes, median_change



# 2. SINGLE SESSION BEHAVIOR


def plot_masked_positions(x, y, arena_mask, wheel_mask, color_arena='green', color_wheel='purple'):
    
    plt.figure(figsize=(12, 10))
    plt.scatter(x[arena_mask], y[arena_mask], c=color_arena, s=2)
    plt.scatter(x[wheel_mask], y[wheel_mask], c=color_wheel, s=2)
    plt.show

def show_ROI_on_frame(frame, center_x, center_y, radius, subject_id, date):
        # Draw the circle in the image
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            # Draw a rectangle at the center of the circle
            cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)

            #show results
            plt.figure(figsize=(10, 8))
            plt.title(f'{subject_id}-{date}: center: ({center_x}, {center_y}), radius: {radius}')
            plt.imshow(frame)



def plot_correlation_histogram(correlations, color='blue', p_vals=None):

    fig, ax = plt.subplots(figsize=(4, 2))

    if p_vals:
        sig_pos_mask = np.where((correlations >0) & (p_vals < 0.05))
        sig_neg_mask = np.where((correlations < 0) & (p_vals < 0.05))
        neutral_mask = np.where( p_vals > 0.05)
        ax.hist(correlations[sig_pos_mask], alpha=0.7, color=color, bins=30)
        ax.hist(correlations[sig_neg_mask], alpha=0.7, color= color, bins=30)
        ax.hist(correlations[neutral_mask], alpha=0.7, color="grey", bins=30)

    else:
        ax.hist(correlations, alpha=0.7, color=color, bins=30)

    ax.set_xlabel('Pearson correlation', fontsize=10)
    ax.set_ylabel('Number of neurons', fontsize=10)
    ax.set_xticks(np.arange(-0.5, 0.505, 0.5))
    ax.set_yticks(np.arange(0,100.05, 50))
    ax.tick_params(axis='both', labelsize=10)
    fig.show()




##### ENCODING / CORRELATON #####


# 1. SINGLE SESSION
def plot_raster_pos_neg(spike_counts, speed, mask, correlations, color='blue', n_neurons=100, w_start=0, w_end=1200):
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), 
                                       gridspec_kw={'height_ratios': [1, 1, 1]},
                                       sharex=True)

    # Mask velocities
    speed[~mask] = 0


    # Get arena positive neurons
    pos_idx = np.where(correlations > 0)[0]
    pos_sort = pos_idx[np.argsort(correlations[pos_idx])[::-1]][:n_neurons]
        
    # Get arena negative neurons
    neg_idx = np.where(correlations < 0)[0]
    neg_sort = neg_idx[np.argsort(correlations[neg_idx])][:n_neurons]
        

   
    spike_sets = [
            spike_counts[pos_sort, w_start:w_end],
            spike_counts[neg_sort, w_start:w_end]
        ]
        
   
    vmax = np.percentile(np.concatenate([s.flatten() for s in spike_sets]), 75)
        
   
    axes[0].plot(speed, color=color, linewidth=1.5)
    axes[0].set_ylabel('Arena\n(cm/s)', fontsize=10)
    axes[0].set_ylim(0, max(speed) * 1.1)
        
        
    # Clean velocity axes
    for ax in axes[:1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        
    
    labels = ['Arena (+)', 'Arena (-)']
    for i, (ax, spikes, label) in enumerate(zip(axes[1:], spike_sets, labels)):
        if len(spikes) > 0:
            ax.matshow(spikes, aspect='auto', cmap='Grays', vmin=0, vmax=vmax, interpolation='none')
            if i == len(spike_sets) - 1:
                ax.set_ylabel(f'{len(spikes)} neurons', fontsize=10, loc="bottom")
            
            
            ax.text(-0.13, 0.5, label, transform=ax.transAxes, 
                    rotation=90, va='center', ha='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


        axes[-1].set_xlabel("2 minutes", fontsize=10, loc="left" ) #HARDCODED FOR NOW
    
    plt.tight_layout()
    plt.show()


def plot_reliability(corr_half1, corr_half2, color='blue'):
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(corr_half1, corr_half2, alpha=0.7, color=color, zorder=2)
    ax.set_xlabel('1st half Pearson Correlation', fontsize=10)
    ax.set_ylabel('2nd half Pearson Correlation', fontsize=10)
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1.5, alpha=0.5, zorder=1)
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

def plot_cross_context_correlation(corr_arena, corr_wheel):

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(corr_arena, corr_wheel, alpha=0.5, color="#221B69", zorder=2)
    ax.set_xlabel('Pearson Correlation Arena', fontsize=10)
    ax.set_ylabel('Pearson Correlation Wheel', fontsize=10)
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1.5, alpha=0.5, zorder=1)
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

def plot_stability_occupation(stability, occupation, color='blue'):
    plt.figure(figsize=(3, 3))
    plt.scatter(stability, occupation, alpha=0.5, color= color, zorder=2)
    plt.xlabel("arena stability", fontsize=10)
    plt.ylabel("proportion of time spent in arena", fontsize=10)
    plt.plot([0, 1], [0, 1], 'k-', linewidth=1.5, alpha=0.5, zorder=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.01, 0.5), fontsize=10)  
    plt.yticks(np.arange(0, 1.01, 0.5), fontsize=10)
    plt.show()

# 2. MULTISESSION
        
def plot_reliability_stability(all_session_results):
    

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)

    reliability_values = []
    stability_values = []
    brain_regions = []

    for result in all_session_results:
        if result.spike_counts is None:
             continue
        
        reliability_values.append(result.correlation.reliability)
        stability_values.append(result.correlations.stability)
        
    
        if result.metadata.subject_id[:2] == 'EB':
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



### DECODING ###

def plot_weight_correlation(weights_arena, weights_wheel, color="#009980"):

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(weights_arena, weights_wheel, alpha=0.5, color=color, zorder=2)
    ax.set_xlabel('weights Arena', fontsize=10)
    ax.set_ylabel('weights Wheel', fontsize=10)
    ax.plot([-0.5, 0.7], [-0.5, 0.7], 'k-', linewidth=1.5, alpha=0.5, zorder=1)
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.5))  
    ax.set_yticks(np.arange(-0.5, 0.51, 0.5))
    ax.tick_params(axis='both', labelsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
     
def plot_decoding_comparison(measured_speed, wc_pred_speed, cc_pred_speed, performance_within, performance_between, w_start, w_end, color="#009980"):
                
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), 
                                                gridspec_kw={'height_ratios': [1, 1, 1]},
                                                sharex=True)
                
            # Plot velocities
            ax1.plot(measured_speed[w_start:w_end], color=color, alpha=0.8)
            ax1.set_xticklabels([])
            ax1.set_xticks([])
            #ax1.set_yticks([])
            #ax1.spines['top'].set_visible(False)
            #ax1.spines['right'].set_visible(False)
            #ax1.spines['bottom'].set_visible(False)
            #ax1.spines['left'].set_visible(False)
            ax1.set_ylabel('Measured', color='black', fontsize=18, weight="bold")
            ax1.set_title(f'within:{performance_within:.2f}-between:{performance_between:.2f}', fontsize=18)

            ax2.plot(wc_pred_speed[w_start:w_end], color=color, alpha=0.8)
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            #ax2.set_yticks([])
            #ax2.spines['top'].set_visible(False)
            #ax2.spines['right'].set_visible(False)
            #ax2.spines['bottom'].set_visible(False)
            #ax2.spines['left'].set_visible(False)
            ax2.set_ylabel('Trained Same Context', color='black', fontsize=18, weight='bold')

            ax3.plot(cc_pred_speed[w_start:w_end], color=color, alpha=0.8)
            ax3.set_xticklabels([])
            ax3.set_xticks([])
            #ax3.set_yticks([])
            #ax3.spines['top'].set_visible(False)
            #ax3.spines['right'].set_visible(False)
            #ax3.spines['left'].set_visible(False)
            ax3.set_ylabel('Trained Different Context', color='black', fontsize=18, weight='bold')

            plt.tight_layout()
            plt.show()



            

