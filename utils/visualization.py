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
from utils.session_analysis import load_session_data
import cv2
import math
from matplotlib import cm
from scipy import stats
import pandas as pd




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


# 2. ROI and Locomotion

def plot_roi(roi_x, roi_y, roi_radius,mouse_x,mouse_y,mask_arena, mask_wheel, save=False):
    circle = plt.Circle((roi_x, roi_y),roi_radius, color="#0F0F0FFF", fill=None, lw=2.5, alpha=0.8)
    fig, ax = plt.subplots(1,1, figsize=(4, 3), dpi=100)
    plt.scatter(mouse_x[mask_arena], mouse_y[mask_arena], c="#195A2C", s=0.8)
    plt.scatter(mouse_x[mask_wheel], mouse_y[mask_wheel], c="#7D0C81", s=0.8)
    ax.set_xlim( left=np.min(mouse_x)-5, right= np.max(mouse_x)+1)
    ax.set_ylim( bottom=np.min(mouse_y)-1, top= np.max(mouse_y)+5)#test best dimensions
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(circle)
    ax.set_xlabel('x position', fontsize=10)
    ax.set_ylabel('y position', fontsize=10)
    ax.set_aspect('equal')
    ax.yaxis.set_inverted(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("gray")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color("gray")
    ax.spines['left'].set_linewidth(1.5)
    plt.rcParams['font.sans-serif'] = ['Arial']

    if save:
        plt.savefig(rf'C:\Users\Experiment\Desktop\Leonard_Stuff\plots_for_pres\roi_plot.png', bbox_inches='tight', dpi=300)

def plot_rotary_wheel_alignment(rotary_position, mask_wheel, mask_arena, corner, w_start=0, w_end=20000):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))

    time = np.arange(len(mask_wheel[w_start:w_end])) / 10 / 60 

    # Plot rotary movement trace
    rotary_moving = np.abs(np.diff(rotary_position[w_start:w_end])) > 0
    rotary_moving = np.concatenate([[False], rotary_moving])
    rotary_data = rotary_moving.astype(float) * 0.2 + 3.4
    rotary_data[~rotary_moving] = np.nan
    ax.fill_between(time, 3.4, rotary_data, color='black', alpha=0.8, linewidth=0)
    
    # Plot wheel mask trace
    wheel_data = mask_wheel[w_start:w_end].astype(float) * 0.2 + 2.4
    wheel_data[~mask_wheel[w_start:w_end]] = np.nan
    ax.fill_between(time, 2.4, wheel_data, color="#7D0C81", alpha=0.8, linewidth=0)
    
    # Plot arena mask trace
    arena_data = mask_arena[w_start:w_end].astype(float) * 0.2 + 1.4
    arena_data[~mask_arena[w_start:w_end]] = np.nan
    ax.fill_between(time, 1.4, arena_data, color="#195A2C", alpha=0.8, linewidth=0)
    
    # Plot corner trace
    corner_data = corner[w_start:w_end].astype(float) * 0.2 + 0.4
    corner_data[~corner[w_start:w_end]] = np.nan
    ax.fill_between(time, 0.4, corner_data, color="#929292", alpha=0.8, linewidth=0)

    # Set limits and labels
    ax.set_ylim(0, 4)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel('time (min)')

    # Set y-tick positions and labels
    ax.set_yticks([3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels(['rotary\ndisplacement', 'wheel\noccupancy', 'arena\noccupancy','corner\nregion' ])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.rcParams['font.sans-serif'] = ['Arial']

def visualize_annotated_frame(subject_id, date, metadata, behavior,dlc_df, frame_idx, save=False):

    spinal_points = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3']
    
    # Video path
    video_path = fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}.mp4'
    
    # Get the frame
    print(f"Getting frame {frame_idx}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return None
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Draw all DLC bodyparts in cyan
    bodyparts = dlc_df.columns.get_level_values(0).unique()
    for bodypart in bodyparts:
        try:
            x = dlc_df[(bodypart, 'x')].iloc[frame_idx]
            y = dlc_df[(bodypart, 'y')].iloc[frame_idx]
            
            if  not (np.isnan(x) or np.isnan(y)):
                if bodypart in spinal_points:
                    cv2.circle(vis_frame, (int(x), int(y)), 4, (255, 255, 0), -1)
                else:
                    cv2.circle(vis_frame, (int(x), int(y)), 2, (192, 192, 192), -1)
            
        except:
            continue
    
    # Draw median position in red (if within behavioral data range)
    behavioral_idx = frame_idx - metadata.exp_onset
    if 0 <= behavioral_idx < len(behavior.mouse_x):
        median_x = behavior.mouse_x[behavioral_idx]
        median_y = behavior.mouse_y[behavioral_idx]
        if not (np.isnan(median_x) or np.isnan(median_y)):
            cv2.circle(vis_frame, (int(median_x), int(median_y)), 4, (0, 0, 255), -1)  # red filled
    
    # Draw wheel ROI in green
    if not (np.isnan(metadata.roi_x) or np.isnan(metadata.roi_y)):
        cv2.circle(vis_frame, (int(metadata.roi_x), int(metadata.roi_y)), 
                   int(metadata.roi_radius), (0, 0, 0), 3)  
    
    # Convert BGR to RGB for matplotlib
    vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    
    # Display
    plt.figure(figsize=(12, 9))
    plt.imshow(vis_frame_rgb)
    plt.axis('off')

    if save:
        plt.savefig(rf'C:\Users\Experiment\Desktop\Leonard_Stuff\plots_for_pres\annotated_frame.png', bbox_inches='tight', dpi=300)
    
    return vis_frame





        

def plot_correlation_histogram(correlations,  p_vals=None,color='blue',save_path=None, save=False):

        fig, ax = plt.subplots(figsize=(4, 8), dpi=150)

       
        sig_pos_mask = np.where((correlations >0) & (p_vals < 0.05))
        sig_neg_mask = np.where((correlations < 0) & (p_vals < 0.05))
        neutral_mask = np.where( p_vals > 0.05)
        ax.hist(correlations[sig_pos_mask], alpha=0.7, color=color, bins=40, orientation='horizontal')
        ax.hist(correlations[sig_neg_mask], alpha=0.7, color= color, bins=40, orientation= 'horizontal')
        ax.hist(correlations[neutral_mask], alpha=0.7, color="grey", bins=40, orientation='horizontal')

       

        ax.set_ylabel('Pearson correlation', fontsize=12)
        ax.set_xlabel('number of neurons', fontsize=12)
        ax.set_yticks(np.arange(-0.5, 0.505, 0.5))
        ax.set_xticks(np.arange(0,100.05, 50))
        ax.tick_params(axis='both', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(rf'C:\Users\Experiment\Desktop\Leonard_Stuff\plots_for_pres\{save_path}.png', bbox_inches='tight', dpi=300)

        fig.show()




##### ENCODING / CORRELATON #####


# 1. SINGLE SESSION
def plot_raster_pos_neg(spike_counts, speed, mask, correlations, color='blue', n_neurons=100, w_start=0, w_end=1200,save_path=None, save=False):
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), 
                                       gridspec_kw={'height_ratios': [0.5, 1, 1]},
                                       sharex=True, dpi=300)

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
        
   
    vmax_values = []
    for spikes in spike_sets:
        if len(spikes) > 0:
            
            vmax = np.percentile(spikes.flatten(), 77)
            vmax_values.append(vmax)
        else:
            vmax_values.append(1)  
        

    axes[0].plot(speed[w_start:w_end], color=color, linewidth=1.5)
    axes[0].set_ylabel('Arena\n(cm/s)', fontsize=12)
    axes[0].set_ylim(0, math.ceil(int(max(speed[w_start:w_end])) // 5 + 1)*5)
        
        
    # Clean velocity axes
    for ax in axes[:1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        
    
    labels = ['Arena (+)', 'Arena (-)']
    for i, (ax, spikes, label, vmax) in enumerate(zip(axes[1:], spike_sets, labels, vmax_values)):
        if len(spikes) > 0:
            # Use individual vmax for each subplot
            ax.matshow(spikes, aspect='auto', cmap='Grays', vmin=0, vmax=vmax, interpolation='none')
            if i == len(spike_sets) - 1:
                ax.set_ylabel(f'{len(spikes)} neurons', fontsize=12, loc="bottom")
            
            
            ax.text(-0.13, 0.5, label, transform=ax.transAxes, 
                    rotation=90, va='center', ha='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


        axes[-1].set_xlabel("2 minutes", fontsize=12, loc="left" ) #HARDCODED FOR NOW

        if save:
            plt.savefig(rf'C:\Users\Experiment\Desktop\Leonard_Stuff\plots_for_pres\{save_path}.png', bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    plt.show()

def plot_unsorted_raster(speed_arena, speed_wheel, mask_arena, mask_wheel, spike_counts, w_start=0, w_end=1200):

    fig, (ax1, ax2, ax3)  = plt.subplots(3,1, figsize=(8,6),gridspec_kw={'height_ratios': [0.5, 0.5, 1.5]},dpi=300)

    
    speed_arena[~mask_arena] = 0

    speed_wheel[~mask_wheel] = 0

    vmax = np.percentile(spike_counts[w_start:w_end], 75)


    ax1.plot(speed_arena[w_start:w_end], color=  "#195A2C")
    ax1.set_ylabel('Arena\n(cm/s)', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_ylim(0, math.ceil(int(max(speed_arena[w_start:w_end])) // 5 + 1)*5)


    ax2.plot(speed_wheel[w_start:w_end], color= "#7D0C81")
    ax2.set_ylabel('Wheel\n(cm/s)', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_ylim(0, math.ceil(int(max(speed_wheel[w_start:w_end])) // 5 + 1)*5)
    ax2.set_xticks([])

    ax3.matshow(spike_counts[w_start:w_end],  aspect='auto', cmap='Grays', interpolation='None', vmin=0, vmax= vmax)
    ax3.set_ylabel(f'{len(spike_counts)} neurons', fontsize=10, loc="bottom")     
    ax3.set_xlabel("2 minutes", fontsize=10, loc="left" )      
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_roi_time(time_arena, time_wheel):

        fig, ax = plt.subplots(1, 1, figsize=(3.5, 4))

        positions = [1, 2]
        data = [time_arena, time_wheel]
        colors = ['#2E7D32', '#6A1B9A']  

        # Create thin boxplots
        bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='black', linewidth=1.2),
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)

        # Add scatter points offset to the right
        offset = 0.15
        np.random.seed(42)  # For reproducible jitter
        for i, (data_points, color) in enumerate(zip(data, colors)):
            # Add jitter to x-position
            x = np.ones(len(data_points)) * (positions[i] + offset)
            x += np.random.normal(0, 0.02, len(data_points)) 
            
            ax.scatter(x, data_points, alpha=0.6, s=15, color=color, 
                    edgecolors='none', zorder=10)

        # Styling
        ax.set_xlim(0.5, 2.7)
        ax.set_ylim(0, 100)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Arena', 'Wheel'], fontsize=11)
        ax.set_ylabel('Time spent (%)', fontsize=11)


        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, labelsize=10)



        plt.tight_layout()


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
        
        reliability_values.append(result.correlations.reliability)
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
                zorder=3)

    ax.plot([0, 1], [0, 1], 'k-', linewidth=1.5, alpha=0.5, zorder=1)

    ax.set_xlim(-0.10, 1.05)
    ax.set_ylim(-0.10, 1.05)

    ax.set_xlabel('reliability within context', fontsize=16)
    ax.set_ylabel('stability between contexts', fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.legend().set_visible(False)


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
     

def plot_decoding_comparison(measured_speed, wc_pred_speed, cc_pred_speed, w_start, w_end, color="#000000", save_path=None, save=False):
                
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3), 
                                        gridspec_kw={'height_ratios': [1, 1, 1]},
                                        sharex=True)
        
    # Plot velocities
    ax1.plot(measured_speed[w_start:w_end], color=color, alpha=0.8)
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # Add horizontal text label for Prediction (positioned outside plot area)
    ax1.text(0, 0.5, 'observed\nwheel', transform=ax1.transAxes, fontsize=12, color='#7D0C81', horizontalalignment='right', verticalalignment='center')
    

    ax2.plot(wc_pred_speed[w_start:w_end], color=color, alpha=0.8)
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # Add horizontal text label for Arena Model (positioned outside plot area)
    ax2.text(0, 0.5, 'wheel\nmodel', transform=ax2.transAxes, fontsize=12, color='#7D0C81', horizontalalignment='right', verticalalignment='center')  # Signature green

    ax3.plot(cc_pred_speed[w_start:w_end], color=color, alpha=0.8)
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    # Add horizontal text label for Wheel Model (positioned outside plot area)
    ax3.text(0, 0.5, 'arena\nmodel', transform=ax3.transAxes, fontsize=12, color="#195A2C", horizontalalignment='right', verticalalignment='center')  # Signature purple
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()
    if save:
        plt.savefig(rf'C:\Users\Experiment\Desktop\Leonard_Stuff\plots_for_pres\{save_path}.png', bbox_inches='tight', dpi=300)

    plt.show()





def plot_context_preference(all_session_results):
    """Plot paired comparison of time spent in arena vs wheel."""
    
    # Extract data
    occupancy_arena = []
    occupancy_wheel = []
    
    for result in all_session_results:
        if result.spike_counts is None or result.behavior.summary is None:
            continue
        occupancy_arena.append(result.behavior.summary.occupancy.arena)
        occupancy_wheel.append(result.behavior.summary.occupancy.wheel)

    
    occupancy_arena = np.array(occupancy_arena)
    occupancy_wheel = np.array(occupancy_wheel)
    
    # Statistics
    mean_arena = np.mean(occupancy_arena)
    sem_arena = stats.sem(occupancy_arena)
    mean_wheel = np.mean(occupancy_wheel)
    sem_wheel = stats.sem(occupancy_wheel)
    stat, p_value = stats.wilcoxon(occupancy_arena, occupancy_wheel)
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # Plot individual sessions
    for i in range(len(occupancy_arena)):
        color = "#1F1F1F" 
        jitter = np.random.uniform(-0.05, 0.05, 2)
        # Plot the connecting line with lower alpha
        ax.plot([0 + jitter[0], 1 + jitter[1]], [occupancy_arena[i], occupancy_wheel[i]], 
            '--', color=color, alpha=0.2, linewidth=1)
    
        # Plot the dots with higher alpha
        ax.scatter([0 + jitter[0], 1 + jitter[1]], [occupancy_arena[i], occupancy_wheel[i]], 
               color=color, alpha=0.7, s=35, zorder=3)
    
   # Mean lines with arena/wheel colors
    ax.plot([-0.2, 0.2], [mean_arena, mean_arena], 
            color='#195A2C', linewidth=1.5, solid_capstyle='round', zorder=5, ls='dotted')
    ax.plot([0.8, 1.2], [mean_wheel, mean_wheel], 
            color='#7D0C81', linewidth=1.5, solid_capstyle='round', zorder=5, ls='dotted')

    
    # Formatting
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['arena', 'wheel'], fontsize=11)
    xcolors = ["#195A2C", "#7D0C81"]
    for xtick, xcolor in zip(ax.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax.set_ylabel('time spent (%)', fontsize=11)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)


    plt.rcParams['font.sans-serif'] = ['Arial']
    
    plt.tight_layout()
    #return fig

def plot_mean_speed_comparison(all_session_results):
    """Plot paired comparison of mean speed in arena vs wheel."""
    
    # Extract data
    speed_arena = []
    speed_wheel = []
    
    for result in all_session_results:
        if result.spike_counts is None or result.behavior.summary is None:
            continue
        speed_arena.append(result.behavior.summary.mean_speed.arena)
        speed_wheel.append(result.behavior.summary.mean_speed.wheel)
    
    speed_arena = np.array(speed_arena)
    speed_wheel = np.array(speed_wheel)
    
    # Statistics
    mean_arena = np.mean(speed_arena)
    sem_arena = stats.sem(speed_arena)
    mean_wheel = np.mean(speed_wheel)
    sem_wheel = stats.sem(speed_wheel)
    stat, p_value = stats.wilcoxon(speed_arena, speed_wheel)
    
   # Create figure
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # Plot individual sessions
    for i in range(len(speed_arena)):
        color = "#1F1F1F" 
        jitter = np.random.uniform(-0.05, 0.05, 2)
        # Plot the connecting line with lower alpha
        ax.plot([0 + jitter[0], 1 + jitter[1]], [speed_arena[i], speed_wheel[i]], 
            '--', color=color, alpha=0.2, linewidth=1)
    
        # Plot the dots with higher alpha
        ax.scatter([0 + jitter[0], 1 + jitter[1]], [speed_arena[i], speed_wheel[i]], 
               color=color, alpha=0.7, s=35, zorder=3)
    
   # Mean lines with arena/wheel colors
    ax.plot([-0.2, 0.2], [mean_arena, mean_arena], 
            color='#195A2C', linewidth=1.5, solid_capstyle='round', zorder=5, ls='dotted')
    ax.plot([0.8, 1.2], [mean_wheel, mean_wheel], 
            color='#7D0C81', linewidth=1.5, solid_capstyle='round', zorder=5, ls='dotted')

    
    # Formatting
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['arena', 'wheel'], fontsize=11)
    xcolors = ["#195A2C", "#7D0C81"]
    for xtick, xcolor in zip(ax.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax.set_ylabel('mean speed (cm/s)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.rcParams['font.sans-serif'] = ['Arial']
    
    plt.tight_layout()
    #return fig






            

