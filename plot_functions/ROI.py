
from matplotlib import pyplot as plt
import numpy as np
import cv2


def plot_roi(roi_x, roi_y, roi_radius,mouse_x,mouse_y,mask_arena, mask_wheel):

    """
    Plot x,y position of mouse for each frame.
    Color indicates mask based classification. Wheel ROI overlaid as circle.

    Parameters:

    roi_x : numpy array
        x coordinate of wheel ROI center
    roi_y : numpy array
        y coordinate of wheel ROI center
    roi_radius : int
        radius of wheel ROI
    mouse_x : numpy array
        x coordinate of mouse position for each frame
    mouse_y : numpy array
        y coordinate of mouse position for each frame
    mask_arena : numpy array
        boolean mask for all frames (True if classified in arena)
    mask_wheel : numpy array
        boolean mask for all frames (True if classified in wheel)
    """

    # draw wheel location
    circle = plt.Circle((roi_x, roi_y),roi_radius, color="#0F0F0FFF", fill=None, lw=2.5, alpha=0.8)

    # plot mouse x,y positions colored by mask classification
    fig, ax = plt.subplots(1,1, figsize=(4, 3), dpi=100)
    plt.scatter(mouse_x[mask_arena], mouse_y[mask_arena], c="#195A2C", s=0.8)
    plt.scatter(mouse_x[mask_wheel], mouse_y[mask_wheel], c="#7D0C81", s=0.8)

    # add wheel location to figure 
    ax.add_patch(circle)

    # axis formatting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('x position', fontsize=10)
    ax.set_ylabel('y position', fontsize=10)
    ax.set_aspect('equal')
    ax.yaxis.set_inverted(True)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("gray")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color("gray")
    ax.spines['left'].set_linewidth(1.5)

    # optional adjust dimension for prefered visualization
    ax.set_xlim( left=np.min(mouse_x)-5, right= np.max(mouse_x)+1)
    ax.set_ylim( bottom=np.min(mouse_y)-1, top= np.max(mouse_y)+5)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()


def plot_rotary_wheel_alignment(rotary_position, mask_wheel, mask_arena, corner, w_start, w_end):

    """
    Plot periods when rotary encoder displaced to compare with mask based classification.

    Parameters:

    rotary_position : numpy array
        rotary encoder position for each frame
    mask_wheel : numpy array
        boolean mask for all frames (True if classified in wheel)
    mask_arena : numpy array
        boolean mask for all frames (True if classified in arena)
    corner : numpy array
        boolean mask for all frames (True if in corner region of arena)
    w_start : int
        start index for window to plot
    w_end : int
        end index for window to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))

    time = np.arange(len(mask_wheel[w_start:w_end])) / 10 / 60 

    # identify periods of rotary encoder displacement
    rotary_moving = np.abs(np.diff(rotary_position[w_start:w_end])) > 0
    rotary_moving = np.concatenate([[False], rotary_moving])

    # plot rotary encoder trace
    rotary_data = rotary_moving.astype(float) * 0.2 + 3.4
    rotary_data[~rotary_moving] = np.nan
    ax.fill_between(time, 3.4, rotary_data, color='black', alpha=0.8, linewidth=0)
    
    # plot wheel mask trace
    wheel_data = mask_wheel[w_start:w_end].astype(float) * 0.2 + 2.4
    wheel_data[~mask_wheel[w_start:w_end]] = np.nan
    ax.fill_between(time, 2.4, wheel_data, color="#7D0C81", alpha=0.8, linewidth=0)
    
    # plot arena mask trace
    arena_data = mask_arena[w_start:w_end].astype(float) * 0.2 + 1.4
    arena_data[~mask_arena[w_start:w_end]] = np.nan
    ax.fill_between(time, 1.4, arena_data, color="#195A2C", alpha=0.8, linewidth=0)
    
    # plot corner trace
    corner_data = corner[w_start:w_end].astype(float) * 0.2 + 0.4
    corner_data[~corner[w_start:w_end]] = np.nan
    ax.fill_between(time, 0.4, corner_data, color="#929292", alpha=0.8, linewidth=0)

    # axis formatting
    ax.set_ylim(0, 4)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel('time (min)')
    ax.set_yticks([3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels(['rotary\ndisplacement', 'wheel\noccupancy', 'arena\noccupancy','corner\nregion' ])

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()



def plot_annotated_frame(metadata, behavior, selected_bodyparts, frame_idx):

    """
    Plot single video frame annotated with bodypart positions based on DLC model.
    Bodyparts included in analysis are highlighted in color.
    Median position based on behavioral data is shown in red. 
    Remaining bodyparts are shown in gray.
    Wheel ROI is overlaid as cirlce.
    
    Parameters:
    
    metadata: Bunch 
        experiment metatadata containing wheel ROI information
    behavior: Bunch
        behavioral data containing median x,y position of mouse
    dlc_df: pandas DataFrame
        DLC output dataframe with x,y coordinates for each bodypart and frame
    selected_bodyparts: list
        list of bodyparts included in analysis 
    frame_idx: int
        index of frame to plot
    video_path: str
        path to video file
    """

    cap = cv2.VideoCapture(metadata.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return None
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Draw all DLC bodyparts in cyan
    bodyparts = behavior.dlc_df.columns.get_level_values(0).unique()
    for bodypart in bodyparts:
        try:
            x = behavior.dlc_df[(bodypart, 'x')].iloc[frame_idx]
            y = behavior.dlc_df[(bodypart, 'y')].iloc[frame_idx]
            
            if  not (np.isnan(x) or np.isnan(y)):
                if bodypart in selected_bodyparts:
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
            cv2.circle(vis_frame, (int(median_x), int(median_y)), 4, (0, 0, 255), -1)  
    
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

    

