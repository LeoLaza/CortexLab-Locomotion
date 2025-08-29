"""
Behavioral analysis functions for locomotion and context detection.

This module handles:
- Velocity calculations from position data
- Context classification (arena vs wheel)
- Locomotion bout detection
- Temporal smoothing of behavioral states
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import cv2


    
def calculate_median_position(x,y):
    x = x.median(axis=1)
    y = y.median(axis=1)
    return  x, y


def calculate_oa_speed(x, y, bin_width):
    
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    #max_distance = np.percentile(distances, 99) 
    #distances[distances > max_distance] = np.nan
    #valid_indices = np.where(~np.isnan(distances))[0]
    #distances = np.interp(
    #np.arange(len(distances)),  
    #valid_indices,              
    #distances[valid_indices]    

    oa_speed_pix = distances / bin_width
    conversion_factor = 0.07 
    oa_speed = oa_speed_pix * conversion_factor
    oa_speed = gaussian_filter1d(oa_speed, 3)
    oa_speed = np.concatenate(([oa_speed[0] if len(oa_speed) > 0 else 0], oa_speed)) 
    return oa_speed


def calculate_wh_speed(rotary_position, bin_width, wh_diameter=15):
    wh_circumference = np.pi * wh_diameter
    linear_distance_cm = np.diff(rotary_position) * wh_circumference / 360 
    wh_speed = np.abs(linear_distance_cm / bin_width) 
    wh_speed = gaussian_filter1d(wh_speed, 3)
    wh_speed = np.concatenate(([wh_speed[0] if len(wh_speed) > 0 else 0], wh_speed))
    return wh_speed


def get_ROI(subject_id, date):
    frame_idx = 0

    cap = cv2.VideoCapture(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}.mp4')
    while frame_idx < 1000: 

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        preprocessed_frame = preprocess_frame(frame)
        circles = cv2.HoughCircles(preprocessed_frame, cv2.HOUGH_GRADIENT, dp=1.5, param1=50, param2= 20,minDist=90, minRadius=104, maxRadius=110)

        xy_list = []
        r_list = []
        if circles is not None:
            
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            xy_list.append((x, y))
            r_list.append(r)

        #if frame_idx % 10 == 0:
            #print(f"Processing frame {frame_idx}/10")

    cap.release()
    cv2.destroyAllWindows()

    if len(xy_list) == 0:
        print("No circles found in the video.")
        return None, None, None, None
    
    # Get ROI center and radius
    center_x = int(np.median([xy[0] for xy in xy_list]))
    center_y = int(np.median([xy[1] for xy in xy_list]))
    radius = int(np.median(r_list))

    return frame, center_x, center_y, radius
        
        
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = cv2.medianBlur(gray_frame, 21)
    return preprocessed_frame


def get_position_masks (x, y, center_x, center_y, radius, subject_id):
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    wh_pos = distances <= radius
    wh_pos = temporal_buffer(wh_pos)

    
    right_corner =(
        (x > center_x ) &  
        (y > center_y)
    )

    left_corner = (
        (x < center_x ) &  
        (y < center_y)
    )   

    if not subject_id == 'GB012':
        oa_pos = ~wh_pos & (~right_corner)
        corner = right_corner

    else:
        oa_pos = ~wh_pos & (~left_corner)
        corner = left_corner

    return oa_pos, wh_pos, corner


def temporal_buffer(pos_mask, buffer_size=10):
    transition_indices = np.where(np.diff(pos_mask.astype(int)) != 0)[0] + 1
    
    for i in range(len(transition_indices) - 1):
        if transition_indices[i+1] - transition_indices[i] < buffer_size:
            if transition_indices[i] > 0:
                pos_mask[transition_indices[i]:transition_indices[i+1]] = pos_mask[transition_indices[i]-1]
    
    return pos_mask

def get_classification_accuracy(mask_wheel, rotary_position, bin_width, movement_threshold=0):
    
    # Calculate wheel speed
    speed_wheel = calculate_wh_speed(rotary_position, bin_width)
    wheel_moving = speed_wheel > movement_threshold
    
    # Get indices where wheel is moving
    wheel_moving_indices = np.where(wheel_moving)[0]
    
    if len(wheel_moving_indices) == 0:
        return np.nan
    
    # Calculate accuracy: when wheel moves, how often is classification correct?
    accuracy = np.mean(mask_wheel[wheel_moving_indices])
    
    return accuracy

def get_locomotion_bouts(speed, context_mask, onset_threshold=2, offset_threshold=2, 
                             min_bout_duration=20, min_stable_offset=10):
   
    # Initialize output mask
    bout_mask = np.zeros_like(speed, dtype=bool)
    
    # Get indices where animal is in this context
    context_indices = np.where(context_mask)[0]
    
    if len(context_indices) == 0:
        return bout_mask, {'onsets': [], 'offsets': [], 'durations': []}
    
    # Only analyze speed values when in context
    # This is more explicit than masking
    onsets = []
    offsets = []
    
    # Work with continuous segments of context
    # Find where context starts and stops
    context_diff = np.diff(np.concatenate([[0], context_mask.astype(int), [0]]))
    segment_starts = np.where(context_diff == 1)[0]
    segment_ends = np.where(context_diff == -1)[0] - 1
    
    for start, end in zip(segment_starts, segment_ends):
        # Analyze this continuous segment
        segment_speed = speed[start:end+1]
        
        # Detect bouts within this segment
        running = False
        onset_idx = None
        
        for i in range(len(segment_speed)):
            if not running:
                if i + min_stable_offset <= len(segment_speed):
                    if np.all(segment_speed[i:i+min_stable_offset] >= onset_threshold):
                        running = True
                        onset_idx = i
            else:
                if i + min_stable_offset <= len(segment_speed) and np.all(segment_speed[i:i+min_stable_offset] < offset_threshold):
                    if np.all(segment_speed[i:i+min_stable_offset] < offset_threshold):
                        if i - onset_idx >= min_bout_duration:
                            # Convert to global indices
                            global_onset = start + onset_idx
                            global_offset = start + i
                            onsets.append(global_onset)
                            offsets.append(global_offset)
                            # Mark the bout
                            bout_mask[global_onset:global_offset+1] = True
                        running = False
                        onset_idx = None
                if i == len(segment_speed) - 1:
                    if running and i - onset_idx >= min_bout_duration:
                        global_onset = start + onset_idx
                        global_offset = start + i
                        onsets.append(global_onset)
                        offsets.append(global_offset)
                        bout_mask[global_onset:global_offset+1] = True
    
    durations = [off - on + 1 for on, off in zip(onsets, offsets)]
    
    bout_info = {
        'onsets': np.array(onsets),
        'offsets': np.array(offsets),
        'durations': np.array(durations)
    }
    
    return bout_mask, bout_info









