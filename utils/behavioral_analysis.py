import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import cv2


    
def calculate_median_position(x,y):
    x = x.median(axis=1)
    y = y.median(axis=1)
    return  x, y


def calculate_oa_speed(x, y, bin_width):
    """
    Calculate open arena speed from eucledian distance between frames.
    
    Parameters:
    x, y : arrays - position coordinates (pixels)
    bin_width : float - time per bin (seconds)
    
    Returns:
    array - speed in cm/s (smoothed with Gaussian filter)
    """
    # calculate eucledian distance
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
     
    # calculate speed in pixels
    oa_speed_pix = distances / bin_width

    # convert to cm
    conversion_factor = 0.07 
    oa_speed = oa_speed_pix * conversion_factor

    # smooth
    oa_speed = gaussian_filter1d(oa_speed, 3)

    # first frame set speed to 0
    oa_speed = np.concatenate(([oa_speed[0] if len(oa_speed) > 0 else 0], oa_speed))

    return oa_speed


def calculate_wh_speed(rotary_position, bin_width, wh_diameter=15):
    """
    Convert rotary position to linear running speed.
    
    Parameters:
    rotary_position : array - rotary encoder angular position (degrees)
    bin_width : float - time per bin (seconds)
    wh_diameter : float - wheel diameter (cm)
    
    Returns:
    array - linear speed in cm/s (smoothed)
    """
    # convert angular position to linear position
    wh_circumference = np.pi * wh_diameter
    linear_distance_cm = np.diff(rotary_position) * wh_circumference / 360 

    # calculate wheel speed
    wh_speed = np.abs(linear_distance_cm / bin_width) 

    # smooth
    wh_speed = gaussian_filter1d(wh_speed, 3)

    # first frame set to 0
    wh_speed = np.concatenate(([wh_speed[0] if len(wh_speed) > 0 else 0], wh_speed))

    return wh_speed


def get_ROI(subject_id, date):
    """
    Detect running wheel position using Hough circle detection.
    Samples first 1000 frames to find stable wheel location.
    (Simplify frame selection and make function more flexible).
    
    Returns:
    frame : last processed frame
    center_x, center_y : int - wheel center coordinates (pixels)
    radius : int - wheel radius in pixels
    """

    frame_idx = 0

    # capture all frames of video
    cap = cv2.VideoCapture(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}.mp4')

    # read first 1000 frames
    while frame_idx < 1000: 

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # apply Hough Circle Transform
        preprocessed_frame = preprocess_frame(frame)
        ### PARAMETERS HARDCODED!!!###
        circles = cv2.HoughCircles(preprocessed_frame, cv2.HOUGH_GRADIENT, dp=1.5, param1=50, param2= 20,minDist=90, minRadius=104, maxRadius=110)

        # extract center x,y coordinates and radius
        xy_list = []
        r_list = []
        if circles is not None:
            
            # convert the center coordinates and radius of the circles to integers
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
    
    # get ROI center and radius
    center_x = int(np.median([xy[0] for xy in xy_list]))
    center_y = int(np.median([xy[1] for xy in xy_list]))
    radius = int(np.median(r_list))

    return frame, center_x, center_y, radius
        
        
def preprocess_frame(frame):
    """
    Prepare video frame for circle detection.
    Converts to grayscale and applies median blur.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = cv2.medianBlur(gray_frame, 21)
    return preprocessed_frame


def get_position_masks (x, y, center_x, center_y, radius, subject_id):
    """
    Classify mouse position as arena, wheel, or corner.

    x, y : float  - mouse position coordinates (pixels)
    center_x, center_y : int - wheel center coordinates (pixels)
    radius : int - wheel radius in pixels
    
    Returns:
    oa_pos : bool array - arena position mask
    wh_pos : bool array - wheel position mask  
    corner : bool array - corner position mask (excluded)
    """

    # calculate eucledian distance of mouse to wheel center 
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # classify distances smaller as radius as wheel position
    wh_pos = distances <= radius

    # reclassify brief transitions
    wh_pos = temporal_buffer(wh_pos)

    # exclude corner regions (assumes wheel in bottom right corner or top left corner)
    right_corner =(
        (x > center_x ) &  
        (y > center_y)
    )

    left_corner = (
        (x < center_x ) &  
        (y < center_y)
    )   

    # exclude corner depending on where wheel was in experiment
    if not subject_id == 'GB012': # currently assignment hardcoded
        oa_pos = ~wh_pos & (~right_corner)
        corner = right_corner

    else:
        oa_pos = ~wh_pos & (~left_corner)
        corner = left_corner

    return oa_pos, wh_pos, corner


def temporal_buffer(pos_mask, buffer_size=20):
    """
    Remove brief position transitions in position based mask.
    
    Parameters:
    pos_mask : bool array - position classification
    buffer_size : int - minimum bins for valid transition
    
    Returns:
    bool array - smoothed position mask
    """
    transition_indices = np.where(np.diff(pos_mask.astype(int)) != 0)[0] + 1
    
    for i in range(len(transition_indices) - 1):
        if transition_indices[i+1] - transition_indices[i] < buffer_size:
            if transition_indices[i] > 0:
                pos_mask[transition_indices[i]:transition_indices[i+1]] = pos_mask[transition_indices[i]-1]
    
    return pos_mask



def get_locomotion_bouts(speed, context_mask, onset_threshold=2, offset_threshold=2, 
                             min_bout_duration=20, min_stable_offset=10):
    """
    Detect running bouts.
    
    Parameters:
    speed : array - instantaneous speed (cm/s)
    context_mask : bool array - restrict detection to this context
    onset_threshold : float - speed to start bout (cm/s)
    offset_threshold : float - speed to end bout (cm/s)
    min_bout_duration : int - minimum bins for valid bout
    min_stable_offset : int - bins below threshold to end bout
    
    Returns:
    bout_mask : bool array - True during locomotion
    bout_info : dict with onset/offset indices and durations
    """
   
    # initialize output mask
    bout_mask = np.zeros_like(speed, dtype=bool)
    
    # get indices where animal is in this context
    context_indices = np.where(context_mask)[0]
    
    if len(context_indices) == 0:
        return bout_mask, {'onsets': [], 'offsets': [], 'durations': []}
    

    onsets = []
    offsets = []
    
    # find where context starts and stops
    context_diff = np.diff(np.concatenate([[0], context_mask.astype(int), [0]]))
    segment_starts = np.where(context_diff == 1)[0]
    segment_ends = np.where(context_diff == -1)[0] - 1
    
    for start, end in zip(segment_starts, segment_ends):
        # analyze this continuous segment
        segment_speed = speed[start:end+1]
        
        # detect bouts within this segment
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
                            # convert to global indices
                            global_onset = start + onset_idx
                            global_offset = start + i
                            onsets.append(global_onset)
                            offsets.append(global_offset)
                            # mark the bout
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









