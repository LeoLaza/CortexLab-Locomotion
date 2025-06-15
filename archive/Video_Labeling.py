import numpy as np
from scipy.io import loadmat
import pandas as pd
import cv2
from Velocity_Calculations import calculate_median_position, calculate_distance,calculate_velocity, get_rotary_metadata, calculate_wheel_velocity, get_top_cam_timesamps
from collections import deque
import matplotlib.pyplot as plt
from Data_Loading_Functions import load_specific_experiment_data, get_DLC_data, get_subjectIDs_and_dates, get_experiment_path 


# DLC coordinates that will be displayed and their associated colors
BODYPART_COLOURS = {
    'nose': (255, 102, 178),
    'mouse_center': (214, 150, 92),
    'head_midpoint': (214, 137, 92),
    'neck': (214, 170, 92),
    'mid_back': (214, 203, 92),
    'mid_backend': (192, 214, 92),
    'mid_backend2': (159, 214, 92),
    'mid_backend3': (126, 214, 92),
    'tail_base': (93, 214, 92),
    'tail1': (92, 214, 123),
    'tail2': (92, 214, 156),
    'tail3': (92, 214, 189),
    'tail4': (92, 206, 214),
    'tail5': (92, 173, 214),
    'left_ear': (255, 164, 110),
    'right_ear': (255, 164, 110),
    'left_shoulder': (255, 204, 110),
    'right_shoulder': (255, 204, 110),
    'left_midside': (255, 244, 110),
    'right_midside': (255, 244, 110),
    'left_hip': (230, 255, 110),
    'right_hip': (230, 255, 110)
    }



def process_frame(frame, frame_idx, median_position_df, x_positions, y_positions): 
    """
    Annotates a frame of the original video with displays of bodypart coordinates
    and median position of spine coordiantes.

    frame: raw frame of video file 
    median_position_df: data frame containing median x and y position for each frame, 
                        as calculated based on x/y values of BODYPARTS specified in
                         Velocity_Calcs
    x_position: data frame of x coordinates for each frame for each bodypart in
                 BODYPART_COlOURS
    y_position: data frame of y coordinates for each frame for each bodypart in
                BODYPART_COLOURS
    Output: frame with specified annotations

    """

    frame = frame.copy()

    # putting colored circle at the x,y position of each bodypart 
    for part, color in BODYPART_COLOURS.items():
        x = int(x_positions[part] [frame_idx].item())
        y = int(y_positions[part][frame_idx].item())
        cv2.circle(frame, (x, y), 5, color, 1)
    

    # marking the border between wheel and open field (should probably specify ROI
    # somewhere to make code more interpretable)
    cv2.line(frame, (343, 291), (343, 512), (0, 0, 0), 3)
    cv2.line(frame, (343, 291), (640, 291), (0, 0, 0), 3)

    # putting colored circle at median x, y position
    cv2.circle(frame, (int(median_position_df.x[frame_idx]),int(median_position_df.y[frame_idx])), 5, (0, 0, 255), -1)

    # writing text specifying Median X and Median Y position
    cv2.putText(frame, 
                f"Avg. X: {median_position_df.x[frame_idx]:.2f}", 
                (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(frame, 
                f"Avg. Y: {median_position_df.y[frame_idx]:.2f}", 
                (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    return frame

def initialize_plot():
    """
    Creates plot that will be used to display velocity as sliding window.

    Output: empty plot with axis and line properties
    + two empty lists with max index of sliding window length

    """
    plot = plt.figure(figsize=(8,3))
    ax = plot.gca()
    line, = ax.plot([], [], 'b-')

    # creating lists to keep x/y value of list length
    x_data = deque(maxlen = 180) 
    y_data = deque(maxlen = 180)

    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)

    return plot, ax, x_data, y_data, line
    
def convert_plot_to_image(plot, width=640, height=200):
    """
    Converting matplotlib plot to fit OpenCV format.

    Output: converted frame

    """
    plot.canvas.draw()
    plot_image = np.frombuffer(plot.canvas.tostring_argb(), dtype=np.uint8)
    plot_image = plot_image.reshape(plot.canvas.get_width_height()[::-1] + (4,)) 
    
    plot_image = plot_image[:, :, 1:]  
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
    
    plot_image = cv2.resize(plot_image, (width, height))
    
    return plot_image


# specify experiments for whom you want to create an annotated video
DATES = ['2024-03-04','2024-03-05','2024-03-11', '2024-03-08', '2024-03-12','2024-03-13', '2024-03-14', '2024-03-15' '2024-03-19', '2024-03-20', '2024-03-21', '2024-03-22', '2024-03-25', '2024-03-26', '2024-05-09', '2024-05-10', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17', '2024-06-19', '2024-06-20','2024-06-21', '2024-06-25', '2024-07-11', '2024-07-12', '2024-07-16']
MICE = ['AV043', 'EB036', 'EB037', 'GB011', 'GB012']


for mouse in MICE:
    for date in DATES:

            try:
                # trying to load datafiles and timestamps
                data = load_specific_experiment_data(mouse, date)
                exp_idx = data.index[data.expDef.isin(['spontaneousActivity'])]
                exp_num, exp_folder = get_experiment_path(data)
                delay, top_cam_timestamps = get_top_cam_timesamps(mouse, date, exp_num, exp_folder)
                rotary_timestamps, rotary_position = get_rotary_metadata(exp_folder)
                dlc_df, scorer = get_DLC_data(mouse, date, delay)
                print(exp_folder)


                # loading x and y positions for every bodypart in BODYPART_COLOURS
                x_positions = {part: dlc_df.loc[:, (scorer, part, 'x')].values for part in BODYPART_COLOURS.keys()}
                y_positions = {part: dlc_df.loc[:, (scorer, part, 'y')].values for part in BODYPART_COLOURS.keys()}

                
                # loading every frame of the raw video 
                cap = cv2.VideoCapture(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{mouse}_{date}.mp4')
                
                # get parameters and plot
                median_position_df = calculate_median_position(dlc_df, scorer)
                distance = calculate_distance(median_position_df)
                velocity = calculate_velocity(distance)
                
                downsampled_timestamps, wheel_velocity = calculate_wheel_velocity(rotary_timestamps, rotary_position, dlc_df)
                
                plot, ax, x_data, y_data, line = initialize_plot()
                plot_wv, ax_wv, x_data_wv, y_data_wv, line_wv = initialize_plot()
                ax.set_xlabel('Frame')
                ax.set_ylabel('Velocity')
                ax.set_xlim (0, 180)
                ax.set_ylim(0, 20)

                ax_wv.set_xlabel('Frame')
                ax_wv.set_ylabel('Wheel Velocity')
                ax_wv.set_xlim (0, 180)
                ax_wv.set_ylim(-0.001, 0.001)

                
                # create video writer
                cap.set(cv2.CAP_PROP_POS_FRAMES, delay)
                plot_height = 400
                path = f"H:/Annotated_Videos/annotated_video_{mouse}_{date}.mp4"
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print(f'FPS: {fps}')
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                labeled_video = cv2.VideoWriter(path,
                                                cv2.VideoWriter_fourcc(*'mp4v'),
                                                60, 
                                                (frame_width, frame_height + plot_height))

                # reset frame_idx and specify number of frames in video
                frame_idx = 0
                total_frames = len(top_cam_timestamps)
                
                print(f"Total video frames: {total_frames}")
                print(f"Wheel velocity length: {len(wheel_velocity)}")
                print(f'dlc: {dlc_df.shape[0]}')
                # loop over each frame for all frames
                while frame_idx < total_frames:

                    ret, frame = cap.read() # get frame
                    if not ret:
                        break

                    # append data for sliding window plot
                    x_data.append(frame_idx)
                    y_data.append(velocity[frame_idx])
                    x_data_wv.append(frame_idx)
                    y_data_wv.append(wheel_velocity[frame_idx])
                    line.set_data(x_data, y_data)
                    line_wv.set_data(x_data_wv, y_data_wv)

                    # create sliding effect by updating x values displayed every 180 frames
                    if frame_idx > 180:  
                        ax.set_xlim(frame_idx - 180, frame_idx)
                        ax_wv.set_xlim(frame_idx - 180, frame_idx)

                            
                    # annotating frame
                    processed_frame = process_frame(frame, frame_idx, median_position_df, x_positions, y_positions)
                    
                    # joining plot and video frame 
                    plot_image = convert_plot_to_image(plot, width=640, height=200)
                    plot_image_wv = convert_plot_to_image(plot_wv, width=640, height=200)
                    concatenated_plots = cv2.vconcat([plot_image, plot_image_wv])
                    concatenated_frame = cv2.vconcat([concatenated_plots, processed_frame])


                    labeled_video.write(concatenated_frame)

                    if frame_idx % 100 == 0:
                        print(f'frame: {frame_idx} out of {total_frames}')

                    if frame_idx == 15000:
                        break
                    frame_idx += 1
                    
                labeled_video.release()
                print(f'I just released video {mouse}_{date} and loaded {frame_idx} out of {len(dlc_df)} frames')
                cap.release()

            except FileNotFoundError:
                print(f"File not found for mouse {mouse} on date {date}")