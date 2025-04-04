import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Loading_Functions import load_specific_experiment_data, get_DLC_data, get_subjectIDs_and_dates, get_experiment_path
from pathlib import Path
import os
import glob
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from plotly_resampler import FigureResampler
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import binned_statistic
import math

def calculate_median_position(dlc_df, scorer):

    BODYPARTS = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3']

    # Group by bodyparts and filter for only the ones we want to use in our calculation
    selected_data = dlc_df.loc[:, (scorer, BODYPARTS, slice(None))]
    

    # Select Values of High Quality
    likelihood_values = selected_data.xs('likelihood', level='coords', axis=1)
    likelihood_values[likelihood_values <= 0.95] = np.nan
    likelihood_values = likelihood_values.interpolate(method='linear')
    strong_xs = selected_data.xs('x', level='coords', axis=1)
    strong_ys = selected_data.xs('y', level='coords', axis=1)
    x_medians = strong_xs.median(axis=1)
    y_medians = strong_ys.median(axis=1)
    median_position = {idx: {'x': x, 'y': y} for idx, (x, y) in enumerate(zip(x_medians, y_medians))}
    median_position_df = pd.DataFrame.from_dict(median_position, orient='index')
    return median_position_df

UNREALISTIC_DISTANCE = 50
def calculate_distance(median_position_df):
    distances = median_position_df.diff()**2
    distances = np.sqrt(distances.sum(axis=1))
    unrealistic_mask = distances > UNREALISTIC_DISTANCE
    
    
    distances[unrealistic_mask] = np.nan
    distances = distances.interpolate(method='linear')
    return distances

FPS = 60
def calculate_velocity(distances):
    #velocity = distances.div(1/FPS)
    velocity = distances
    return velocity

WINDOW = 30
def calculate_average_velocity(velocity):
    # Use integer division to group frames into chunks of 30
    chunk_indices = velocity.index // WINDOW
    # Calculate mean for each chunk
    chunked_average = velocity.groupby(chunk_indices).mean()
    return chunked_average

def assign_ROI(median_position_df):
    ROI_labels = np.where((median_position_df['x'] > 343) & (median_position_df['y'] > 291), 'ROI', 'OF')
    median_position_df['label'] = ROI_labels
    return median_position_df


def get_rotary_metadata(exp_folder):
        try:
            TICKS_PER_CYCLE = 1024
            rotary = np.load(os.path.join(exp_folder, 'rotaryEncoder.raw.npy'), allow_pickle=True)
            rotary = rotary.flatten()
            rotary[rotary > 2**31] = rotary[rotary > 2**31] - 2**32
            
            timeline_file = glob.glob(os.path.join(exp_folder, f'*_Timeline.mat'))[0]   
            time = loadmat(timeline_file)
            rotary_timestamps = time['Timeline']['rawDAQTimestamps'].item()[0, :]
            rotary_position = 360* rotary / (TICKS_PER_CYCLE*4)

            return rotary_timestamps, rotary_position
            

        except Exception as e:
            print(f"Error accessing {exp_folder}: {e}")
            
        return None, None

def get_top_cam_timesamps(subject_id, date, exp_num, exp_folder):
     
    top_cam_path = fr'ONE_preproc\topCam\camera.times.{date}_{exp_num}_{subject_id}_topCam.npy'
    #print(f'exp_folder: {exp_folder}')
    #print(f'top_cam_folder: {top_cam_path} ')
    #print(f'combo: {os.path.join(exp_folder, top_cam_path)}')
    top_cam_timestamps= np.load(os.path.join(exp_folder, top_cam_path), allow_pickle=True)
    delay = math.ceil(abs((top_cam_timestamps[0] / .01660)))
    print(delay)
    top_cam_timestamps = top_cam_timestamps[delay:]

    return delay, top_cam_timestamps
     
def calculate_wheel_velocity(rotary_timestamps, rotary_position, top_cam_timestamps):
    sampling_rate = 1 / np.diff(rotary_timestamps)[0]
    wheel_velocity_raw = np.diff(np.concatenate([[0], rotary_position])) / sampling_rate
    downsampled_timestamps = np.linspace(rotary_timestamps[0], rotary_timestamps[-1], len(top_cam_timestamps))
    wheel_velocity = gaussian_filter1d(wheel_velocity_raw, 2)
    wheel_velocity = np.interp(downsampled_timestamps, rotary_timestamps, wheel_velocity)

    return downsampled_timestamps, wheel_velocity



# date = '2024-03-04'
# subject_id = 'AV043'

# data = load_specific_experiment_data(subject_id, date)
# dlc_df, scorer = get_DLC_data(subject_id, date, delay)
# exp_num, exp_folder = get_experiment_path(data)
# median_position_df = calculate_median_position(dlc_df, scorer)
# distances = calculate_distance(median_position_df)
# velocity = calculate_velocity(distances)
# rotary_timestamps, rotary_speed = get_rotary_metadata(exp_folder)
# top_cam_timestamps= get_top_cam_timesamps(subject_id, date, exp_num, exp_folder)
# print(top_cam_timestamps[0])
# #downsampled_timestamps, wheel_velocity = calculate_wheel_velocity(rotary_timestamps, rotary_speed, velocity)
