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

def calculate_median_position(dlc_df, scorer, BODYPARTS = ['neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3']):


    
    bodypart_positions = dlc_df.loc[:, (scorer, BODYPARTS, slice(None))]
    

    
    likelihood_values = bodypart_positions.xs('likelihood', level='coords', axis=1)
    low_filter = likelihood_values <= 0.95
    strong_x = bodypart_positions.xs('x', level='coords', axis=1)  # Define first
    strong_y = bodypart_positions.xs('y', level='coords', axis=1) 
    strong_x[low_filter] = np.nan
    strong_y[low_filter] = np.nan
    strong_x = strong_x.interpolate(method='linear', axis=0)
    strong_y = strong_y.interpolate(method='linear', axis=0)
    x = strong_x.median(axis=1)
    y = strong_y.median(axis=1)
    
    return  x, y

def bin_median_positions(x, y,timestamps, start_time, bin_centers): 
    binned_x = np.interp(bin_centers, np.linspace(timestamps[start_time], timestamps[-1], len(x)), x)
    binned_y = np.interp(bin_centers, np.linspace(timestamps[start_time], timestamps[-1], len(y)), y)
    return binned_x, binned_y
    

def calculate_velocity(binned_x, binned_y, bin_width):
    
    distances = np.sqrt(np.diff(binned_x)**2 + np.diff(binned_y)**2)
    max_distance = np.percentile(distances, 99)  
    distances[distances > max_distance] = np.nan
    valid_indices = np.where(~np.isnan(distances))[0]
    distances = np.interp(
    np.arange(len(distances)),  
    valid_indices,              
    distances[valid_indices]    
)
    velocity_pix = distances / bin_width
    conversion_factor = 0.07 
    velocity = velocity_pix * conversion_factor
    velocity = gaussian_filter1d(velocity, 3)
    velocity = np.concatenate(([velocity[0] if len(velocity) > 0 else 0], velocity)) 
    return velocity




def calculate_wheel_velocity(rotary_position, bin_width, wheel_diameter=10):
    wheel_circumference = np.pi * wheel_diameter
    linear_distance_cm = np.diff(rotary_position) * wheel_circumference / 360 
    wheel_velocity = np.abs(linear_distance_cm / bin_width) 
    wheel_velocity = gaussian_filter1d(wheel_velocity, 3)
    wheel_velocity = np.concatenate(([wheel_velocity[0] if len(wheel_velocity) > 0 else 0], wheel_velocity))
    return wheel_velocity

