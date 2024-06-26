# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:21:52 2024

@author: Johannes
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os 
import re

DIRECTORY = "C:/PATH/TO/LOAD/DATA"

def load_frame(file_path):
    frame = np.load(file_path)
    return frame

def numerical_sort(value):
    """
    Helper function to extract numbers from filenames for correct sorting.
    """
    parts = re.compile(r'(\d+)').findall(value)
    parts = map(int, parts)  # convert all to integer
    return list(parts)

def replay_frames(save_directory=DIRECTORY):
    while True:
        # list all files in the directory for depth, height and color separately
        depth_files = [f for f in os.listdir(save_directory) if f.startswith('depth_frame_') and f.endswith('.npy')]
        height_files = [f for f in os.listdir(save_directory) if f.startswith('height_frame_') and f.endswith('.npy')]
        color_files = [f for f in os.listdir(save_directory) if f.startswith('color_frame_') and f.endswith('.npy')]

        # sorted numerically
        depth_files = sorted(depth_files, key=numerical_sort)
        height_files = sorted(height_files, key=numerical_sort)
        color_files = sorted(color_files, key=numerical_sort)

        # load and process each pair of depth and color frames
        for depth_file, height_file, color_file in zip(depth_files, height_files, color_files):
            depth_path = os.path.join(save_directory, depth_file)
            height_path = os.path.join(save_directory, height_file)
            color_path = os.path.join(save_directory, color_file)
            
            depth_frame = load_frame(depth_path)
            height_frame = load_frame(height_path)
            color_frame = load_frame(color_path)
            
            print(f"Loaded depth frame from {depth_path}")
            print(f"Loaded height frame from {height_path}")
            print(f"Loaded color frame from {color_path}")
            
            # visualize
            visualize_frame(depth_frame, 'Depth Frame')
            visualize_frame(height_frame, 'Height Frame')
            visualize_frame(color_frame, 'Color Frame')
            
            # user input to break loop or continue
            while True:
                key = cv2.waitKey(0) & 0xFF 
                if key == ord('q'):  # exit loop
                    cv2.destroyAllWindows()
                    return
                elif key == ord('p'):  # press p to go to next frame. manually go to next frame
                    break
                
def visualize_frame(frame, window_name):
    cv2.imshow(window_name, frame)
    #cv2.waitKey(50)  # Display frame until a key is pressed. plays automatically

replay_frames()