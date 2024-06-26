# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:44:36 2024

@author: Johannes
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

DIRECTORY = "C:/PATH/TO/STORE/DATA"

# mass for bulkdensity calibration
mass = 1000 #g

# dried apricots
density = 0.0012104976483512985 #g/mm³

# red plastic cubes
#density = 0.0008444570839365841 #g/mm³

# white rubber fries
#density = 0.0006681194862550151 #g/mm³

SCREEN_WIDTH = 640 #pixels
SCREEN_HEIGHT = 480 #pixels

HFOV = 0 #°
VFOV = 0 #°

HFOV_RAD = 0 #rad
VFOV_RAD = 0 #rad

FPS = 60

intrinsics = 0

FOCAL_LENGTHX = 0
FOCAL_LENGTHY = 0

# ranges for depth colormap
#DEPTHRANGE = [720 , 735]
DEPTHRANGE = [650 , 750]
#DEPTHRANGE = [10 , 10000]
#DEPTHRANGE = [450 , 550]

# cutoff ranges
DEPTH_THRESHOLD_MIN = 0  #mm
DEPTH_THRESHOLD_MAX = DEPTHRANGE[1] #mm

# range for height colormap
HEIGHTRANGE = [0, 100]

# stepsize for changing depth cut-off value
VARIATION = 1 #mm 

###############################################################

recording = 0
'''tarrecounter = 0'''
calibrate_counter = 0
framecounter = 0

mean_distance = 0 #m

def getFOV(profile):
    global HFOV, VFOV, HFOV_RAD, VFOV_RAD, FOCAL_LENGTHX, FOCAL_LENGTHY, SCREEN_WIDTH, SCREEN_HEIGHT, intrinsics
    
    # Get camera intrinsics
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    FOCAL_LENGTHX = intrinsics.fx  # fx focal length in x
    FOCAL_LENGTHY = intrinsics.fy  # fy focal length in y
    
    # screens pixels
    SCREEN_WIDTH = intrinsics.width
    SCREEN_HEIGHT = intrinsics.height
    
    # Horizontal and vertical FOV
    HFOV_RAD = 2 * math.atan(SCREEN_WIDTH / (2 * FOCAL_LENGTHX)) # rad
    VFOV_RAD = 2 * math.atan(SCREEN_HEIGHT / (2 * FOCAL_LENGTHY)) # rad
    
    HFOV = HFOV_RAD * (180 / math.pi) # °
    VFOV = VFOV_RAD * (180 / math.pi) # °
    
    # Print camera intrinsics
    print("Focal Lengths: {}x{}".format(intrinsics.fx, intrinsics.fy))
    print("Principal Point: {}x{}".format(intrinsics.ppx, intrinsics.ppy))
    print("Distortion Coefficients: {}".format(intrinsics.coeffs))
    
    print("screenwidth:", SCREEN_WIDTH, "pixels")
    print("screenheight:", SCREEN_HEIGHT, "pixels")
    
    print("Horizontale FOV:", HFOV, "graden")
    print("Verticale FOV:", VFOV, "graden")
    
######################################################################

def define_area_of_interest(small):
    if small:
        topleft_x = 260  # pixels
        topleft_y = 150  # pixels
        height = 150  # pixels
        width = 80  # pixels
    else:
        topleft_x = 110  # pixels
        topleft_y = 50  # pixels
        height = 390  # pixels
        width = 430  # pixels

    return topleft_x, topleft_y, height, width

RECT_TOPLEFTX, RECT_TOPLEFTY, RECT_HEIGHT, RECT_WIDTH = define_area_of_interest(True)
area_of_interest = [(RECT_TOPLEFTX, RECT_TOPLEFTY), (RECT_TOPLEFTX + RECT_WIDTH, RECT_TOPLEFTY + RECT_HEIGHT)]

#######################################################################

depthpoint = (0, 0)
def show_depth(event, x, y, args, params):
    global depthpoint
    
    x = min(x, SCREEN_WIDTH-1)
    y = min(y, SCREEN_HEIGHT-1)
    
    depthpoint = (x, y)

heightpoint = (0, 0)
def show_height(event, x, y, flags, param):
    global heightpoint
    
    x = min(x, SCREEN_WIDTH-1)
    y = min(y, SCREEN_HEIGHT-1)
    
    heightpoint = (x, y)

def apply_filters(depth_frame):
    #spatial filter
    Filter_mag = 5 #1-5
    Smooth_alpha = 0.25 #0.25-1
    Smooth_delta = 20 #1-50
    Hole_filling = 0 #0-5
    
    #temporal filter
    Alpha = 0.5 #0-1
    Delta = 20 #1-100
    Persistence = 3 #1-8
    
    #hole filling filter
    filling = 1 #0-2
    
    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter(Smooth_alpha, Smooth_delta, Filter_mag, Hole_filling)
    temp_filter = rs.temporal_filter(Alpha, Delta, Persistence)
    hole_filter = rs.hole_filling_filter(filling)

    # Apply spatial and temporal filters
    #depth_frame = dec_filter.process(depth_frame)
    #depth_frame = spat_filter.process(depth_frame)
    #depth_frame = temp_filter.process(depth_frame)
    depth_frame = hole_filter.process(depth_frame)
    
    return depth_frame

def visualize_depth(depth_image):
    global depthpoint

    #clipping
    depth_image_scaled = (depth_image - DEPTHRANGE[0]) * (255 / (DEPTHRANGE[1]-DEPTHRANGE[0]))
    depth_image_scaled = np.clip(depth_image_scaled, 0, 255).astype(np.uint8)
    
    depth_colormap = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_HOT)
    
    cv2.circle(depth_colormap, depthpoint, 4, (0, 0, 255), -1)
    depth_cursor = depth_image[depthpoint[1], depthpoint[0]]
    cv2.putText(depth_colormap, "Depth: {:.0f}mm".format(depth_cursor), (depthpoint[0] - 50, depthpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.rectangle(depth_colormap, (area_of_interest[0][0], area_of_interest[0][1]), (area_of_interest[1][0], area_of_interest[1][1]), (0, 255, 0), 2)
    
    return depth_colormap

def visualize_height(depth_image):
    global heightpoint
    
    height_image = DEPTH_THRESHOLD_MAX - (depth_image)
    
    height_image[height_image < 0] = 0 # all negative values become 0
    height_image[height_image >= DEPTH_THRESHOLD_MAX] = 0 # values too great get limited

    #clipping
    height_image_scaled = (height_image - HEIGHTRANGE[0]) * (255 / (HEIGHTRANGE[1]-HEIGHTRANGE[0]))
    height_image_scaled = np.clip(height_image_scaled, 0, 255).astype(np.uint8)

    height_colormap = cv2.applyColorMap(height_image_scaled, cv2.COLORMAP_HOT)
    cv2.circle(height_colormap, heightpoint, 4, (0, 0, 255), -1)
    
    # Draw the pre-defined rectangle on the height window
    cv2.rectangle(height_colormap, (area_of_interest[0][0], area_of_interest[0][1]), (area_of_interest[1][0], area_of_interest[1][1]), (0, 255, 0), 2)
    
    volume = measure(depth_image)
    
    cv2.putText(height_colormap, "Sum in rectangle: {:.0f}mm3".format(volume), (10, height_colormap.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(height_colormap, "Max Depth: {:.0f}mm".format(DEPTH_THRESHOLD_MAX), (height_colormap.shape[0] - 60, height_colormap.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display height value at cursor position
    height_cursor = height_image[heightpoint[1], heightpoint[0]]
    cv2.putText(height_colormap, "Height: {:.0f}mm".format(height_cursor), (heightpoint[0] - 50, heightpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    
    return height_colormap    

def plot3Dpixels(depth_image):
    
    depths_aoi = depth_image[area_of_interest[0][1]:area_of_interest[1][1], area_of_interest[0][0]:area_of_interest[1][0]]
    heights_aoi = DEPTH_THRESHOLD_MAX - depths_aoi
    
    heights_aoi[heights_aoi < 0] = 0
    heights_aoi[heights_aoi >= DEPTH_THRESHOLD_MAX] = DEPTH_THRESHOLD_MIN
    
    height, width = heights_aoi.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, heights_aoi, cmap='viridis')

    ax.set_xlabel('X axis (pixels)')
    ax.set_ylabel('Y axis (pixels)')
    ax.set_zlabel('Height (mm)')
    ax.set_title('3D Plot of Height Data')

    plt.show()

def plot3Dreal(depth_image):
    depths_aoi = depth_image[area_of_interest[0][1]:area_of_interest[1][1], area_of_interest[0][0]:area_of_interest[1][0]]
    heights_aoi = DEPTH_THRESHOLD_MAX - depths_aoi
    
    heights_aoi[heights_aoi < 0] = 0
    heights_aoi[heights_aoi >= DEPTH_THRESHOLD_MAX] = DEPTH_THRESHOLD_MIN
    
    heightmax = np.max(heights_aoi)
    z_mean = np.mean(depths_aoi)
    
    height, width = heights_aoi.shape
    x_mean = (2 * np.tan(HFOV_RAD / 2) * z_mean) / SCREEN_WIDTH  # mm/pixel
    y_mean = (2 * np.tan(VFOV_RAD / 2) * z_mean) / SCREEN_HEIGHT  # mm/pixel

    x_coords = np.linspace(0, width - 1, width) * x_mean
    y_coords = np.linspace(0, height - 1, height) * y_mean

    x_coords, y_coords = np.meshgrid(x_coords, y_coords)

    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    heights_flat = heights_aoi.flatten()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x_coords_flat, y_coords_flat, heights_flat, c=heights_flat, cmap='viridis')

    ax.set_xlabel('X axis (mm)')
    ax.set_ylabel('Y axis (mm)')
    ax.set_zlabel('Height (mm)')
    ax.set_title('3D Plot of Height Data')

    fig.colorbar(sc, ax=ax, label='Height (mm)')

    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    z_range = heightmax - 0

    max_range = max(x_range, y_range, z_range)

    ax.set_xlim([x_coords.min(), x_coords.max()])
    ax.set_ylim([y_coords.min(), y_coords.max()])
    ax.set_zlim([0, heightmax])

    ax.set_box_aspect([x_range, y_range, z_range])

    plt.show()


def put_filter(depth_frame):
    # filtering
    depth_frame_filtered = apply_filters(depth_frame)
    # to numpy array
    depth_array = np.asanyarray(depth_frame_filtered.get_data())
    
    return depth_array

# Global variables to store measurements
object_measurements = []
volume_calibration = []

tarre2_execute_flag = False

def tarre2(depth_image):
    global tarre2_execute_flag, DEPTH_THRESHOLD_MAX
    
    volume = measure(depth_image)
    
    if volume > 10000:
        DEPTH_THRESHOLD_MAX = DEPTH_THRESHOLD_MAX - 1
    else:
        tarre2_execute_flag = False
        
def measure(depth_image_filtered):
    global mean_distance, object_measurements, recordcounter
    
    depths_aoi = depth_image_filtered[area_of_interest[0][1]:area_of_interest[1][1], area_of_interest[0][0]:area_of_interest[1][0]]
    heights_aoi = DEPTH_THRESHOLD_MAX - depths_aoi
    
    heights_aoi[heights_aoi < 0] = 0
    heights_aoi[heights_aoi >= DEPTH_THRESHOLD_MAX] = DEPTH_THRESHOLD_MIN

    voxels = dynamic_areas(depths_aoi) * heights_aoi
    object_volume = np.sum(voxels)
    
    #print("Volume object = {:.2f}mm³".format(object_volume)) # * density
    
    return object_volume

def calc_density(mass, depth_image):
    
    volume = measure(depth_image)
    
    #  list to get average --> bulkdensity
    volume_calibration.append(volume)
    
    density = mass/volume
    return 

def get_density(depth_image, mass):
    global calibrate_counter, volume_calibration, density
    
    if calibrate_counter == 1:
        volume = measure(depth_image)
        volume_calibration.append(volume)
    if calibrate_counter == 2:
        
        avg_volume = np.mean(volume_calibration) #mm³
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(volume_calibration, label='Raw Volume Measurements', alpha=0.5)
        #plt.plot(range(window_size - 1, len(smoothed_measurements) + window_size - 1), smoothed_measurements, label='Smoothed Volume Measurements', color='orange')
        plt.axhline(y=avg_volume, color='r', linestyle='-', label='Average Raw Volume')
        #plt.axhline(y=average_volume_smoothed, color='green', linestyle='--', label='Average Smoothed Volume')
        plt.xlabel('Measurement Number')
        plt.ylabel('Volume in mm³')
        plt.title('Background Volume Measurements')
        plt.legend()
        plt.show()
        
        density = mass / avg_volume # g/mm³
        
        print(density)
        
        volume_calibration = [] #reset
        calibrate_counter = 0

def get_flatness(depths):
    depths_aoi = depths[area_of_interest[0][1]:area_of_interest[1][1], area_of_interest[0][0]:area_of_interest[1][0]]
    
    minimum = np.min(depths_aoi)
    maximum = np.max(depths_aoi)
    
    flatness = maximum - minimum
    print("Depth difference inside area of interest: {:.2f}mm".format(flatness))
    
    if flatness < 20:
        value = "flat enough"
    else:
        value = "not flat enough"
    
    '''
    average = np.mean(depths_aoi)
    print(f"Average : {average}")
    '''
    print(f"Flatness : {value}")
    
    return flatness

def dynamic_area(x_mean, y_mean, z_mean, z_new):
    new_area = (z_new/z_mean * x_mean)*(z_new/z_mean * y_mean)
    return new_area

def dynamic_areas(depths):
    global HFOV_RAD, VFOV_RAD, SCREEN_WIDTH, SCREEN_HEIGHT
    
    z_mean = np.mean(depths)
    x_mean = (2 * math.tan(HFOV_RAD/ 2) * z_mean)/SCREEN_WIDTH # mm/pixel
    y_mean = (2 * math.tan(VFOV_RAD/ 2) * z_mean)/SCREEN_HEIGHT # mm/pixel
    
    #print(f" pixelX: {x_mean} mm/pixel")
    #print(f" pixelY: {y_mean} mm/pixel")
    
    new_areas = dynamic_area(x_mean, y_mean, z_mean, depths)
    return new_areas
'''
def calculatePixelAreas(depths):
    #depth in mm
    global HFOV_RAD, VFOV_RAD
    
    pixel_x = 2 * math.tan(HFOV_RAD/ 2) * depths # mm
    pixel_y = 2 * math.tan(VFOV_RAD/ 2) * depths # mm
    
    px = pixel_x/SCREEN_WIDTH
    py = pixel_y/SCREEN_HEIGHT
    
    pixel_areas = px * py #mm²
    return pixel_areas

def calc(depths):
    #depth in mm
    global HFOV_RAD, VFOV_RAD, SCREEN_WIDTH, SCREEN_HEIGHT

    mean_distance = np.mean(depths)

    avg_pixel_hor = (2 * math.tan(HFOV_RAD/ 2) * mean_distance)/SCREEN_WIDTH # mm/pixel
    avg_pixel_ver = (2 * math.tan(VFOV_RAD/ 2) * mean_distance)/SCREEN_HEIGHT # mm/pixel
    
    avg_pixel_area = avg_pixel_hor * avg_pixel_ver
    print(avg_pixel_area)
    
    return avg_pixel_area

def getPixelArea(depth):
    #depth in mm
    global HFOV_RAD, VFOV_RAD
    
    pixel_x = 2 * math.tan(HFOV_RAD/ 2) * depth # mm
    pixel_y = 2 * math.tan(VFOV_RAD/ 2) * depth # mm
    
    px = pixel_x/SCREEN_WIDTH
    py = pixel_y/SCREEN_HEIGHT
    
    pixel_area = px * py #mm²
    return pixel_area

def getPixelAreaMean(depth):
    #depth in mm
    global HFOV_RAD, VFOV_RAD
    
    z_mean = np.mean(depth)
    x_mean = (2 * math.tan(HFOV_RAD/ 2) * z_mean)/SCREEN_WIDTH # mm/pixel
    
    pixel_area = x_mean * x_mean #mm²
    return pixel_area
'''
# Create a pipeline
pipe = rs.pipeline()

# Configuration for the pipeline
cfg = rs.config()
cfg.enable_stream(rs.stream.color, SCREEN_WIDTH, SCREEN_HEIGHT, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, SCREEN_WIDTH, SCREEN_HEIGHT, rs.format.z16, FPS)

cv2.namedWindow("Depth Colormap with Legend")
cv2.setMouseCallback("Depth Colormap with Legend", show_depth)

cv2.namedWindow("Height Colormap with Legend")
cv2.setMouseCallback("Height Colormap with Legend", show_height)

# Start the pipeline
profile = pipe.start(cfg)
getFOV(profile)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()

if depth_sensor:
    # Get depth scale (units) - how many meters each depth unit corresponds to
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale (units): {depth_scale} meters per unit")
else:
    print("No depth sensor found.")

def moving_average(data, window_size):
    """Compute the moving average of the given data using a window of specified size."""
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

def outputToCSV(depth_frame, color_frame, save_directory=DIRECTORY):
    global recording, object_measurements, framecounter
    
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Define file path
    file_path = os.path.join(save_directory, "output.csv")
    
    if recording == 1:
        object_volume = measure(depth_frame)
        object_measurements.append(object_volume * density)
        framecounter += 1
        save_depth_frame(depth_frame, framecounter)
        save_height_frame(depth_frame, framecounter)
        save_color_frame(color_frame, framecounter)
    if recording == 2:
        np.savetxt(file_path, object_measurements, delimiter=",", fmt='%d')
        object_measurements = []
        recording = 0

def save_depth_frame(depth_frame, frame_index, save_directory=DIRECTORY):
    
    depth_colormap = visualize_depth(depth_frame)
    legend_depth = create_legend(SCREEN_HEIGHT, 80, DEPTHRANGE)
    
    combined_image_depth = np.hstack((depth_colormap, legend_depth))
    
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define file path
    file_path = os.path.join(save_directory, f"depth_frame_{frame_index}.npy")

    # Save the depth frame as a .npy file
    np.save(file_path, combined_image_depth)
    
def save_height_frame(depth_frame, frame_index, save_directory=DIRECTORY):
    
    height_colormap = visualize_height(depth_frame)
    legend_height = create_legend(SCREEN_HEIGHT, 80, HEIGHTRANGE)
    
    combined_image_height = np.hstack((height_colormap, legend_height))
    
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define file path
    file_path = os.path.join(save_directory, f"height_frame_{frame_index}.npy")

    # Save the depth frame as a .npy file
    np.save(file_path, combined_image_height)

def save_color_frame(color_frame, frame_index, save_directory=DIRECTORY):
    
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define file path
    file_path = os.path.join(save_directory, f"color_frame_{frame_index}.npy")

    # Convert the pyrealsense2 video_frame to a numpy array
    # Assuming color_frame is a pyrealsense2 frame object
    color_array = np.asanyarray(color_frame.get_data())

    # Save the depth frame as a .npy file
    np.save(file_path, color_array)


def create_legend(height, width, ranges):
    pad_top=10
    pad_bottom=10
    pad_left=10
    pad_right=10
    
    #height = height - pad_top - pad_bottom
    
    # Create a gradient image for the legend
    gradient = np.tile(np.linspace(255, 0, height, dtype=np.uint8), (width, 1)).T
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_HOT)
    
    intervals = 10  # Change the number of intervals based on your preference
    font_scale = 0.4
    font_thickness = 1
    font_color = (0, 255, 0)  # Green color
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    
    legend = cv2.copyMakeBorder(gradient, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    for i in range(intervals + 1):
        cv2.putText(legend, f' - {int(ranges[0] - i / intervals * (ranges[0] - ranges[1]))}mm',
                    (0, int(height - i * height / intervals - pad_top - 3)), font_type,
                    font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    return legend

# Continuous loop: working camera
while True:
    frame = pipe.wait_for_frames()

    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    # Using own functions: colormap is scaled depthframe to get a distinct in difference in colors at a range of interest
    depth_image = put_filter(depth_frame)
    depth_colormap = visualize_depth(depth_image)
    height_colormap = visualize_height(depth_image)
    
    # Use cv2 to display images
    cv2.imshow('rgb', np.asanyarray(color_frame.get_data()))
    #cv2.imshow('depth', depth_colormap)
    #cv2.imshow('height', height_colormap)

    legend_depth = create_legend(SCREEN_HEIGHT, 80, DEPTHRANGE)
    legend_height = create_legend(SCREEN_HEIGHT, 80, HEIGHTRANGE)
    
    combined_image_depth = np.hstack((depth_colormap, legend_depth))
    cv2.imshow('Depth Colormap with Legend', combined_image_depth)

    combined_image_height = np.hstack((height_colormap, legend_height))
    cv2.imshow('Height Colormap with Legend', combined_image_height)
    
    # Check for user input to calibrate and start the measurements
    key = cv2.waitKey(1)
    
    if key == ord('a'):
        get_flatness(depth_image)
        
    # manual cutoff
    elif key == ord('t'):
        DEPTH_THRESHOLD_MAX = min(DEPTH_THRESHOLD_MAX + VARIATION, 1000.0)  # Ensure max_depth does not go above 1m
    elif key == ord('g'):
        DEPTH_THRESHOLD_MAX = max(DEPTH_THRESHOLD_MIN, DEPTH_THRESHOLD_MAX - VARIATION)  # Ensure max_depth does not go below min_depth
    
    # automatic cutoff (recommended)
    elif key == ord('o'): #start tarre
        tarre2_execute_flag = True
    
    elif key == ord('l'): #reset tarre
    
        DEPTH_THRESHOLD_MAX = DEPTHRANGE[1]
        tarre2_execute_flag = False
    
    # calibrate/measure density
    elif key == ord('z'):
        calibrate_counter = 1
    elif key == ord('s'):
        calibrate_counter = 2
    
    # start recording
    elif key == ord('p'): #start measuring and append to the list
        recording = 1
        framecounter = 0
    elif key == ord('m'): #stop measurements and output into csv file.
        recording = 2
        
    if key == ord('v'): #make a 3d plot of the area of interest
        plot3Dpixels(depth_image)
        plot3Dreal(depth_image)
        
    if key == ord('q'):
        break
    
    if tarre2_execute_flag == True:
        tarre2(depth_image)
    
    get_density(depth_image, mass)
    outputToCSV(depth_image, color_frame)
# Stop the pipeline
pipe.stop()
cv2.destroyAllWindows()