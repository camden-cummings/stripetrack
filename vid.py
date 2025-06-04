import os
from multiprocessing import Pool
#import cProfile, pstats, io
#from pstats import SortKey

import PySpin
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import pandas as pd
import serial

from precise_time import PreciseTime
from tracker.cell_finder_helpers.calc_mode import calc_mode
from AcquireAndDisplayClass import get_image
from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_device_serial_number
from gui_helpers import GUIHelpers
from tracker.helpers.centroid_manip import find_centroid_of_contour, check_masked_image, generate_row_col
from structural_sim_from_scratch import correlate1d_x, correlate1d_y, run_math, setup as ssim_setup #structural_similarity

import math

global continue_recording
global run_once

run_once = True

DESIRED_MODE_FRAMES = 50
   
weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
 0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]

np_weights = np.asarray(weights)
    
FRAMES_TO_SAVE_AFTER = 100
pre_output_filepath = 'pre-processed.csv'

class RunCV:
    def __init__(self, FRAME_WIDTH, FRAME_HEIGHT, output_filepath, cell_contours, cell_centers, shape_of_rows):
        self.async_result = None

        self.mode_noblur_img = None
        self.curr_img = None
        self.curr_img_data = None
        self.mask = None
        self.masked_mode_noblur_img = None

        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.FRAME_WIDTH = FRAME_WIDTH
        self.detected_centroids = []
        self.output_filepath = output_filepath
        self.movie_deq = []
        self.cell_contours = cell_contours
        self.cell_centers = cell_centers
        self.shape_of_rows = shape_of_rows
        
        self.ux_tmp, self.uy_tmp, self.uxx_tmp, self.uyy_tmp, self.uxy_tmp = ssim_setup(self.FRAME_WIDTH, self.FRAME_HEIGHT)
        self.ux, self.uy, self.uxx, self.uyy, self.uxy = ssim_setup(self.FRAME_HEIGHT, self.FRAME_WIDTH)

        sigma = 1.5
        truncate = 3.5
        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1

        # ndim will always be 2 because we're always dealing with 2d images 
        ndim = 2
        
        self.cov_norm = win_size**ndim / (win_size**ndim - 1)  # sample covariance

        weight_size = len(weights)
        self.size1 = math.floor(weight_size / 2)
        self.size2 = weight_size - self.size1 - 1

        self.np_weights = np.array(weights, dtype=np.float64)
        
        
    def find_mode(self, frame_counter):
        global run_once

        if len(self.movie_deq) < DESIRED_MODE_FRAMES and frame_counter % 50 == 0:
            self.movie_deq.append(self.curr_img)
        elif len(self.movie_deq) >= DESIRED_MODE_FRAMES and run_once == True:
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (self.movie_deq, self.FRAME_HEIGHT, self.FRAME_WIDTH))
            #mode_noblur_img = calc_mode(movie_deq, FRAME_HEIGHT, FRAME_WIDTH)
            
            run_once = False

        if self.async_result is not None and self.async_result.ready():
            self.mode_noblur_img = self.async_result.get()

            #self.gui.mode_calculated = True
            #self.gui.rt_tracker.standard_image_noise = self.gui.rt_tracker.CV_image_noise_light_background(self.mode_noblur_img)
            #dpg.configure_item(self.gui.status, default_value="Status: Ready")

    def sort_contours_by_area(self, contours, frame_count, time, diff, min_centroid_size, max_centroid_size):  # TODO put this back into runcv -- but want to check that it makes sense first
        #min_centroid_size // self.gui.contour_definer.centroid_size
        #max_centroid_size // 500000
        # darkest_pixel_val = 255
        posns = [[[] for j in range(self.shape_of_rows[i])] for i in
                 range(len(self.shape_of_rows))]
        last_confident_centroid = [[self.cell_centers[row][col] for col in range(self.shape_of_rows[row])] for row in
                                   range(len(self.shape_of_rows))]
        sorted_contours = []

        # print(shape_of_rows)
        for c in contours:
            # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
            # a well, and that we know which one it is in)

            contours = [c for c in contours if min_centroid_size < cv2.contourArea(c) < max_centroid_size]
            if 50 < cv2.contourArea(c) < 500000:
                x, y, w, h = cv2.boundingRect(c)

                point_x, point_y = (int(x + w / 2),
                                    int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster

                if not check_masked_image((point_x, point_y), self.mask):
                    for row, col in generate_row_col(self.shape_of_rows):
                        if math.dist(self.cell_centers[row][col], (point_x, point_y)) < 50:
                            cell_count = row * self.shape_of_rows[row] + col

                            in_polygon = cv2.pointPolygonTest(self.cell_contours[cell_count], (point_x, point_y),
                                                              False)
                            # print(in_polygon)
                            if in_polygon >= 0:
                                # print(diff_box)
                                n = cv2.mean(diff[y:y + h, x:x + w])[0]
                                posns[row][col].append(((point_x, point_y), n))

                                break

        for row, col in generate_row_col(self.shape_of_rows):
            ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]
            if len(ten_darkest_centroids) > 0:
                sorted_contours.append(
                    [time, frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1]])
                last_confident_centroid[row][col] = ten_darkest_centroids[0][0]
            else:
                sorted_contours.append([time, frame_count, row, col, last_confident_centroid[row][col][0],
                                        last_confident_centroid[row][col][1]])
                # for c in ten_darkest_centroids:
                #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
        return sorted_contours

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
        at_time[0] += 12

    video_type = cmd_string.iloc[2]

    return at_time, arduino_command, video_type
