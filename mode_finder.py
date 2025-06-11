from multiprocessing import Pool
#import cProfile, pstats, io
#from pstats import SortKey

import PySpin
import cv2
import numpy as np
import pandas as pd

from tracker.cell_finder_helpers.calc_mode import calc_mode
from tracker.helpers.centroid_manip import check_masked_image, generate_row_col
from structural_sim_from_scratch import setup as ssim_setup #structural_similarity

import math

global continue_recording
global run_once

run_once = True

DESIRED_MODE_FRAMES = 2#50
FRAMES_TO_SAVE_AFTER = 2#100

class ModeFinder:
    def __init__(self, FRAME_WIDTH, FRAME_HEIGHT):
        self.async_result = None

        self.mode_noblur_img = None
        self.curr_img = None
        self.curr_img_data = None
        self.mask = None
        self.masked_mode_noblur_img = None

        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.FRAME_WIDTH = FRAME_WIDTH

        self.movie_deq = []

        self.not_found_mode = True
        self.setup = False
        
        
    def find_mode(self, frame_counter):
        global run_once
        #print(len(self.movie_deq))
        if len(self.movie_deq) < DESIRED_MODE_FRAMES and frame_counter % 50 == 0:
            print("1 - ")
            self.movie_deq.append(self.curr_img)
        elif len(self.movie_deq) >= DESIRED_MODE_FRAMES and run_once == True:
            print("2 - ")
            print("run once", run_once)
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (self.movie_deq, self.FRAME_HEIGHT, self.FRAME_WIDTH))
            #mode_noblur_img = calc_mode(movie_deq, FRAME_HEIGHT, FRAME_WIDTH)
            
            run_once = False
       
        if self.async_result is not None:
            print("asnc", self.async_result.ready())
        if self.async_result is not None and self.async_result.ready():
            print("3 - ")
            self.mode_noblur_img = self.async_result.get()
            self.movie_deq.clear()
            self.setup=False
            self.async_result = None
            self.not_found_mode = False
            run_once = True
            print("4")
            #self.gui.mode_calculated = True
            #self.gui.rt_tracker.standard_image_noise = self.gui.rt_tracker.CV_image_noise_light_background(self.mode_noblur_img)
            #dpg.configure_item(self.gui.status, default_value="Status: Ready")

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
        at_time[0] += 12

    video_type = cmd_string.iloc[2]

    return at_time, arduino_command, video_type
