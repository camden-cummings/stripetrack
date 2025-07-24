import cProfile
import gc
import io
import math
import multiprocessing
import pstats
from multiprocessing import Process
from pstats import SortKey

import PySpin
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import serial

from command_reader import process_command_string
from gui_helpers import GUIHelpers
from mode_finder import ModeFinder
from no_gui_tracker import PoolRun
from precise_time import PreciseTime
from sort_contours_by_area import sort_contours_by_area
from strsim_for_speed.structural_sim_from_scratch import correlate1d_x, correlate1d_y, run_math, run_math_complete, normalize_diff, setup as ssim_setup, generate_weights
# do this through GUI instead
fn_start = "C:\\Users\\ThymeLab\\Desktop\\6-27-25-test\\"

import logging

# TODO try memlog to double check
"""
# create logger
mem_logger = logging.getLogger('memory_profile_log')
mem_logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler("memory_profile.log")
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to the logger
mem_logger.addHandler(fh)

from memory_profiler import LogFile
import sys
sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)

#from memory_profiler import profile
"""
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{fn_start}run.log', encoding='utf-8', level=logging.DEBUG)

fps = 30.0

class GUIPoolRun(PoolRun):
    def __init__(self):
        self.FRAME_HEIGHT, self.FRAME_WIDTH = 660, 992
        self.image_data = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT, 3))

  #  @profile
    def tracking_pool(self, img_queue, done, start_recording):
        logger.info("start gui pool")

        dpg.create_context()
        window = dpg.add_window(label="Video player", pos=(50, 50), width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT) 
        gui = GUIHelpers(window, self.FRAME_WIDTH, self.FRAME_HEIGHT)

        timer = PreciseTime()

        dpg.set_primary_window(window, True)
        dpg.create_viewport(width=int(self.FRAME_WIDTH*1.5), height=self.FRAME_HEIGHT+20, title="ROI Selector")
        dpg.setup_dearpygui()
        dpg.show_viewport()

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)#, f'{fn_start}pre-processed.csv', gui)
        frame_counter = 0
            

        data_range = 255
        
        weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)

        weight_size = len(weights)
        size1 = math.floor(weight_size / 2)
        size2 = weight_size - size1 - 1

        np_weights = np.asarray(weights, dtype=np.float32)

        ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = ssim_setup(self.FRAME_WIDTH, self.FRAME_HEIGHT, order='C')
        ux, uy, uxx, uyy, uxy = ssim_setup(self.FRAME_HEIGHT,self.FRAME_WIDTH,  order='C') # doing this so we can transpose it later, numba requires C major order s.t. when we go in the Y direction, we want to

        vy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), dtype=np.float32, order='C')
        diff = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

        image = img_queue.get()
        rearr = np.concatenate((image[0:size1][::-1], image, image[-size2:][::-1]))
        rearr = rearr.astype(dtype=np.float32, order='C')
        correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)  # , curr_scipy)
        T = uyy_tmp.transpose()
        rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)   
        rearr = np.ascontiguousarray(rearr)
        correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
        
        detected_centroids = []
        
        FRAMES_TO_SAVE_AFTER = 1800
        output_filepath =  f'{fn_start}pre-processed.csv'
        
        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32, order='C')
    
        while not done.is_set():    
#            try:
            logger.info(img_queue.qsize())
            
            image = img_queue.get()
            image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if r.mode_noblur_img is None:
                r.find_mode(frame_counter, image)
            elif gui.contour_overlay:
                if gui.contours_updated or r.mode_updated:
                    print("setting up")
                    contour_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3))
                    for c in gui.cell_contours:
                        contour_mask = cv2.drawContours(contour_mask, [c],
                                                        -1, (255, 255, 255), thickness=cv2.FILLED)
                        
                    mask = cv2.cvtColor(
                        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
                    
                    mode_noblur_img = r.mode_noblur_img.astype(np.float32, copy=False)
                    masked_mode_noblur_img = cv2.bitwise_and(
                        mode_noblur_img, mode_noblur_img, mask=mask)
                    masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float32, copy=False) 
                    
                    rearr = np.concatenate((masked_mode_noblur_img[0:size1][::-1], masked_mode_noblur_img, masked_mode_noblur_img[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uy_tmp)  # , curr_scipy)
                    
                    T = uy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uy)  # , curr_scipy)
                                    
                    inp = masked_mode_noblur_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)  # , curr_scipy)
                    
                    T = uyy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
                    
                    uy_squared = uy * uy
                    vy = cov_norm * (uyy - uy_squared)
                    
                    last_confident_centroid = [[gui.cell_centers[i][j] for j in range(gui.shape_of_rows[i])] for i in range(len(gui.shape_of_rows))]

                    gui.contours_updated = False
                    if r.mode_updated:
                        r.mode_updated = False
                        print(r.mode_noblur_img)
                        dpg.configure_item(gui.status, default_value="Status: Mode Ready")
                
                
                time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))
                
                masked_curr_img = cv2.bitwise_and(
                    image, image, mask=mask)
                masked_curr_img = masked_curr_img.astype(np.float32, copy=False)
                
                # TODO bundle into function
                # TODO rethink how to organize
                rearr = np.concatenate((masked_curr_img[0:size1][::-1], masked_curr_img, masked_curr_img[-size2:][::-1]))
                correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, ux_tmp)
                
                T = ux_tmp.transpose()
                rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)

                correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, ux) 
            
                inp = masked_curr_img * masked_curr_img
                rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxx_tmp) 
                
                T = uxx_tmp.transpose()
                rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxx)
                            
                if gui.contour_definer.going_to_mode_method:
                    rearr = np.concatenate((masked_mode_noblur_img[0:size1][::-1], masked_mode_noblur_img, masked_mode_noblur_img[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uy_tmp)  # , curr_scipy)
                    
                    T = uy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uy)  # , curr_scipy)
                                    
                    inp = masked_mode_noblur_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)  # , curr_scipy)
                    
                    T = uyy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
                    
                    gui.contour_definer.going_to_mode_method = False
                    
                if "Prev2Curr" in gui.contour_definer.cv_method:
                    inp = masked_curr_img * prev_masked_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxy_tmp)
                    
                    T = uxy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxy)
    
                    prev_masked_img = masked_curr_img
    
                elif "Mode" in gui.contour_definer.cv_method:
                    inp = masked_curr_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxy_tmp)
                    
                    T = uxy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxy)                    
                
                S = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)

                diff = S.transpose()
                diff = normalize_diff(diff, self.FRAME_WIDTH, self.FRAME_HEIGHT)
                
                if "Prev2Curr" in gui.contour_definer.cv_method:
                    uy = ux.copy()
                    uy_squared = uy * uy
                    vy = cov_norm * (uxx - uy_squared)
                    
                thresh_img = cv2.threshold(diff, gui.contour_definer.thresh, 255, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                sorted_contours = sort_contours_by_area(contours, last_confident_centroid, frame_counter, time_, diff, mask, gui.shape_of_rows, gui.cell_contours, gui.cell_centers, gui.contour_definer.centroid_size)
                
                detected_centroids.extend(sorted_contours)

                if "Structural Similarity" in gui.contour_definer.cv_method:
                    for time, frame_count, row, col, x, y in sorted_contours:
                        cv2.circle(image_data, (x, y), 1, (0, 0, 255), 5, cv2.LINE_4)
                    
                if frame_counter % FRAMES_TO_SAVE_AFTER == 0 and len(detected_centroids) > 0:
                    self.save_centroids_to_csv(output_filepath, detected_centroids)
                    detected_centroids.clear()
            
            data = np.flip(image_data, 2)
            data = data.ravel()
            data = np.asarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
            
            frame_counter += 1
            
            if "Structural Similarity" in gui.contour_definer.cv_method or gui.contour_definer.cv_method == "":
                dpg.set_value("texture_tag", texture_data)
            elif "Diff" in gui.contour_definer.cv_method:
                diff_data = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                diff_data = np.flip(diff_data, 2)
                diff_data = diff_data.ravel()
                diff_data = np.asfarray(diff_data, dtype='f')
                new = np.true_divide(diff_data, 255.0)
                
                dpg.set_value("texture_tag", new)

            dpg.render_dearpygui_frame()
            
            if gui.start_recording:
                start_recording.set()
            

            #except Exception as e:
                #print(e)
            #    logger.error(f"gui failed because: {e}")
            #    logger.error(img_queue.qsize())
            #    logger.error(gc.get_stats())

        dpg.destroy_context()

#    @profile
    def printer_pool(self, done, start_recording, fps_commands, recording_commands):
        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(f"{fn_start}scheduled-events", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)
        logger.info("start printer pool")
        while counter < num_of_instructions and not done.is_set():
            try:
                #TODO decide about start_recording
                #if start_recording.is_set():
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    # print(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    else:  # long video
                        duration = 1800

                    j = [3600, 60, 1]
                    curr_time = timer.formatted_time(timer.now())
                    diff = sum([at_time[i] * j[i] for i in range(len(at_time))]) - sum(
                        [curr_time[i] * j[i] for i in range(len(at_time))])

                    if abs(diff) > 120:
                        logger.info("sending val to fps")
                        # print("sending val to fps")
                        fps_commands.send(fps)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False

                if (timer.formatted_time(timer.now()) == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        timer.formatted_time(timer.now()) >= at_time and type_of_video == 2):
                    if end_time == np.inf:
                        if duration != 0:
                            if type_of_video == 1:
                                logger.info("sending 285.0")
                                # print("sending 285.0")
                                fps_commands.send(285.0)
                            else:
                                logger.info(f"sending {fps}")
                                # print(f"sending {val}")
                                fps_commands.send(fps)

                        start_time = int(timer.now())
                        end_time = start_time + duration
                        recording_commands.send(["start_now", start_time, "end_now", end_time])
                        dev.write(bytes(command_string, 'utf-8'))

                if timer.now() >= end_time:
                    # recording_commands.send(["stop"])
                    counter += 1
                    first_time = True
                    end_time = np.inf
                    
            except Exception as e:
                #print(e)
                logger.error(f"timer failed because: {e}")
                logger.error(gc.get_stats())

        done.set()
            
if __name__ == '__main__':   
    poolrun = GUIPoolRun()

    print('Acquiring images...')
    
    pr = cProfile.Profile()
    pr.enable()
    
    queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    start_recording = multiprocessing.Event()
    fps_commands_vid, fps_commands_p = multiprocessing.Pipe()
    recording_commands_gui, recording_commands_p = multiprocessing.Pipe()

    vid_p = Process(target=poolrun.video_pool, args=(queue, done, fps_commands_vid, recording_queue,))
    tracking_p = Process(target=poolrun.tracking_pool, args=(queue, done, start_recording, ))
    p = Process(target=poolrun.printer_pool, args=(done, start_recording, fps_commands_p, recording_commands_p, ))
    vid_rec_p = Process(target=poolrun.video_recorder_pool, args=(recording_queue, recording_commands_gui, done, ))
    vid_p.start()    
    tracking_p.start()
    p.start()
    vid_rec_p.start()
    vid_p.join()
    tracking_p.join()
    p.join()
    vid_rec_p.join()
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())
