import cProfile
import gc
import io
import math
import multiprocessing
import os
import pstats
from multiprocessing import Process
from pstats import SortKey

# TODO add PySpin download instructions to git somewhere
import PySpin
import cv2
import keyboard
import numpy as np
import pandas as pd
import serial

from roi_manip import convert_to_contours
from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image
from command_reader import process_command_string
from mode_finder import ModeFinder
from precise_time import PreciseTime
from sort_contours_by_area import sort_contours_by_area
from strsim_for_speed.computer_vision.structural_sim_from_scratch import correlate1d_x, correlate1d_y, run_math, run_math_complete, normalize_diff, setup as ssim_setup, generate_weights

fn_start = "C:\\Users\\ThymeLab\\Desktop\\6-27-25-test\\"

import logging

# create logger
#mem_logger = logging.getLogger('memory_profile_log')
#mem_logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
#fh = logging.FileHandler("memory_profile.log")
#fh.setLevel(logging.DEBUG)

# create formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)

# add the handlers to the logger
#mem_logger.addHandler(fh)

#from memory_profiler import LogFile
#import sys

#sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)

# from memory_profiler import profile

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{fn_start}run.log', encoding='utf-8', level=logging.DEBUG)

val = 30.0
#def f8_alt(x):
#    return "%14.9f" % x
#pstats.f8 = f8_alt   

class PoolRun:
    def __init__(self):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 1760, 1200

    # @profile
    # TODO this might be able to be static
    def video_pool(self, img_queue, done, fps_commands, recording_queue):
        #    logger.info("start video pool")
        try:
            timer = PreciseTime()

            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            cam_list = setup(system)

            # Run example on each camera
            cam = cam_list[0]

            nodemap, nodemap_tldevice = setup_nodemap(cam)
            #        logger.info('*** IMAGE ACQUISITION ***\n')

            set_node_acquisition_mode(nodemap)

            node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            node_fps.SetValue(val)

            cam.BeginAcquisition()

            count = 1
            count_mod = 9
            while not done.is_set():
                if fps_commands.poll():
                    fps = float(fps_commands.recv())
                    node_fps.SetValue(fps)

                image = get_image(cam)
                
                fps = node_fps.GetValue()
                if fps == val or fps == 285.0:
                    time = timer.now()
                    recording_queue.put([image, time])

                # TODO is this necessary anymore (?)
                if img_queue.qsize() > 10:
                    while img_queue.qsize() > 2:
                        try:
                            img_queue.get()
                        except Exception as e:
                            logger.info(e)
                            break
                
                if fps == 285.0:
                    if count == count_mod:
                        img_queue.put(image)
                        
                        if count_mod == 9:
                            count_mod = 10
                            count = 1
                        else:
                            count_mod = 9
                            count = 1
                    count += 1
                elif fps == val:    
                    img_queue.put(image)
                    count = 1
                if keyboard.is_pressed('q'):
                    done.set()

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except Exception as ex:
            #print(ex)
            logger.error(f'Error: {ex}')
            logger.error(gc.get_stats())

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

    #    @profile
    def video_recorder_pool(self, recording_queue, recording_commands, done):
        start_time = -1
        end_time = -1

        while not done.is_set():
            try:
                image, time = recording_queue.get()

                if time > end_time:
                    if recording_commands.poll():
                        cmds = recording_commands.recv()

                        if cmds[0] == "start_now":
                            start_time = cmds[1]
                            end_time = cmds[3]
                        else:
                            duration, at_time, type_of_video, counter = cmds

                            vid_name = "_".join(str(at_time).strip("[]").split(", ")) + "-" + str(counter)

                            if type_of_video == 1:
                                result = cv2.VideoWriter(f'{fn_start}{vid_name}.avi',
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         285, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)
                            else:
                                result = cv2.VideoWriter(f'{fn_start}{vid_name}-long.avi',
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         val, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)

                if start_time != -1 and start_time < time < end_time:
                    result.write(image)

            except Exception as e:
                logger.error(f"video recorder failed at: {e}")
                logger.error(gc.get_stats())


    def tracking_pool(self, img_queue, done, start_recording):
        logger.info("start tracking pool")
        timer = PreciseTime()

        cell_contours, contour_mask, cell_centers, shape_of_rows = convert_to_contours(f"{fn_start}\\zebrafish-tracker-6-24-25-realrun.cells", self.FRAME_WIDTH, self.FRAME_HEIGHT)

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)
           
        image = img_queue.get()

        frame_counter = 0
        
        data_range = 255

        weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)
        
        weight_size = len(weights)
        size1 = math.floor(weight_size / 2)
        size2 = weight_size - size1 - 1

        np_weights = np.asarray(weights, dtype=np.float32)

        ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = ssim_setup(self.FRAME_WIDTH, self.FRAME_HEIGHT, order='C')
        ux, uy, uxx, uyy, uxy = ssim_setup(self.FRAME_HEIGHT, self.FRAME_WIDTH, order='C') # doing this so we can transpose it later, numba requires C major order s.t. when we go in the Y direction, we want to
        vy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C')
        
        run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
        
        rearr = np.concatenate((image[0:size1][::-1], image, image[-size2:][::-1]))
        rearr = rearr.astype(dtype=np.float32)
        correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)

        T = uyy_tmp.transpose()
        rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
        correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)
        
        detected_centroids = []

        FRAMES_TO_SAVE_AFTER = 1800
        output_filepath =  f'{fn_start}pre-processed.csv'
        
        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32)

        while not done.is_set():
            try:
                image = img_queue.get()

                arr_time = str(timer.formatted_time(timer.now())).strip("[]").split(", ")
                time = "_".join(arr_time)
                logger.info(time)
                logger.info(img_queue.qsize())

                if (0 <= int(arr_time[1]) <= 10 and not r.found_mode) or (r.mode_noblur_img is None):
                    r.find_mode(frame_counter, image)
                elif arr_time[1] == 11:
                    r.found_mode = False

                # TODO make clean way to go between mode method & prev2curr method
                if r.mode_noblur_img is not None:
                    if r.mode_updated:
                        print('setting up')
                        mode_noblur_img = r.mode_noblur_img.astype(np.float32, copy=False)
                        masked_mode_noblur_img = cv2.bitwise_and(
                            mode_noblur_img, mode_noblur_img, mask=contour_mask)
                        masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float32, copy=False)
                        r.mode_updated = False
                        
                        last_confident_centroid = [[cell_centers[i][j] for j in range(shape_of_rows[i])] for i in range(len(shape_of_rows))]

                    pr = cProfile.Profile()
                    pr.enable()
                    
                    masked_curr_img = cv2.bitwise_and(image, image, mask=contour_mask)

                    masked_curr_img = masked_curr_img.astype(np.float32, copy=False)#, order='C')
                    
                    #TODO update
                    """
                    rearr = np.concatenate((prev_masked_img[0:size1][::-1], prev_masked_img, prev_masked_img[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uy_tmp)
                    
                    T = uy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uy)
                    
                    inp = prev_masked_img * prev_masked_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)
                    
                    T = uyy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy) 
                    """
                    
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
                                    
                    inp = masked_curr_img * prev_masked_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxy_tmp) 
                    
                    T = uxy_tmp.transpose()
                    rearr = np.concatenate((T[0:size1][::-1], T, T[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxy)
                    
                    #S = run_math_complete(cov_norm, data_range, ux, uy, uxx, uyy, uxy)    
                    S = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)
                    diff = S.transpose()
                    diff = normalize_diff(diff, self.FRAME_WIDTH, self.FRAME_HEIGHT)
                    
                    #cv2.imshow('frame', diff)
                    #cv2.imshow('prev', prev_masked_img)
                    #cv2.imshow('curr', masked_curr_img)
    
                    #if cv2.waitKey(1) == 1:
                    #    break
                        
                    prev_masked_img = masked_curr_img
                    
                    uy = ux.copy()
                    uy_squared = uy * uy
                    vy = cov_norm * (uxx - uy_squared)
                    
                    thresh_img = cv2.threshold(diff, 155, 255, cv2.THRESH_BINARY)[1]
                    contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]

                    sorted_contours = sort_contours_by_area(contours, last_confident_centroid, frame_counter, time, diff, contour_mask, shape_of_rows, cell_contours, cell_centers, 50)
                    
                    detected_centroids.extend(sorted_contours)
                    
                    if frame_counter % FRAMES_TO_SAVE_AFTER == 0 and len(detected_centroids) > 0:
                        self.save_centroids_to_csv(output_filepath, detected_centroids)
                        detected_centroids.clear()

                    pr.disable()
                    s = io.StringIO()
                    sortby = SortKey.CUMULATIVE
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    print(s.getvalue())

                frame_counter += 1

            except Exception as e:
                logger.error(f"tracking failed because: {e}")
                logger.error(gc.get_stats())
            
    @staticmethod
    def save_centroids_to_csv(output_filepath, detected_centroids):
        if not os.path.exists(output_filepath):
            new = pd.DataFrame(np.matrix(detected_centroids),
                               columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y'])
            new.to_csv(output_filepath, sep=',', index=False)
        else:
            new = pd.DataFrame(np.matrix(detected_centroids),
                               columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y'])
            new.to_csv(output_filepath, sep=',', mode='a', index=False, header=False)

    #    @profile
    def printer_pool(self, done, start_recording, fps_commands, recording_commands):
        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(f"{fn_start}scheduled-events", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)

        while counter < num_of_instructions and not done.is_set():
            try:
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    #print(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    else: # long video
                        duration = 1800

                    j = [3600, 60, 1]
                    curr_time = timer.formatted_time(timer.now())
                    diff = sum([at_time[i] * j[i] for i in range(len(at_time))]) - sum(
                        [curr_time[i] * j[i] for i in range(len(at_time))])

                    if abs(diff) > 120:
                        logger.info("sending val to fps")
                        #print("sending val to fps")
                        fps_commands.send(val)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False

                if (timer.formatted_time(timer.now()) == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        timer.formatted_time(timer.now()) >= at_time and type_of_video == 2):
                    if end_time == np.inf:
                        if duration != 0:
                            if type_of_video == 1:
                                logger.info("sending 285.0")
                                #print("sending 285.0")
                                fps_commands.send(285.0)
                            else:
                                logger.info(f"sending {val}")
                                #print(f"sending {val}")
                                fps_commands.send(val)

                        start_time = int(timer.now())
                        end_time = start_time + duration
                        recording_commands.send(["start_now", start_time, "end_now", end_time])
                        dev.write(bytes(command_string, 'utf-8'))

                if timer.now() >= end_time:
                    counter += 1
                    first_time = True
                    end_time = np.inf

            except Exception as e:
                logger.error(f"timer failed because: {e}")
                logger.error(gc.get_stats())
                #print(f"timer failed because: {e}")
                #print(gc.get_stats())

        done.set()

if __name__ == '__main__':
    poolrun = PoolRun()

    print('Acquiring images...')

    #    pr = cProfile.Profile()
    #    pr.enable()

    #TODO try tossing all into func

    queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    start_recording = multiprocessing.Event()
    fps_commands_vid, fps_commands_p = multiprocessing.Pipe()
    recording_commands, recording_commands_p = multiprocessing.Pipe()

    vid_p = Process(target=poolrun.video_pool, args=(queue, done, fps_commands_vid, recording_queue,))
    tracking_p = Process(target=poolrun.tracking_pool, args=(queue, done, start_recording,))
    p = Process(target=poolrun.printer_pool, args=(done, start_recording, fps_commands_p, recording_commands_p,))
    vid_rec_p = Process(target=poolrun.video_recorder_pool, args=(recording_queue, recording_commands, done,))
    vid_p.start()
    tracking_p.start()
    p.start()
    vid_rec_p.start()
    vid_p.join()
    tracking_p.join()
    p.join()
    vid_rec_p.join()

#    pr.disable()
#    s = io.StringIO()
#    sortby = SortKey.CUMULATIVE
#    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#    ps.print_stats()
#    print(s.getvalue())
