import gc
import os
import keyboard
import serial
import logging

from pathlib import PurePath

import PySpin
import cv2
import numpy as np
import pandas as pd

from .camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image, reset_settings
from .command_reader import process_command_string
from .precise_time import PreciseTime
from .config import TEENSY_PORT, CAMERA_NUM, FRAMES_TO_SAVE_AFTER, HIGH_SPEED_MOVIE_FPS, FPS, DEFAULT_DIRECTORY

from pathlib import PurePath
import shutil

import argparse

import datetime


class PoolRun:
    def __init__(self, exp_folder, event_schedule, folder_name, debug, frame_width, \
                 frame_height):
        
        self.FRAME_WIDTH, self.FRAME_HEIGHT = frame_width, frame_height
        self.FRAMES_TO_SAVE_AFTER = FRAMES_TO_SAVE_AFTER
        self.HIGH_SPEED_MOVIE_FPS = HIGH_SPEED_MOVIE_FPS
        self.FPS = FPS

        self.exp_folder = exp_folder
        self.event_schedule = event_schedule
        self.folder_name = folder_name
        self.debug = debug

    def video_pool(self, img_queue, done, fps_commands, recording_queue):
        logging.basicConfig(filename=(PurePath(f'{self.exp_folder}/run.log')), encoding='utf-8')
        logger = logging.getLogger('tracker_log')

        logger.info("start video pool")

        try:
            timer = PreciseTime()

            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            cam_list = setup(system)

            cam = cam_list[CAMERA_NUM]

            nodemap, nodemap_tldevice = setup_nodemap(cam)

            set_node_acquisition_mode(nodemap)
            
            reset_settings(nodemap)

            node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            node_fps.SetValue(self.FPS)
        
            width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            
            try:
                width.SetValue(self.FRAME_WIDTH)
                height.SetValue(self.FRAME_HEIGHT)
            except Exception as e:
                print(f'Failed to set width and height to the values given, {self.FRAME_WIDTH}, {self.FRAME_HEIGHT}, with following error:')    
                print(e)

            cam.BeginAcquisition()

            count = 1
            vidcount = 0
            count_mod = 9
            while not done.is_set():
                if fps_commands.poll():
                    fps = float(fps_commands.recv())
                    node_fps.SetValue(fps)

                image = get_image(cam)

                fps = node_fps.GetValue()
                if fps == self.FPS or fps == self.HIGH_SPEED_MOVIE_FPS:
                    time = timer.now()
                    recording_queue.put([image, time])
                    vidcount += 1

                if img_queue.qsize() > 11:
                    print(img_queue.qsize())
                    while img_queue.qsize() > 2:
                        try:
                            img_queue.get()
                        except Exception as e:
                            logger.info(e)
                            break

                if fps == self.HIGH_SPEED_MOVIE_FPS:
                    if count == count_mod:
                        img_queue.put(image)

                        if count_mod == 9:
                            count_mod = 10
                            count = 1
                        else:
                            count_mod = 9
                            count = 1
                    count += 1
                elif fps == self.FPS:
                    img_queue.put(image)
                    count = 1

                if keyboard.is_pressed('q'):
                    done.set()

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except Exception as ex:
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


    def video_recorder_pool(self, recording_queue, recording_commands, done):
        logging.basicConfig(filename=(PurePath(f'{self.exp_folder}/run.log')), encoding='utf-8', level=logging.DEBUG)
        logger = logging.getLogger('tracker_log')

        start_time = -1
        end_time = -1
        result = None
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

                            vid_name = f"{int(at_time / 3600)}_{int((at_time % 3600) / 60)}_{int(at_time % 60)}-{str(counter)}"
                            if type_of_video == 1:
                                result = cv2.VideoWriter(str(PurePath(f'{self.exp_folder}/{vid_name}.avi')),
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         self.HIGH_SPEED_MOVIE_FPS, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)
                            else:
                                result = cv2.VideoWriter(str(PurePath(f'{self.exp_folder}/{vid_name}-long.avi')),
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         self.FPS, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)

                if start_time != -1 and start_time < time < end_time:  # and vidcount <= self.HIGH_SPEED_MOVIE_FPS:
                    result.write(image)

            except Exception as e:
                logger.error(f"video recorder failed at: {e}")
                logger.error(gc.get_stats())


    @staticmethod
    def save_centroids_to_csv(output_filepath, detected_centroids):
        if not os.path.exists(output_filepath):
            new = pd.DataFrame(np.matrix(detected_centroids),
                               columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y', 'dpix'])
            new.to_csv(output_filepath, sep=',', index=False)
        else:
            new = pd.DataFrame(np.matrix(detected_centroids),
                               columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y', 'dpix'])
            new.to_csv(output_filepath, sep=',', mode='a', index=False, header=False)


    def printer_pool(self, done, fps_commands, recording_commands):
        logging.basicConfig(filename=str(PurePath(f'{self.exp_folder}/run.log')), encoding='utf-8', level=logging.DEBUG)
        logger = logging.getLogger('tracker_log')

        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(str(PurePath(f"{self.exp_folder}/{self.event_schedule}")), sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        try:
            dev = serial.Serial(port=f'COM{TEENSY_PORT}', baudrate=115200, timeout=.1)
        except serial.SerialException as e:
            print(f'Could not open Teensy port. Check value of TEENSY_PORT: {TEENSY_PORT} in live_tracker/config and that Arduino IDE is closed. Quitting...')
            done.set()
            
        while counter < num_of_instructions and not done.is_set():
            try:
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    else:  # long video
                        duration = 1800
                    
                    diff = at_time - int(timer.now()) % 86400

                    if abs(diff) > 120:
                        logger.info("sending val to fps")
                        fps_commands.send(self.FPS)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False
                    sent = False

                curr_time = int(timer.now()) % 86400
                if (curr_time == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        curr_time >= at_time and type_of_video == 2):

                    if end_time == np.inf:
                        start_time = int(timer.now())
                        end_time = start_time + duration
                        recording_commands.send(["start_now", start_time, "end_now", end_time])

                        if duration != 0:
                            if type_of_video == 1:
                                logger.info(f"sending {self.HIGH_SPEED_MOVIE_FPS}")
                                fps_commands.send(self.HIGH_SPEED_MOVIE_FPS)
                            else:
                                logger.info(f"sending {self.FPS}")
                                fps_commands.send(self.FPS)

                        dev.write(bytes(command_string, 'utf-8'))

                if type_of_video == 1 and not sent:  # HIGH_SPEED_MOVIE_FPS fps vids need to be at HIGH_SPEED_MOVIE_FPS fps when command to record starts
                    diff = at_time - curr_time

                    if diff == 5:
                        fps_commands.send(self.HIGH_SPEED_MOVIE_FPS)
                        sent = True

                if timer.now() >= end_time:
                    counter += 1
                    first_time = True
                    end_time = np.inf

            except Exception as e:
                logger.error(f"timer failed because: {e}")
                logger.error(gc.get_stats())

        done.set()
        

def check_directory(exp_folder, event_schedule):
    if len(exp_folder) == 0:
        curr_day = datetime.datetime.now()
        name = f"{curr_day.month}-{curr_day.day}-{curr_day.year}" 
        print(f'Making experiment directory with {name} as name & copying event-schedule from standard_stimulus_files')
        exp_folder = PurePath(f"{DEFAULT_DIRECTORY}/{name}")
        event_schedule = 'scheduled-events'

    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    
    if not os.path.exists(f'{exp_folder}/{event_schedule}'):
        shutil.copy2(str(PurePath(f'standard_stimulus_files/{event_schedule}')), str(PurePath(f'{exp_folder}/{event_schedule}')))

    return exp_folder, event_schedule, name
