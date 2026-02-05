import gc
import os

import PySpin
import cv2
import keyboard
import numpy as np
import pandas as pd
import serial

from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image
from command_reader import process_command_string
from precise_time import PreciseTime

import logging

class PoolRun:
    def __init__(self, exp_folder, event_schedule, debug):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 992, 660
        self.FRAMES_TO_SAVE_AFTER = 1800
        self.FPS = 30

        self.exp_folder = exp_folder
        self.event_schedule = event_schedule
        self.debug = debug

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{exp_folder}\\run.log', encoding='utf-8', level=logging.DEBUG)


    def video_pool(self, img_queue, done, fps_commands, recording_queue):
        self.logger.info("start video pool")

        try:
            timer = PreciseTime()

            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            cam_list = setup(system)

            # Run example on each camera
            cam = cam_list[0]

            nodemap, nodemap_tldevice = setup_nodemap(cam)

            set_node_acquisition_mode(nodemap)

            node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            node_fps.SetValue(self.FPS)

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
                if fps == self.FPS or fps == 285.0:
                    time = timer.now()
                    recording_queue.put([image, time])
                    vidcount += 1

                if img_queue.qsize() > 11:
                    while img_queue.qsize() > 2:
                        try:
                            img_queue.get()
                        except Exception as e:
                            self.logger.info(e)
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
                elif fps == self.FPS:
                    img_queue.put(image)
                    count = 1

                if keyboard.is_pressed('q'):
                    done.set()

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except Exception as ex:
            self.logger.error(f'Error: {ex}')
            self.logger.error(gc.get_stats())

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
                                result = cv2.VideoWriter(f'{self.exp_folder}{vid_name}.avi',
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         285, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)
                            else:
                                result = cv2.VideoWriter(f'{self.exp_folder}{vid_name}-long.avi',
                                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                                         self.FPS, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)

                if start_time != -1 and start_time < time < end_time:  # and vidcount <= 285:
                    result.write(image)

            except Exception as e:
                self.logger.error(f"video recorder failed at: {e}")
                self.logger.error(gc.get_stats())


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
        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(f"{self.exp_folder}/{self.event_schedule}", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)

        while counter < num_of_instructions and not done.is_set():
            try:
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    self.logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    else:  # long video
                        duration = 1800

                    diff = at_time - int(timer.now() - 3600 * 5) % 86400

                    if abs(diff) > 120:
                        self.logger.info("sending val to fps")
                        fps_commands.send(self.FPS)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False
                    sent = False

                curr_time = int(timer.now() - 3600 * 5) % 86400

                if (curr_time == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        curr_time >= at_time and type_of_video == 2):

                    if end_time == np.inf:
                        start_time = int(timer.now())
                        end_time = start_time + duration
                        recording_commands.send(["start_now", start_time, "end_now", end_time])

                        if duration != 0:
                            if type_of_video == 1:
                                self.logger.info("sending 285.0")
                                fps_commands.send(285.0)
                            else:
                                self.logger.info(f"sending {self.FPS}")
                                fps_commands.send(self.FPS)

                        dev.write(bytes(command_string, 'utf-8'))

                if type_of_video == 1 and not sent:  # 285 fps vids need to be at 285 fps when command to record starts
                    diff = at_time - curr_time

                    if diff == 5:
                        fps_commands.send(285.0)
                        sent = True

                if timer.now() >= end_time:
                    counter += 1
                    first_time = True
                    end_time = np.inf

            except Exception as e:
                self.logger.error(f"timer failed because: {e}")
                self.logger.error(gc.get_stats())

        done.set()