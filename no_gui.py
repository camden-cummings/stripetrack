from multiprocessing import Process
import multiprocessing
import cProfile, pstats, io
from pstats import SortKey

import PySpin
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from AcquireAndDisplayClass import get_image
from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode
#from gui_helpers import GUIHelpers

from precise_time import PreciseTime
#from tracker.str_sim_run import shape_of_rows

from vid import RunCV, process_command_string
#import queue

import gc

import pandas as pd

import keyboard
import serial

import pickle

fn_start = "C:\\Users\\ThymeLab\\Desktop\\5-6-25\\"

#import logging

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

#logger = logging.getLogger(__name__)
#logging.basicConfig(filename=f'{fn_start}run.log', encoding='utf-8', level=logging.DEBUG)

val = 30.0


class PoolRun:
    def __init__(self):
        self.FRAME_HEIGHT, self.FRAME_WIDTH = 660, 992
        self.image_data = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT, 3))

    # @profile
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

            # width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            # height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))

            # min_width = PySpin.CIntegerPtr(nodemap.GetNode("MaxWidth"))
            # print("width ",width.GetValue(), ", height ", height.GetValue())
            # width.SetValue(32*5)
            # height.SetValue(100)
            cam.BeginAcquisition()

            recording = False
            while not done.is_set():
                if fps_commands.poll():
                    fps = float(fps_commands.recv())
                    node_fps.SetValue(fps)

                image = get_image(cam)

                if node_fps.GetValue() == val or node_fps.GetValue() == 285.0:
                    time = timer.now()
                    recording_queue.put([image, time])

                """
                if img_queue.qsize() > 10:
                    time = timer.now()
                    print(img_queue.qsize(), time)
                    while img_queue.qsize() > 2:
                        try:
                            img_queue.get()
                        except Exception as e:
                            logger.error(e)
                            break
                """
                img_queue.put(image)
                if keyboard.is_pressed('q'):
                    done.set()

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except Exception as ex:
            print(ex)
            # logger.error(f'Error: {ex}')
            # logger.error(gc.get_stats())

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
        cmd_list = []
        start_time = -1
        end_time = -1
        # counter = 0

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

                            # counter = 0

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
                    # counter += 1
            #                    logger.info(str(counter))

            except Exception as e:
                print(e)
                #logger.error(f"video recorder failed at: {e}")
                #logger.error(gc.get_stats())

    #  @profile

    def gui_pool(self, img_queue, done, start_recording):
        #logger.info("start gui pool")
        #dpg.create_context()
        #window = dpg.add_window(label="Video player", pos=(50, 50), width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT)
        #gui = GUIHelpers(window, self.FRAME_WIDTH, self.FRAME_HEIGHT)
        #gui.start()
        #recording = False
        timer = PreciseTime()

        #dpg.set_primary_window(window, True)
        #dpg.create_viewport(width=int(self.FRAME_WIDTH*1.5), height=self.FRAME_HEIGHT+20, title="ROI Selector")
        #dpg.setup_dearpygui()
        #dpg.show_viewport()
        with open(f"{fn_start}\\zebrafish-tracker-full-restart.cells", 'rb') as filename:
            cell_contours = pickle.load(filename)
        
        shape_of_rows = []
        r = RunCV(self.FRAME_WIDTH, self.FRAME_HEIGHT, f'{fn_start}pre-processed.csv', cell_contours, shape_of_rows)
        frame_counter = 0

        ux = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        uy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        uxx = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        uyy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        uxy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        setup = False
        
        while not done.is_set():
            try:
                print(img_queue.qsize())

                image = img_queue.get()
                image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                #cv2.imshow('im', image_data)
                #f cv2.waitKey(1) == 1:
                #    break
                r.curr_img = image
                r.curr_img_data = image_data

                if r.mode_noblur_img is None:
                    r.find_mode(frame_counter)
                elif not setup:
                    contour_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3))
                    for c in cell_contours:
                        non_int = np.array(c)
                        contour_mask = cv2.drawContours(contour_mask, [non_int],
                                                        -1, (255, 255, 255), thickness=cv2.FILLED)
                        
                    r.mask = cv2.cvtColor(
                        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

                    masked_mode_noblur_img = cv2.bitwise_and(
                        r.mode_noblur_img, r.mode_noblur_img, mask=r.mask)
                    masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float64, copy=False)
                    r.masked_mode_noblur_img = masked_mode_noblur_img
                    
                    setup = True
                else:
                    time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))
                    pr = cProfile.Profile()
                    pr.enable()

                    r.run_CV(frame_counter, time_, ux, uy, uxx, uyy, uxy)

                    pr.disable()
                    s = io.StringIO()
                    sortby = SortKey.CUMULATIVE
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    print(s.getvalue())
                    
                """
                elif gui.contour_overlay:
                    if gui.contours_updated:
                        contour_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3))
                        for c in gui.rt_tracker.cell_contours:
                            contour_mask = cv2.drawContours(contour_mask, [c],
                                                            -1, (255, 255, 255), thickness=cv2.FILLED)

                        r.mask = cv2.cvtColor(
                            np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

                        masked_mode_noblur_img = cv2.bitwise_and(
                            r.mode_noblur_img, r.mode_noblur_img, mask=r.mask)
                        masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float64, copy=False)
                        r.masked_mode_noblur_img = masked_mode_noblur_img

                        gui.contours_updated = False

                    time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))
                    r.run_CV(frame_counter, time_, ux, uy, uxx, uyy, uxy)

                data = np.flip(image_data, 2)
                data = data.ravel()
                data = np.asfarray(data, dtype='f')
                texture_data = np.true_divide(data, 255.0)

                frame_counter += 1

                dpg.set_value("texture_tag", texture_data)
                dpg.render_dearpygui_frame()

                if gui.start_recording:
                    start_recording.set()
                """



            except Exception as e:
                print(e)
                #logger.error(f"gui failed because: {e}")
                #logger.error(gc.get_stats())

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

        while counter < num_of_instructions and not done.is_set():
            try:
            #if start_recording.is_set():
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    #logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    print(f"COMMANDS: {at_time} {command_string} {type_of_video}")
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
                        #logger.info("sending val to fps")
                        print("sending val to fps")
                        fps_commands.send(val)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False

                if (timer.formatted_time(timer.now()) == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        timer.formatted_time(timer.now()) >= at_time and type_of_video == 2):
                    if end_time == np.inf:
                        if duration != 0:
                            if type_of_video == 1:
                                #logger.info("sending 285.0")
                                print("sending 285.0")
                                fps_commands.send(285.0)
                            else:
                                #logger.info(f"sending {val}")
                                print(f"sending {val}")
                                fps_commands.send(val)

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
                # print(e)
                #logger.error(f"timer failed because: {e}")
                #logger.error(gc.get_stats())
                print(f"timer failed because: {e}")
                print(gc.get_stats())

        done.set()


if __name__ == '__main__':
    poolrun = PoolRun()

    print('Acquiring images...')

    #    pr = cProfile.Profile()
    #    pr.enable()

    queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    start_recording = multiprocessing.Event()
    fps_commands_vid, fps_commands_p = multiprocessing.Pipe()
    recording_commands_gui, recording_commands_p = multiprocessing.Pipe()

    vid_p = Process(target=poolrun.video_pool, args=(queue, done, fps_commands_vid, recording_queue,))
    gui_p = Process(target=poolrun.gui_pool, args=(queue, done, start_recording,))
    p = Process(target=poolrun.printer_pool, args=(done, start_recording, fps_commands_p, recording_commands_p,))
    vid_rec_p = Process(target=poolrun.video_recorder_pool, args=(recording_queue, recording_commands_gui, done,))
    vid_p.start()
    gui_p.start()
    p.start()
    vid_rec_p.start()
    vid_p.join()
    gui_p.join()
    p.join()
    vid_rec_p.join()

#    pr.disable()
#    s = io.StringIO()
#    sortby = SortKey.CUMULATIVE
#    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#    ps.print_stats()
#    print(s.getvalue())

# pool.apply_async(video_pool)
# video_thread = Thread(target=video)
# video_thread.start()