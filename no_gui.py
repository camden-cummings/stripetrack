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
from tracker.roi_manip import convert_to_contours

from structural_sim_from_scratch import correlate1d_x, correlate1d_y, correlate1d, run_math, setup as ssim_setup
from scipy.ndimage import correlate1d as correlate1d_scipy
from skimage.metrics import structural_similarity

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
#def f8_alt(x):
#    return "%14.9f" % x
#pstats.f8 = f8_alt   

class PoolRun:
    def __init__(self):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 992, 660
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

                
                if img_queue.qsize() > 10:
                    time = timer.now()
                    print(img_queue.qsize(), time)
                    while img_queue.qsize() > 2:
                        try:
                            img_queue.get()
                        except Exception as e:
                            print(e)
                            break
                
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
        #with open(f"{fn_start}\\zebrafish-tracker-full-restart.cells", 'rb') as filename:
        #    cell_contours = pickle.load(filename)
        cell_contours, contour_mask, cell_centers, shape_of_rows = convert_to_contours(f"{fn_start}\\zebrafish-tracker-full-restart.cells", self.FRAME_WIDTH, self.FRAME_HEIGHT)

        r = RunCV(self.FRAME_WIDTH, self.FRAME_HEIGHT, f'{fn_start}pre-processed.csv', cell_contours, shape_of_rows)
           
        # just for testing
        image = img_queue.get()
        r.mode_noblur_img=image
        # --------------------------------------
        frame_counter = 0

        setup = False
        
        sigma = 1.5
        truncate = 3.5
        radius = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * radius + 1
        ndim = 2
        NP = win_size ** ndim
        cov_norm = NP / (NP - 1)  # sample covariance
        
        data_range = 255
        
        C1 = (0.01 * data_range) ** 2 #K = 0.01
        C2 = (0.03 * data_range) ** 2 #K = 0.03

        ux, uy, uxx, uyy, uxy = ssim_setup(self.FRAME_WIDTH,self.FRAME_HEIGHT) # doing this so we can transpose it later, numba requires C major order s.t. when we go in the Y direction, we want to
        
        weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
                   0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]        
        
        np_weights = np.asarray(weights)

        # instead treat it like going in the X direction instead
        ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = ssim_setup(self.FRAME_WIDTH, self.FRAME_HEIGHT)
        
        diff = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

        correlate1d_x(image, weights, uyy_tmp)  # , curr_scipy)
        correlate1d_y(uyy_tmp, weights, uyy)  # , curr_scipy)
        
        #pr = cProfile.Profile()
        #pr.enable()
        while not done.is_set():
            try:
                print(img_queue.qsize())

                image = img_queue.get()
                image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                if cv2.waitKey(1) == 1:
                    break

                r.curr_img = image
                r.curr_img_data = image_data

                if r.mode_noblur_img is None:
                    r.find_mode(frame_counter)
                elif not setup:
                    r.mask = contour_mask#.astype(np.float64, copy=False)
                    r.mode_noblur_img = r.mode_noblur_img.astype(np.float64, copy=False)
                    print(r.mask.shape, r.mode_noblur_img.shape)
                    masked_mode_noblur_img = cv2.bitwise_and(
                        r.mode_noblur_img, r.mode_noblur_img, mask=r.mask)
                    print(r.mode_noblur_img.dtype)
                    r.masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float64, copy=False)
                    masked_mode_noblur_img = r.masked_mode_noblur_img
                    setup = True
                    
                    # we don't have to do this every time - will remain constant
                    #correlate1d_x(masked_mode_noblur_img, weights, uy_tmp)  # , curr_scipy)
                    correlate1d_x(masked_mode_noblur_img, weights, uy_tmp)  # , curr_scipy)
                    correlate1d_y(uy_tmp, weights, uy)  # , curr_scipy)
                    
                    correlate1d_x(masked_mode_noblur_img * masked_mode_noblur_img, weights, uyy_tmp)  # , curr_scipy)
                    correlate1d_y(uyy_tmp, weights, uyy)  # , curr_scipy)
                    uy_squared = uy * uy
                    vy = cov_norm * (uyy - uy_squared)

                else:
                    #pr = cProfile.Profile()
                    #pr.enable()
                    
                    time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))

                    print(timer.now())

                    #image = image.astype(np.float64, copy=False)
                    masked_curr_img = cv2.bitwise_and(
                        image, image, mask=contour_mask)
                    masked_curr_img = masked_curr_img.astype(np.float64, copy=False)


                    """
                    correlate1d_x(masked_curr_img, np_weights, ux_tmp)  # , curr_scipy)
                    correlate1d_y(ux_tmp, np_weights, ux)  # , curr_scipy)
                    
                    correlate1d_x(masked_curr_img * masked_curr_img, np_weights, uxx_tmp)  # , curr_scipy)
                    correlate1d_y(uxx_tmp, np_weights, uxx)  # , curr_scipy)
                    
                    correlate1d_x(masked_mode_noblur_img * masked_curr_img, np_weights, uxy_tmp)  # , curr_scipy)
                    correlate1d_y(uxy_tmp, np_weights, uxy)  # , curr_scipy)
                    """
                
                    cv2.imshow('mode', masked_mode_noblur_img)
                    cv2.imshow('curr', masked_curr_img)
                   # print(mode_noblur_img.shape, mode_noblur_img.dtype, image.shape, image.dtype)

                    #correlate1d_x(masked_curr_img, weights, ux_tmp)  # , curr_scipy)
                    correlate1d_x(masked_curr_img, weights, ux_tmp)  # , curr_scipy)
                    correlate1d_y(ux_tmp, weights, ux)  # , curr_scipy)

                    #correlate1d_x(masked_curr_img * masked_curr_img, weights, uxx_tmp)  # , curr_scipy)
                    correlate1d_x(masked_curr_img * masked_curr_img, weights, uxx_tmp)  # , curr_scipy)
                    correlate1d_y(uxx_tmp, weights, uxx)  # , curr_scipy)
                    
                    #correlate1d_x(r.mode_noblur_img * masked_curr_img, weights, uxy_tmp)  # , curr_scipy)
                    correlate1d_x(masked_curr_img * masked_mode_noblur_img, weights, uxy_tmp)  # , curr_scipy)
                    #print(correlate1d_scipy(mode_noblur_img * image, np_weights, axis=0), uxy_tmp)

                    correlate1d_y(uxy_tmp, weights, uxy)  # , curr_scipy)
                    
                    #print(ux, uy, uxx, uyy, uxy, )
                    S = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)

                    xs, ys = np.where(r.mask == 0)
                    
                    for i in range(len(xs)):
                        if masked_mode_noblur_img[xs[i]][ys[i]] != 0.0 or masked_curr_img[xs[i]][ys[i]] != 0.0:
                            print(masked_mode_noblur_img[xs[i]][ys[i]], masked_curr_img[xs[i]][ys[i]])
                    #diff = diff.transpose()
            #        _, str_sim_diff = structural_similarity(curr_img_gray, self.masked_mode_noblur_img, full=True)
                    
                    #cv2.imshow('curr in find', curr_img_gray)
                    diff = (S * 255).astype("uint8")
            
                    cv2.imshow('diff', diff)

                    thresh_img = cv2.threshold(diff, 155, 255, cv2.THRESH_BINARY)[1]
                    contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                
                #    cv2.drawContours(diff, contours, -1, (0,255,0), 1)
                    #contours = [c for c in contours if self.gui.contour_definer.centroid_size < cv2.contourArea(c) < 500000]
                    contours = [c for c in contours if 70 < cv2.contourArea(c) < 500000]
                    
                    #r.run_CV(frame_counter, time_)

                    #cv2.imshow('im', r.curr_img_data)

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


                dpg.set_value("texture_tag", texture_data)
                dpg.render_dearpygui_frame()

                if gui.start_recording:
                    start_recording.set()
                """

                frame_counter += 1
                #print(timer.now())
        
                #pr.disable()
                #s = io.StringIO()
                #sortby = SortKey.CUMULATIVE
                #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                #ps.print_stats()
                #print(s.getvalue())

            except Exception as e:
                print(e)
                #logger.error(f"gui failed because: {e}")
                #logger.error(gc.get_stats())
            
        """
        print(timer.now())

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        """
                    #
        #dpg.destroy_context()


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