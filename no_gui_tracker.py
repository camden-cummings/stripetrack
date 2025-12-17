import argparse
import cProfile
import gc
import io
import logging
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

from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image
from command_reader import process_command_string
from mode_finder import ModeFinder
from precise_time import PreciseTime
from roi_manip import convert_to_contours
from sort_contours_by_area import SortContours
from strsim_for_speed.computer_vision.speedy_str_sim_as_a_class import SpeedyCV
from strsim_for_speed.computer_vision.structural_sim_from_scratch import run_math, run_math_complete, normalize_diff
from arg_helpers import setup_args, get_args

class PoolRun:
    def __init__(self, exp_folder, rois_fname, event_schedule, debug, mode):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 992, 660
        # self.FRAME_WIDTH, self.FRAME_HEIGHT = 1760, 1200
        self.FRAMES_TO_SAVE_AFTER = 1800
        self.FPS = 30

        self.exp_folder = exp_folder
        self.event_schedule = event_schedule
        self.rois_fname = rois_fname
        self.mode = mode


    # TODO this might be able to be static
    def video_pool(self, img_queue, done, fps_commands, recording_queue):
        logger.info("start video pool")
        # pr = cProfile.Profile()
        # pr.enable()
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
                    # print("qsize", img_queue.qsize(), fps)
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
                elif fps == self.FPS:
                    img_queue.put(image)
                    count = 1

                if keyboard.is_pressed('q'):
                    done.set()

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except Exception as ex:
            # print(ex)
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

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print("video pool", s.getvalue())

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

                            #                            if result is not None:
                            #                                result.release()

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
                logger.error(f"video recorder failed at: {e}")
                logger.error(gc.get_stats())

    def tracking_pool(self, img_queue, done):
        logger.info("start tracking pool")

        timer = PreciseTime()

        cell_contours, contour_mask, cell_centers, shape_of_rows = convert_to_contours(f"{self.exp_folder}{self.rois_fname}",
                                                                                       self.FRAME_WIDTH,
                                                                                       self.FRAME_HEIGHT)

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)

        sc = SortContours(contour_mask, shape_of_rows, cell_contours, cell_centers)
        spd = SpeedyCV(self.FRAME_HEIGHT, self.FRAME_WIDTH)

        frame_counter = 0

        vy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C')

        detected_centroids = []

        output_filepath = f'{self.exp_folder}\\pre-processed.csv'

        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32)

        while not done.is_set():
            try:
                # pr = cProfile.Profile()
                # pr.enable()

                image = img_queue.get()

                arr_time = str(timer.formatted_time(timer.now())).strip("[]").split(", ")
                time = "_".join(arr_time)

                # print(time, img_queue.qsize())

                if r.mode_noblur_img is None:
                    r.find_mode(frame_counter, image)
                else:
                    if r.mode_updated:
                        print('setting up')

                        if self.mode:
                            mode_noblur_img = r.mode_noblur_img.astype(np.float32, copy=False)
                            masked_mode_noblur_img = cv2.bitwise_and(
                                mode_noblur_img, mode_noblur_img, mask=contour_mask)
                            masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float32, copy=False)

                            spd.run_mode(masked_mode_noblur_img)

                        r.mode_updated = False

                    masked_curr_img = cv2.bitwise_and(image, image, mask=contour_mask)
                    masked_curr_img = masked_curr_img.astype(np.float32, copy=False)

                    spd.run_corr(masked_curr_img)

                    if self.mode:
                        spd.run_against(masked_mode_noblur_img, masked_curr_img)
                        S = run_math_complete(spd.cov_norm, spd.data_range, spd.ux, spd.uy, spd.uxx, spd.uyy, spd.uxy)
                    else:
                        spd.run_against(prev_masked_img, masked_curr_img)
                        S = run_math(spd.cov_norm, spd.data_range, spd.ux, spd.uy, spd.uxx, vy, spd.uxy)

                    S_t = S.T
                    normalize_diff(S_t, self.FRAME_WIDTH, self.FRAME_HEIGHT, spd.out)

                    if not self.mode:
                        prev_masked_img = masked_curr_img

                        uy = spd.ux.copy()
                        spd.uy = uy
                        uy_squared = np.multiply(uy, uy)
                        vy = spd.cov_norm * (spd.uxx - uy_squared)

                    thresh_img = cv2.threshold(spd.out, 155, 255, cv2.THRESH_BINARY)[1]
                    contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]

                    sorted_contours = sc.sort_contours_by_area(contours, frame_counter, time, spd.out)

                    detected_centroids.extend(sorted_contours)

                    if view:
                        for time, frame_count, row, col, x, y in sorted_contours:
                            cv2.circle(image, (x, y), 1, (0, 0, 255), 5, cv2.LINE_4)

                        cv2.imshow('frame', image)
                        cv2.imshow('diff', spd.out)

                        if cv2.waitKey(1) == 1:
                            break

                    if frame_counter % self.FRAMES_TO_SAVE_AFTER == 0 and len(detected_centroids) > 0:
                        self.save_centroids_to_csv(output_filepath, detected_centroids)
                        detected_centroids.clear()

                frame_counter += 1

                # pr.disable()
                # s = io.StringIO()
                # sortby = SortKey.CUMULATIVE
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print("tracker pool", s.getvalue())

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
    def printer_pool(self, done, fps_commands, recording_commands):
        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(f"{self.exp_folder}\{self.event_schedule}", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)

        pr = cProfile.Profile()
        pr.enable()
        while counter < num_of_instructions and not done.is_set():
            try:
                if first_time:  # setup all necessary pieces
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    logger.info(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    print(f"COMMANDS: {at_time} {command_string} {type_of_video}")
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    else:  # long video
                        duration = 1800

                    diff = at_time - int(timer.now() % 86400 - 3600 * 5)

                    if abs(diff) > 120:
                        logger.info("sending val to fps")
                        print(f"sending {self.FPS} to fps")
                        fps_commands.send(self.FPS)

                    recording_commands.send([duration, at_time, type_of_video, counter])

                    first_time = False
                    sent = False
                    last_time = 0
                    last_time_2 = 0

                curr_time = int(timer.now() % 86400 - 3600 * 5)
                # print(curr_time, at_time)
                #                if curr_time == at_time:
                #                    print("!")

                if last_time_2 != curr_time:
                    current_time = timer.now()
                    logger.info(
                        f"h {curr_time}, {at_time}, {end_time}, {type_of_video}, {type_of_video == 2} {current_time} {current_time % 86400}")
                last_time_2 = curr_time

                if (curr_time == at_time and (
                        type_of_video == 1 or type_of_video == 0)) or (
                        curr_time >= at_time and type_of_video == 2):

                    if last_time != curr_time:
                        logger.info(f"h {curr_time}, {at_time}, {end_time}")
                    last_time = curr_time
                    if end_time == np.inf:
                        start_time = int(timer.now())
                        end_time = start_time + duration
                        recording_commands.send(["start_now", start_time, "end_now", end_time])
                        logger.info(f"! {start_time}, {end_time}")
                        if duration != 0:
                            if type_of_video == 1:
                                logger.info("sending 285.0")
                                print("sending 285.0")
                                fps_commands.send(285.0)
                            else:
                                logger.info(f"sending {self.FPS}")
                                print(f"sending {self.FPS}")
                                fps_commands.send(self.FPS)

                        dev.write(bytes(command_string, 'utf-8'))

                if type_of_video == 1 and not sent:  # 285 fps vids need to be at 285 fps when command to record starts
                    diff = at_time - curr_time

                    if diff == 5:
                        print("SENDING")
                        fps_commands.send(285.0)
                        sent = True

                # logger.info(f"{timer.now()}, {end_time}")
                if timer.now() >= end_time:
                    print('incrementing timer, setting first time to true, setting end time to np.inf')
                    counter += 1
                    first_time = True
                    end_time = np.inf

            except Exception as e:
                logger.error(f"timer failed because: {e}")
                logger.error(gc.get_stats())
                # print(f"timer failed because: {e}")
                # print(gc.get_stats())

        done.set()

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print("printer pool", s.getvalue())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--view",
        action='store_true'
    )

    setup_args(parser)
    args = parser.parse_args()
    exp_folder, rois_fname, event_schedule, debug, mode = get_args(args)

    view = args.view

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{exp_folder}\\run.log', encoding='utf-8', level=logging.DEBUG)

    poolrun = PoolRun(exp_folder, rois_fname, event_schedule, debug, mode)

    print('Acquiring images...')

    #    pr = cProfile.Profile()
    #    pr.enable()

    queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    fps_commands_vid, fps_commands_p = multiprocessing.Pipe()
    recording_commands, recording_commands_p = multiprocessing.Pipe()

    vid_p = Process(target=poolrun.video_pool, args=(queue, done, fps_commands_vid, recording_queue,))
    tracking_p = Process(target=poolrun.tracking_pool, args=(queue, done,))
    p = Process(target=poolrun.printer_pool, args=(done, fps_commands_p, recording_commands_p,))
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
