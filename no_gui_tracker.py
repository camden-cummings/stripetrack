import gc
import multiprocessing
from multiprocessing import Process

import cv2
import numpy as np

from pool_run import PoolRun
from mode_finder import ModeFinder
from precise_time import PreciseTime
from roi_manip import convert_to_contours
from sort_contours_by_area import SortContours
from strsim_for_speed.computer_vision.speedy_str_sim_as_a_class import SpeedyCV
from strsim_for_speed.computer_vision.structural_sim_from_scratch import run_math, run_math_complete, normalize_diff

import logging
import argparse

from arg_helpers import setup_args, get_args

class NoGUIPoolRun(PoolRun):
    def __init__(self, exp_folder, event_schedule, debug, view, rois_fname, mode):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 992, 660
        self.FRAMES_TO_SAVE_AFTER = 1800
        self.FPS = 30

        self.exp_folder = exp_folder
        self.event_schedule = event_schedule
        self.debug = debug
        self.view = view
        self.rois_fname = rois_fname
        self.mode = mode

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{exp_folder}\\run.log', encoding='utf-8', level=logging.DEBUG)

    def tracking_pool(self, img_queue, done):
        self.logger.info("start tracking pool")

        timer = PreciseTime()

        cell_contours, contour_mask, cell_centers, cell_bounds, shape_of_rows = convert_to_contours(f"{self.exp_folder}{self.rois_fname}",
                                                                                       self.FRAME_WIDTH,
                                                                                       self.FRAME_HEIGHT)

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)

        sc = SortContours(contour_mask, shape_of_rows, cell_contours, cell_centers, cell_bounds)
        spd = SpeedyCV(self.FRAME_HEIGHT, self.FRAME_WIDTH)

        frame_counter = 0

        vy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C')

        detected_centroids = []

        output_filepath = f'{self.exp_folder}\\pre-processed.csv'

        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32)
        prev_thresh_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32)

        while not done.is_set():
            try:
                image = img_queue.get()

                arr_time = str(timer.formatted_time(timer.now())).strip("[]").split(", ")
                curr_time = "_".join(arr_time)

                if r.mode_noblur_img is None:
                    r.find_mode(frame_counter, image)
                else:
                    if r.mode_updated:
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

                    dpix_im = prev_thresh_img - thresh_img
                    sorted_contours = sc.sort_contours_by_area(contours, frame_counter, curr_time, spd.out, dpix_im)

                    detected_centroids.extend(sorted_contours)

                    if view:
                        for time, frame_count, row, col, x, y in sorted_contours:
                            cv2.circle(image, (x, y), 1, (0, 0, 255), 5, cv2.LINE_4)

                        cv2.imshow('frame', image)
                        cv2.imshow('diff', spd.out)

                        if cv2.waitKey(1) == 1:
                            break

                    prev_thresh_img = thresh_img
                    if frame_counter % self.FRAMES_TO_SAVE_AFTER == 0 and len(detected_centroids) > 0:
                        #print(frame_counter, np.array(detected_centroids).shape)
                        self.save_centroids_to_csv(output_filepath, detected_centroids)
                        detected_centroids.clear()

                frame_counter += 1

            except Exception as e:
                self.logger.error(f"tracking failed because: {e}")
                self.logger.error(gc.get_stats())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--view",
        action='store_true'
    )

    parser.add_argument(
        "-rois_fname",
        "--rois_fname",
        required=True
    )

    parser.add_argument(
        "-m",
        "--mode",
        action='store_true'
    )

    setup_args(parser)
    args = parser.parse_args()
    exp_folder, event_schedule, debug = get_args(args)

    view = args.view

    poolrun = PoolRun(exp_folder, event_schedule, debug)

    print('Acquiring images...')

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