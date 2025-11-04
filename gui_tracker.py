import cProfile
import io
import math
import multiprocessing
import pstats
from multiprocessing import Process
from pstats import SortKey

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from gui_helpers import GUIHelpers
from mode_finder import ModeFinder
from no_gui_tracker import PoolRun
from precise_time import PreciseTime
from sort_contours_by_area import sort_contours_by_area
from strsim_for_speed.computer_vision.structural_sim_from_scratch import correlate1d_x, correlate1d_y, run_math, normalize_diff, setup as ssim_setup, generate_weights
from strsim_for_speed.computer_vision.speedy_str_sim_as_a_class import SpeedyCV

import logging
import argparse

# TODO try memlog to double check

fps = 30.0

parser = argparse.ArgumentParser()

parser.add_argument(
   "-exp_folder",
   "--exp_folder",
   required=True
)

parser.add_argument(
   "-rois_fname",
   "--rois_fname",
   required=True
)

parser.add_argument(
   "-event_schedule",
   "--event_schedule",
   required=True
)

parser.add_argument(
   "-d",
   "--debug",
   action='store_true'
)

args = parser.parse_args()

exp_folder = args.exp_folder

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{exp_folder}\\run.log', encoding='utf-8', level=logging.DEBUG)

class GUIPoolRun(PoolRun):
    def tracking_pool(self, img_queue, done):
        logger.info("start gui pool")

        dpg.create_context()
        window = dpg.add_window(label="Video player", pos=(50, 50), width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT) 
        gui = GUIHelpers(window, self.FRAME_WIDTH, self.FRAME_HEIGHT)

        timer = PreciseTime()

        dpg.set_primary_window(window, True)
        dpg.create_viewport(width=int(self.FRAME_WIDTH*1.5), height=self.FRAME_HEIGHT+20, title="ROI Selector")
        dpg.setup_dearpygui()
        dpg.show_viewport()

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)
        frame_counter = 0
            
        detected_centroids = []
        
        FRAMES_TO_SAVE_AFTER = 1800
        output_filepath =  f'{exp_folder}\pre-processed.csv'
        
        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32, order='C')
    
        while not done.is_set():    
            pr = cProfile.Profile()
            pr.enable()
      
            image = img_queue.get()

            data = np.asarray(image, dtype='f') 
            image_data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)            

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

                    self.spd.run_mode(masked_mode_noblur_img)
                    uy_squared = self.spd.uy * self.spd.uy
                    vy = self.spd.cov_norm * self.spd.uyy - uy_squared
                    
                    last_confident_centroid = [[gui.cell_centers[i][j] for j in range(gui.shape_of_rows[i])] for i in range(len(gui.shape_of_rows))]

                    gui.contours_updated = False
                    if r.mode_updated:
                        r.mode_updated = False
                        dpg.configure_item(gui.status, default_value="Status: Mode Ready")
                
                
                time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))

                masked_curr_img = cv2.bitwise_and(
                    image, image, mask=mask)
                masked_curr_img = masked_curr_img.astype(np.float32, copy=False)

                self.spd.run_corr(masked_curr_img)
                
                if gui.contour_definer.going_to_mode_method:
                    self.spd.run_mode(masked_mode_noblur_img)

                    gui.contour_definer.going_to_mode_method = False
                
                if "Prev2Curr" in gui.contour_definer.cv_method:
                    self.spd.run_against(masked_curr_img, prev_masked_img)

                    prev_masked_img = masked_curr_img
                    
                elif "Mode" in gui.contour_definer.cv_method:  
                    self.spd.run_against(masked_curr_img, masked_mode_noblur_img)    

                S = run_math(self.spd.cov_norm, self.spd.data_range, self.spd.ux, self.spd.uy, self.spd.uxx, vy, self.spd.uxy)
                S_t = S.T
                normalize_diff(S_t, self.FRAME_WIDTH, self.FRAME_HEIGHT, self.spd.out)


                if "Prev2Curr" in gui.contour_definer.cv_method:
                    uy = self.spd.ux.copy()
                    self.spd.uy = uy
                    uy_squared = np.multiply(uy, uy)
                    vy = self.spd.cov_norm * (self.spd.uxx - uy_squared)

                thresh_img = cv2.threshold(self.spd.out, gui.contour_definer.thresh, 255, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                sorted_contours = sort_contours_by_area(contours, last_confident_centroid, frame_counter, time_, self.spd.out, mask, gui.shape_of_rows, gui.cell_contours, gui.cell_centers, gui.contour_definer.centroid_size)

                detected_centroids.extend(sorted_contours)

                if "Structural Similarity" in gui.contour_definer.cv_method:
                    for time, frame_count, row, col, x, y in sorted_contours:
                        cv2.circle(image_data, (x, y), 1, (0, 0, 255), 5, cv2.LINE_4)
                    
                if frame_counter % FRAMES_TO_SAVE_AFTER == 0 and len(detected_centroids) > 0:
                    self.save_centroids_to_csv(output_filepath, detected_centroids)
                    detected_centroids.clear()

            frame_counter += 1
            
            texture_data = np.true_divide(image_data, 255.0)

            if "Structural Similarity" in gui.contour_definer.cv_method or gui.contour_definer.cv_method == "":
                dpg.set_value("texture_tag", texture_data)
            elif "Diff" in gui.contour_definer.cv_method:
                diff_data = cv2.cvtColor(self.spd.out, cv2.COLOR_GRAY2RGB)
                diff_data = np.asarray(diff_data, dtype='f')
                new = np.true_divide(diff_data, 255.0)
                
                dpg.set_value("texture_tag", new)

            dpg.render_dearpygui_frame()
                    
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print("printer pool", s.getvalue())

        dpg.destroy_context()

        
if __name__ == '__main__':   
    poolrun = GUIPoolRun()

    print('Acquiring images...')

    
    queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    fps_commands_vid, fps_commands_p = multiprocessing.Pipe()
    recording_commands_gui, recording_commands_p = multiprocessing.Pipe()

    vid_p = Process(target=poolrun.video_pool, args=(queue, done, fps_commands_vid, recording_queue,))
    tracking_p = Process(target=poolrun.tracking_pool, args=(queue, done, ))
    p = Process(target=poolrun.printer_pool, args=(done, fps_commands_p, recording_commands_p, ))
    vid_rec_p = Process(target=poolrun.video_recorder_pool, args=(recording_queue, recording_commands_gui, done, ))
    vid_p.start()    
    tracking_p.start()
    p.start()
    vid_rec_p.start()
    vid_p.join()
    tracking_p.join()
    p.join()
    vid_rec_p.join()
    
