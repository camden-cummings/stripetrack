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
from strsim_for_speed.computer_vision.structural_sim_from_scratch import correlate1d_x__ as correlate1d_x, correlate1d_y, run_math, normalize_diff, setup as ssim_setup, generate_weights

# do this through GUI instead
fn_start = "C:\\Users\\ThymeLab\\Desktop\\9-30-25\\"

import logging

# TODO try memlog to double check

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{fn_start}run.log', encoding='utf-8', level=logging.DEBUG)

fps = 30.0

class GUIPoolRun(PoolRun):
    def __init__(self):
        self.FRAME_HEIGHT, self.FRAME_WIDTH = 660, 992
        #self.FRAME_HEIGHT, self.FRAME_WIDTH = 1200, 1760
        
        self.image_data = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT, 3))

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

        r = ModeFinder(self.FRAME_WIDTH, self.FRAME_HEIGHT)#, f'{fn_start}pre-processed.csv', gui)
        frame_counter = 0
            

        data_range = 255
        
        weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)

        weight_size = len(weights)
        size1 = math.floor(weight_size / 2)
        size2 = weight_size - size1 - 1

        np_weights = np.asarray(weights, dtype=np.float32)

        ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = ssim_setup(self.FRAME_HEIGHT, self.FRAME_WIDTH, order='C', data_type=np.float32)
        ux, uy, uxx, uyy, uxy = ssim_setup(self.FRAME_HEIGHT, self.FRAME_WIDTH, order='C', data_type=np.float32) # doing this so we can transpose it later, numba requires C major order s.t. when we go in the Y direction, we want to
        vy = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C')
        uy_squared = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C')

        diff = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), order='C', dtype=np.uint8)
        S = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT), order='C', dtype=np.float64)
        
        S = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)
        S_t = S.T
                
        normalize_diff(S_t, self.FRAME_WIDTH, self.FRAME_HEIGHT, diff)

        image = img_queue.get()
        rearr = np.concatenate((image[0:size1][::-1], image, image[-size2:][::-1]))
        rearr = rearr.astype(dtype=np.float32)

        correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)
        rearr = np.concatenate((uyy_tmp[0:size1][::-1], uyy_tmp, uyy_tmp[-size2:][::-1]), axis=0)   
        correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
        
        detected_centroids = []
        
        FRAMES_TO_SAVE_AFTER = 1800
        output_filepath =  f'{fn_start}pre-processed.csv'
        
        prev_masked_img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.float32, order='C')
    
            
        while not done.is_set():    
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
                    
                    rearr = np.concatenate((uy_tmp[0:size1][::-1], uy_tmp, uy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uy)  # , curr_scipy)
                                    
                    inp = masked_mode_noblur_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)  # , curr_scipy)
                    
                    rearr = np.concatenate((uyy_tmp[0:size1][::-1], uyy_tmp, uyy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
                    
                    uy_squared = uy * uy
                    vy = cov_norm * (uyy - uy_squared)
                    
                    last_confident_centroid = [[gui.cell_centers[i][j] for j in range(gui.shape_of_rows[i])] for i in range(len(gui.shape_of_rows))]

                    gui.contours_updated = False
                    if r.mode_updated:
                        r.mode_updated = False
                        dpg.configure_item(gui.status, default_value="Status: Mode Ready")
                
                
                time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))

                masked_curr_img = cv2.bitwise_and(
                    image, image, mask=mask)
                masked_curr_img = masked_curr_img.astype(np.float32, copy=False)
                
                # TODO bundle into function
                # TODO rethink how to organize
                rearr = np.concatenate((masked_curr_img[0:size1][::-1], masked_curr_img, masked_curr_img[-size2:][::-1]))
                correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, ux_tmp)
                
                rearr = np.concatenate((ux_tmp[0:size1][::-1], ux_tmp, ux_tmp[-size2:][::-1]), axis=0)
                correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, ux) 
            
                inp = masked_curr_img * masked_curr_img
                rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxx_tmp) 
                
                rearr = np.concatenate((uxx_tmp[0:size1][::-1], uxx_tmp, uxx_tmp[-size2:][::-1]), axis=0)
                correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxx)
                
                if gui.contour_definer.going_to_mode_method:
                    rearr = np.concatenate((masked_mode_noblur_img[0:size1][::-1], masked_mode_noblur_img, masked_mode_noblur_img[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uy_tmp)  # , curr_scipy)
                    
                    rearr = np.concatenate((uy_tmp[0:size1][::-1], uy_tmp, uy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uy)  # , curr_scipy)
                                    
                    inp = masked_mode_noblur_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uyy_tmp)  # , curr_scipy)
                    
                    rearr = np.concatenate((uyy_tmp[0:size1][::-1], uyy_tmp, uyy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uyy)  # , curr_scipy)
                    
                    gui.contour_definer.going_to_mode_method = False
                    
                if "Prev2Curr" in gui.contour_definer.cv_method:
                    inp = masked_curr_img * prev_masked_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxy_tmp)
                    
                    rearr = np.concatenate((uxy_tmp[0:size1][::-1], uxy_tmp, uxy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxy)
    
                    prev_masked_img = masked_curr_img
    
                elif "Mode" in gui.contour_definer.cv_method:
                    inp = masked_curr_img * masked_mode_noblur_img
                    rearr = np.concatenate((inp[0:size1][::-1], inp, inp[-size2:][::-1]))
                    correlate1d_x(rearr, np_weights, weight_size, self.FRAME_HEIGHT, uxy_tmp)
                    
                    rearr = np.concatenate((uxy_tmp[0:size1][::-1], uxy_tmp, uxy_tmp[-size2:][::-1]), axis=0)
                    correlate1d_y(rearr, np_weights, weight_size, self.FRAME_WIDTH, uxy)                    
                
                    #arr_time = str(timer.formatted_time(timer.now())).strip("[]").split(", ")
    
                    #if (0 <= int(arr_time[1]) <= 10 and not r.found_mode) or (r.mode_noblur_img is None):
                    #    r.find_mode(frame_counter, image)
                    #elif arr_time[1] == 11:
                    #    r.found_mode = False
                    
                S = run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy)
                S_t = S.T
                normalize_diff(S_t, self.FRAME_WIDTH, self.FRAME_HEIGHT, diff)
                
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
                diff_data = np.asarray(diff_data, dtype='f')
                new = np.true_divide(diff_data, 255.0)
                
                dpg.set_value("texture_tag", new)

            dpg.render_dearpygui_frame()
            
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
    

