import os
from multiprocessing import Pool
import cProfile, pstats, io
from pstats import SortKey

import PySpin
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import pandas as pd
import serial

from precise_time import PreciseTime
from tracker.cell_finder_helpers.calc_mode import calc_mode
from AcquireAndDisplayClass import get_image
from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_device_serial_number
from gui_helpers import GUIHelpers
from tracker.helpers.centroid_manip import find_centroid_of_contour, check_masked_image, generate_row_col
from speedy_str_sim import correlate1d, run_math #structural_similarity

global continue_recording
global run_once

run_once = True

DESIRED_MODE_FRAMES = 50
   
weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
 0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]

np_weights = np.asarray(weights)
    
FRAMES_TO_SAVE_AFTER = 100
pre_output_filepath = 'pre-processed.csv'

class RunCV:
    def __init__(self, FRAME_WIDTH, FRAME_HEIGHT, output_filepath, cell_contours, shape_of_rows):
        self.async_result = None

        self.mode_noblur_img = None
        self.curr_img = None
        self.curr_img_data = None
        self.mask = None
        self.masked_mode_noblur_img = None

        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.FRAME_WIDTH = FRAME_WIDTH
        self.detected_centroids = []
        self.output_filepath = output_filepath
        self.movie_deq = []
        self.cell_contours = cell_contours
        self.shape_of_rows = shape_of_rows
        
        sigma = 1.5
        truncate = 3.5
        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        self.win_size = 2 * r + 1
        
        
    def find_mode(self, frame_counter):
        print('finding mode', len(self.movie_deq), DESIRED_MODE_FRAMES)
        global run_once

        if len(self.movie_deq) < DESIRED_MODE_FRAMES and frame_counter % 50 == 0:
            self.movie_deq.append(self.curr_img)
        elif len(self.movie_deq) >= DESIRED_MODE_FRAMES and run_once == True:
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (self.movie_deq, self.FRAME_HEIGHT, self.FRAME_WIDTH))
            #mode_noblur_img = calc_mode(movie_deq, FRAME_HEIGHT, FRAME_WIDTH)
            
            run_once = False

        if self.async_result is not None and self.async_result.ready():
            self.mode_noblur_img = self.async_result.get()

            #self.gui.mode_calculated = True
            #self.gui.rt_tracker.standard_image_noise = self.gui.rt_tracker.CV_image_noise_light_background(self.mode_noblur_img)
            #dpg.configure_item(self.gui.status, default_value="Status: Ready")
        
    def run_CV(self, frame_counter, time, ux, uy, uxx, uyy, uxy):
        if self.mask is not None:
            """
            match self.gui.contour_definer.cv_method:
                case "Structural Similarity":  
                    #str_pr = cProfile.Profile()
                    #str_pr.enable()

                    #str_pr.disable()
                    #s = io.StringIO()
                    #sortby = SortKey.CUMULATIVE
                    #ps = pstats.Stats(str_pr, stream=s).sort_stats(sortby)
                    #ps.print_stats()
                    #print(s.getvalue())
                    
                case "Real Time":
                    # TODO flesh out or remove
                    pass
            """
            start_frame = 0

            masked_curr_img = cv2.bitwise_and(
                self.curr_img, self.curr_img, mask=self.mask)

            contours, diff_box = self.find_centroids(masked_curr_img, ux, uy, uxx, uyy, uxy)
            sorted_contours = self.sort_contours_by_area(contours, frame_counter, time, diff_box)
            self.detected_centroids.extend(sorted_contours)

            #if gui.show_only_inside_contours:
            #    for i in sorted_contours:
            #        cv2.circle(self.curr_img_data, (i[4], i[5]), 1, (0, 255, 0), 1)
            #else:
            for i in contours:
                center = find_centroid_of_contour(i)
                cv2.circle(self.curr_img_data, center, 1, (0, 255, 0), 1)
            
            
            #cv2.imshow('f', self.curr_img_data)
        
            
            if frame_counter % FRAMES_TO_SAVE_AFTER == 0 and len(self.detected_centroids) > 0:
                if start_frame == 0:
                    new = pd.DataFrame(np.matrix(self.detected_centroids),
                                       columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y'])
                    new.to_csv(self.output_filepath, sep=',', index=False)
                else:
                    new = pd.DataFrame(np.matrix(self.detected_centroids),
                                       columns=['time', 'frame', 'row', 'col', 'pos_x', 'pos_y'])
                    new.to_csv(self.output_filepath, sep=',', mode='a', index=False, header=False)
                    self.detected_centroids.clear()
        
    def find_centroids(self, curr_img_gray, ux, uy, uxx, uyy, uxy):
        ndim = self.masked_mode_noblur_img.ndim
    
        # ndimage filters need floating point data
        curr_img_gray = curr_img_gray.astype(np.float64, copy=False)
    
        correlate1d(curr_img_gray, weights, ux)
        correlate1d(self.masked_mode_noblur_img, weights, uy)
    
        correlate1d(curr_img_gray * curr_img_gray, weights, uxx)
        correlate1d(self.masked_mode_noblur_img * self.masked_mode_noblur_img, weights, uyy)
        correlate1d(curr_img_gray * self.masked_mode_noblur_img, weights, uxy)
    
        cov_norm = self.win_size**ndim / (self.win_size**ndim - 1)  # sample covariance
        diff = run_math(cov_norm, 255, ux, uy, uxx, uyy, uxy)
    
        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])
        
        #thresh_img = cv2.threshold(diff, self.gui.contour_definer.thresh, 255, cv2.THRESH_BINARY)[1]

        thresh_img = cv2.threshold(diff, 155, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
    
    #    cv2.drawContours(diff, contours, -1, (0,255,0), 1)
        #contours = [c for c in contours if self.gui.contour_definer.centroid_size < cv2.contourArea(c) < 500000]
        contours = [c for c in contours if 70 < cv2.contourArea(c) < 500000]

        """
        for row, col in generate_row_col(shape_of_rows):
            cell_count = row * shape_of_rows[row] + col
    
            #darkest_pixel_val = 255 
            posns = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
    
                point_x, point_y = (int(x + w / 2), int(y + h / 2)) #not exact as find_centroid_of_contour, but faster
    
                in_polygon = cv2.pointPolygonTest(cell_contours[cell_count], (point_x, point_y), False)
    
                if in_polygon >= 0:
                    R, G, B, _ = np.array(cv2.mean(diff_box[y:y + h, x:x + w])).astype(np.uint8)
                    gray_avg = 0.299 * R + 0.587 * G + 0.114 * B
    
                    posns.append(((point_x, point_y), gray_avg))
    
                    #if gray_avg < darkest_pixel_val:
                    #    darkest_pixel_val = gray_avg
    
            ten_darkest_centroids = sorted(posns, key=lambda posn: posn[1])[:10]
    
            if len(ten_darkest_centroids) > 0:
                all_centr_in_frame.append([frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1]])
            #for c in ten_darkest_centroids:
            #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
        """
        return contours, diff_box

    def sort_contours_by_area(self, contours, frame_count, time, diff_box):
        # darkest_pixel_val = 255
        posns = [[[] for j in range(self.shape_of_rows[i])] for i in
                 range(len(self.shape_of_rows))]

        sorted_contours = []

        for c in contours:
            # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
            # a well, and that we know which one it is in)

            x, y, w, h = cv2.boundingRect(c)

            point_x, point_y = (int(x + w / 2), int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster

            if not check_masked_image((point_x, point_y), self.mask):
                for row, col in generate_row_col(self.shape_of_rows):
                    cell_count = row * self.shape_of_rows[row] + col

                    in_polygon = cv2.pointPolygonTest(self.cell_contours[cell_count], (point_x, point_y),
                                                      False)

                    if in_polygon >= 0:
                        # TODO this looks extremely speed-up-able, are we just taking the mean of some numbers that will be the same every run?
                        R, G, B, _ = np.array(cv2.mean(diff_box[y:y + h, x:x + w])).astype(np.uint8)
                        gray_avg = 0.299 * R + 0.587 * G + 0.114 * B

                        posns[row][col].append(((point_x, point_y), gray_avg))

                        break
                        # if gray_avg < darkest_pixel_val:
                        #   darkest_pixel_val = gray_avg

        for row, col in generate_row_col(self.shape_of_rows):
            ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]
            if len(ten_darkest_centroids) > 0:
                sorted_contours.append(
                    [time, frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1]])

                # for c in ten_darkest_centroids:
                #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
        return sorted_contours

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
        at_time[0] += 12

    video_type = cmd_string.iloc[2]

    return at_time, arduino_command, video_type

def video():
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    cam_list = setup(system)

    # Run example on each camera
    cam = cam_list[0]
        
    try:
        nodemap, nodemap_tldevice = setup_nodemap(cam)
        print('*** IMAGE ACQUISITION ***\n')
        
        set_node_acquisition_mode(nodemap)

        cam.BeginAcquisition()

        node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        node_fps.SetValue(30.0)

        print('Acquiring images...')
        
        timer = PreciseTime()

        pr = cProfile.Profile()
        pr.enable()
        
        counter = 0
        schedule_times = pd.read_csv(os.getcwd() + "\\shortened-schedule", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        
        frame_counter = 0
        
        r = RunCV(FRAME_WIDTH, FRAME_HEIGHT, pre_output_filepath, gui)
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)

        while counter < num_of_instructions and dpg.is_dearpygui_running():
            image = get_image(cam)
            image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            r.curr_img = image
            r.curr_img_data = image_data
            
            if gui.start_recording:
                if first_time: #setup all necessary pieces    
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    print("COMMANDS: ", at_time, command_string, type_of_video)
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    elif type_of_video == 2:
                        duration = 108000
                    
                    if duration != None:                        
                        vid_name = "_".join(str(at_time).strip("[]").split(", ")) + "-" + str(counter)
                        node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))

                        if type_of_video == 1:
                            node_fps.SetValue(285.0)
                            result = cv2.VideoWriter(f'{vid_name}.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                     285, (FRAME_WIDTH, FRAME_HEIGHT), False)
                        else:
                            node_fps.SetValue(30.0)
                            result = cv2.VideoWriter(f'{vid_name}-long.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                         30, (FRAME_WIDTH, FRAME_HEIGHT), False)
                        
                    first_time = False
                    not_written_to_arduino = True
                    
                if (timer.formatted_time(timer.now()) == at_time and (type_of_video == 1 or type_of_video == 0)) or (timer.formatted_time(timer.now()) >= at_time and type_of_video == 2):
                    if end_time == np.inf:
                        end_time = int(timer.now()) + duration
                        
                    if duration != 0:
                        result.write(image)
                    
                    if not_written_to_arduino:
                        dev.write(bytes(command_string, 'utf-8'))
                        not_written_to_arduino = False
                    
                if (timer.now() >= end_time):
                    counter += 1    
                    first_time = True
                    end_time = np.inf

            if r.mode_noblur_img is None:
                r.find_mode(frame_counter)
            elif gui.contour_overlay:
                if gui.contours_updated:
                    contour_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    for c in gui.rt_tracker.cell_contours:
                        contour_mask = cv2.drawContours(contour_mask, [c],
                                                        -1, (255, 255, 255), thickness=cv2.FILLED)
                        
                    r.mask = cv2.cvtColor(
                        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
                    
                    r.masked_mode_noblur_img = cv2.bitwise_and(
                        r.mode_noblur_img, r.mode_noblur_img, mask=r.mask)
                    
                    gui.contours_updated = False
                
                time_ = "_".join(str(timer.formatted_time(timer.now())).strip("[]").split(", "))
                r.run_CV(frame_counter, time_)
            
            data = np.flip(r.curr_img_data, 2)
            data = data.ravel()
            data = np.asfarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
            
            frame_counter += 1
            
            dpg.set_value("texture_tag", texture_data)
            dpg.render_dearpygui_frame()
            
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        cam.EndAcquisition()
            
        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)


    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

if __name__ == '__main__':    
    continue_recording = True
    contour_overlay = False

    FRAME_HEIGHT, FRAME_WIDTH = 652, 1024

    dpg.create_context()

    gui = GUIHelpers(FRAME_HEIGHT, FRAME_WIDTH)

    dpg.set_primary_window(gui.window, True)
    dpg.create_viewport(width=int(FRAME_WIDTH*1.5), height=FRAME_HEIGHT+20, title="ROI Selector")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    video()
#video_thread = Thread(target=video)
#video_thread.start()
