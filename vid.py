import os
#from threading import Thread
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
from tracker.helpers.centroid_manip import find_centroid_of_contour, check_masked_image
from skimage.metrics import structural_similarity
from tracker.helpers.centroid_manip import ignored_cen, generate_cen_masked_image, check_masked_image, generate_row_col

global continue_recording
global run_once
run_once = True

   
def find_centroids(comp_img, curr_img_gray, frame_count, min_area, max_area, shape_of_rows, cell_contours):
    # Compute SSIM between the two images
    (score, diff) = structural_similarity(comp_img, curr_img_gray, full=True)

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    cv2.drawContours(diff, contours, -1, (0,255,0), 1)

    contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    all_centr_in_frame = []
        
    for row, col in generate_row_col(shape_of_rows):
        cell_count = row * shape_of_rows[row] + col

        darkest_pixel_val = 255 
        posns = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            point_x, point_y = (int(x + w / 2), int(y + h / 2))

            in_polygon = cv2.pointPolygonTest(cell_contours[cell_count], (point_x, point_y), False)

            if in_polygon >= 0:
                R, G, B, _ = np.array(cv2.mean(diff_box[y:y + h, x:x + w])).astype(np.uint8)
                gray_avg = 0.299 * R + 0.587 * G + 0.114 * B

                posns.append(((point_x, point_y), gray_avg))

                if gray_avg < darkest_pixel_val:
                    darkest_pixel_val = gray_avg

        ten_darkest_centroids = sorted(posns, key=lambda posn: posn[1])[:10]

        for c in ten_darkest_centroids:
            all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])

    return all_centr_in_frame
    
    
class RunCV:
    def __init__(self):
        self.async_result = None
        self.mode_noblur_img = None
        self.curr_img = None
        self.curr_img_data = None
        self.beeg_array = np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        self.prev_diff = np.array(self.mask, dtype=np.uint8)

    def find_mode(self, frame_counter):
        global run_once
    
        if len(moviedeq) < 50:# and frame_counter % 50 == 0:
            moviedeq.append(self.curr_img)
        elif len(moviedeq) >= 50 and run_once == True:
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (moviedeq, FRAME_HEIGHT, FRAME_WIDTH))
            #mode_noblur_img = calc_mode(moviedeq, FRAME_HEIGHT, FRAME_WIDTH)
            
            run_once = False

        if self.async_result is not None and self.async_result.ready():
            self.mode_noblur_img = self.async_result.get()
            gui.mode_calculated = True
            gui.rt_tracker.standard_image_noise = gui.rt_tracker.CV_image_noise_light_background(self.mode_noblur_img)
            dpg.configure_item(gui.status, default_value="Status: Ready")
        
    def run_CV(self, frame_counter):
        match gui.contour_definer.cv_method:
            case "Structural Similarity":            
                gui.rt_tracker.MIN_AREA = gui.contour_definer.centroid_size
                #contours, diff = gui.rt_tracker.structural_sim_contours(self.curr_img, self.mode_noblur_img, min_thresh = gui.contour_definer.thresh) 

                masked_curr_img = cv2.bitwise_and(
                    self.curr_img, self.curr_img, mask=self.mask)
        
                self.prev_diff, all_centr_in_frame = find_centroids(self.prev_diff, masked_curr_img, False,
                                                  self.masked_mode_noblur_img, frame_counter, gui.contour_definer.centroid_size, 500000,
                                                  self.curr_img, gui.rt_tracker.shape_of_rows, gui.rt_tracker.cell_contours)
                                    
                for i in all_centr_in_frame:
                    print(i)
                    cv2.circle(self.curr_img_data, (i[3], i[4]), 1, (0,255,0), 1)
                """              
                if gui.show_only_inside_conts:
                    for c in contours:
                        cx, cy = find_centroid_of_contour(c)

                        if not check_masked_image((cx, cy), gui.rt_tracker.mask):
                            cv2.drawContours(self.curr_img_data, c, -1, (255, 0, 0), 1)
                else:
                    cv2.drawContours(self.curr_img_data, contours, -1, (255, 0, 0), 1)
                """
                
            case "Real Time":
                if any([frame_counter % x == 0 for x in range(50,59,1)]):            
                    noise_image = gui.rt_tracker.CV_image_noise_light_background(self.curr_img)
                    self.beeg_array += noise_image
                if frame_counter % 60 == 0:                    
                    self.beeg_array = self.beeg_array/3
                    self.beeg_array[self.beeg_array > 0] = 255
                    gui.rt_tracker.standard_image_noise = self.beeg_array
                
                gui.rt_tracker.MIN_AREA = gui.contour_definer.centroid_size
                sharpened_contours, contour_img = gui.rt_tracker.new_CV(self.curr_img, min_thresh = gui.contour_definer.thresh)
                
                if gui.show_only_inside_conts:
                    for c in sharpened_contours:
                        cx, cy = find_centroid_of_contour(c)

                        if not check_masked_image((cx, cy), gui.rt_tracker.mask):
                            cv2.drawContours(self.curr_img_data, c, -1, (255, 0, 0), 1)
                else:
                    cv2.drawContours(self.curr_img_data, sharpened_contours, -1, (255, 0, 0), 1)  
    
    
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

        print('Acquiring images...')
        
        get_device_serial_number(nodemap_tldevice)
        
        timer = PreciseTime()

        pr = cProfile.Profile()
        pr.enable()
        
        counter = 0
        schedule_times = pd.read_csv(os.getcwd() + "\\scheduled-events", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        
        frame_counter = 0
        
        r = RunCV()
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
                    
                    if type_of_video == 0:
                        duration = None
                    elif type_of_video == 1:
                        duration = 1
                    elif type_of_video == 2:
                        duration = 108000
                    
                    if duration != None:
                        vid_name = "_".join(str(at_time).strip("[]").split(", ")) + "-" + str(counter)
                        
                        if type_of_video == 1:
                            result = cv2.VideoWriter(f'{vid_name}.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                     285, (FRAME_WIDTH, FRAME_HEIGHT), False)
                        else:
                            result = cv2.VideoWriter(f'{vid_name}-long.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                         30, (FRAME_WIDTH, FRAME_HEIGHT), False)
                        
                    first_time = False
                    not_written_to_arduino = True
                    
                if timer.formatted_time(timer.now()) == at_time:
                    if end_time == np.inf:
                        end_time = int(timer.now()) + duration
                    result.write(image)
                    
                    if not_written_to_arduino:
                        dev.write(bytes(command_string, 'utf-8'))
                        not_written_to_arduino = False
    
                if (timer.now() >= end_time):
                    counter += 1    
                    first_time = True
                    end_time = np.inf

            elif r.mode_noblur_img is None:
                r.find_mode(frame_counter)
            elif gui.contour_overlay:
                
                r.masked_mode_noblur_img = cv2.bitwise_and(
                    r.mode_noblur_img, r.mode_noblur_img, mask=r.mask)
                r.run_CV(frame_counter)
            
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

    FRAME_HEIGHT, FRAME_WIDTH = 850, 1248

    moviedeq = []

    dpg.create_context()

    #mode_noblur_path = ex_fn[:-4] + "-mode.png"
    #mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)


    gui = GUIHelpers(FRAME_HEIGHT, FRAME_WIDTH)

    dpg.set_primary_window(gui.window, True)
    dpg.create_viewport(width=int(FRAME_WIDTH*1.5), height=FRAME_HEIGHT+20, title="ROI Selector")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    video()
#video_thread = Thread(target=video)
#video_thread.start()
