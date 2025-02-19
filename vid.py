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

from precise_time import PreciseTime
from tracker.cell_finder_helpers.calc_mode import calc_mode
from AcquireAndDisplayClass import get_image
from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_device_serial_number
from gui_helpers import GUIHelpers

global continue_recording
global run_once
run_once = True

class RunCV:
    def __init__(self):
        self.async_result = None
        self.mode_noblur_img = None
        self.beeg_array = np.zeros((FRAME_HEIGHT, FRAME_WIDTH))

    def find_mode(self, image, image_data, frame_counter):
        global run_once
    
    #    print(len(moviedeq))
        if len(moviedeq) < 50:# and frame_counter % 50 == 0:
            print(frame_counter)
            moviedeq.append(image)
        elif len(moviedeq) >= 50 and run_once == True:
            #mode_thread = Thread(target=calc_mode, args=(moviedeq, FRAME_HEIGHT, FRAME_WIDTH))
            #mode_thread.start()
            print("starting")
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (moviedeq, FRAME_HEIGHT, FRAME_WIDTH))
            #mode_noblur_img = calc_mode(moviedeq, FRAME_HEIGHT, FRAME_WIDTH)
            
            run_once = False

        if self.async_result is not None and self.async_result.ready():
            self.mode_noblur_img = self.async_result.get()
            gui.rt_tracker.standard_image_noise = gui.rt_tracker.CV_image_noise_light_background(self.mode_noblur_img)
        
        
    def run_CV(self, image, image_data, frame_counter):
        match gui.contour_definer.cv_method:
            case "Structural Similarity":            
                gui.rt_tracker.MIN_AREA = gui.contour_definer.centroid_size
                contours, diff = gui.rt_tracker.structural_sim_contours(image, self.mode_noblur_img, min_thresh = gui.contour_definer.thresh)  
                cv2.drawContours(image_data, contours, -1, (255, 0, 0), 1)
            case "Real Time":
                if any([frame_counter % x == 0 for x in range(50,59,1)]):            
                    noise_image = gui.rt_tracker.CV_image_noise_light_background(image)
                    self.beeg_array += noise_image
                if frame_counter % 60 == 0:                    
                    self.beeg_array = self.beeg_array/3
                    self.beeg_array[self.beeg_array > 0] = 255
                    gui.rt_tracker.standard_image_noise = self.beeg_array
                
                gui.rt_tracker.MIN_AREA = gui.contour_definer.centroid_size
                sharpened_contours, contour_img = gui.rt_tracker.new_CV(image, min_thresh = gui.contour_definer.thresh)
                cv2.drawContours(image_data, sharpened_contours, -1, (0, 255, 0), 1)  # RED

        return image_data    
    
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
        while counter < num_of_instructions and dpg.is_dearpygui_running():
            image = get_image(cam)
            
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
                    vid_name = "-".join(str(at_time).strip("[]").split(", "))
                    
                    if type_of_video == 1:
                        result = cv2.VideoWriter(f'{vid_name}.avi',
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                 285, (FRAME_WIDTH, FRAME_HEIGHT), False)
                    else:
                        result = cv2.VideoWriter(f'{vid_name}.avi',
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                     30, (FRAME_WIDTH, FRAME_HEIGHT), False)
                    
                first_time = False
                
            if timer.formatted_time(timer.now()) == at_time:
                if end_time == np.inf:
                    end_time = int(timer.now()) + duration
                result.write(image)

            if (timer.now() >= end_time):
                counter += 1    
                first_time = True
                end_time = np.inf

            image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
             
            
            if r.mode_noblur_img is None:
                r.find_mode(image, image_data, frame_counter)
            else:
                image_data = r.run_CV(image, image_data, frame_counter)
            
            
            data = np.flip(image_data, 2)
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

    FRAME_HEIGHT, FRAME_WIDTH = 660, 1088

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
