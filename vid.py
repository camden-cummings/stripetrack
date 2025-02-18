import PySpin
from AcquireAndDisplayClass import get_image
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pyplot as plt

import cv2
import numpy as np
global continue_recording
import dearpygui.dearpygui as dpg

from arduino_server import PreciseTime
import pandas as pd
import os
from threading import Thread

from tracker.real_time_tracker import RealTimeTracker
from tracker.cell_finder_helpers.calc_mode import calc_mode
from multiprocessing.pool import ThreadPool

from tracker.roi_manip import convert_to_contours

from pathlib import Path

from pynput.keyboard import Key, Controller

from roi_selector_gui_dpg.statemanager import StateManager

import subprocess

from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode

global continue_recording

#from tracker.structural_similarity_tracker import StructuralSimilarityTracker

continue_recording = True
FRAME_HEIGHT, FRAME_WIDTH = 660, 1088
raw_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

moviedeq = []
mode_noblur_img = None

dpg.create_context()

def update_dynamic_texture(new_frame):
    global raw_data
    h2, w2, _ = new_frame.shape
    raw_data[:h2, :w2] = new_frame[:,:] / 255

#mode_noblur_path = ex_fn[:-4] + "-mode.png"
#mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

class ContourDefiner:
    def __init__(self):
        self.cv_method = ""
        self.thresh = 155
        self.centroid_size = 70
        
    def cv_alg_change(self, sender_id, data):
        self.cv_method = data
        
    def threshold_change(self, sender_id, data):
        self.thresh = data
        
    def centroid_change(self, sender_id, data):
        self.centroid_size = data
     
c = ContourDefiner()

global rt_tracker
rt_tracker = RealTimeTracker([], [], 0, 0, 0)

def set_cells(_, appdata):        
    cell_contours = []
    
    min_area = 40
    max_area = 300
    length_req = 40
    
    cell_contours, contour_mask, cell_centers, shape_of_rows = convert_to_contours(appdata["filepathname"], FRAME_WIDTH, FRAME_HEIGHT)

    rt_tracker = RealTimeTracker(cell_contours, shape_of_rows, min_area, max_area, length_req)


def __change(e, data):
    if data == "ROI":
        dpg.show_item(roi)
        dpg.hide_item(line)
        
        state_manager.ROI = True
    elif data == "Line":
        dpg.show_item(line)
        dpg.hide_item(roi)
        state_manager.ROI = False

def __generate_rois():
    dpg.hide_item(roi_and_line_selection)
    dpg.show_item(post_line)
    
    state_manager.generate_rois()

def __restart():
    dpg.show_item(roi_and_line_selection)
    dpg.hide_item(post_line)

    state_manager.clear_window()
    print(state_manager.roi_interface.rois)
    
def __start_movies_and_stimuli():
    print("go to arduino script here")
    keyboard = Controller()
    keyboard.press(Key.enter)
    subprocess.call(['python.exe',"arduino_server.py"])

contour_overlay = False
def tab_callback(_, tab_id):
    match dpg.get_item_configuration(tab_id)["label"]:
        case "ROI Selection":
            contour_overlay=False
        case "Contour Overlay": 
            contour_overlay=True
            min_area = 40
            max_area = 300
            length_req = 40
            
            cell_contours, contour_mask, cell_centers, shape_of_rows = convert_to_contours(state_manager.roi_interface.convert_rois_to_lines(state_manager.roi_interface.rois), FRAME_WIDTH, FRAME_HEIGHT)
            
            rt_tracker = RealTimeTracker(cell_contours, shape_of_rows, min_area, max_area, length_req)
    #def hide_rois(self):
    
#def hide_lines(self):
    

#generate_row_col(shape_of_rows)
    
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(FRAME_WIDTH, FRAME_HEIGHT, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")

right_shift = FRAME_WIDTH+10
std_shift = 8
with dpg.theme() as canvas_theme, dpg.theme_component():
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)

with dpg.window(label="Video player", pos=(0,0), width = FRAME_WIDTH, height=FRAME_HEIGHT+150) as window:    
    with dpg.tab_bar(label="Select", callback=tab_callback):
        with dpg.tab(label='ROI Selection'):
            state_manager = StateManager(FRAME_WIDTH, FRAME_HEIGHT, window, shift=(0,23))
            
            with dpg.handler_registry():
                dpg.add_mouse_move_handler(callback=state_manager.motion_notify_callback)
                dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=state_manager.release)
                dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=state_manager.left_mouse_press_callback)
                dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=state_manager.right_mouse_press_callback)
                dpg.add_key_press_handler(key=dpg.mvKey_C, callback=state_manager.copy)
                dpg.add_key_press_handler(key=dpg.mvKey_LControl, callback=state_manager.control)
                dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=state_manager.delete)
                
            with dpg.child_window(border=False):
                with dpg.group() as roi_and_line_selection:
                    dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[right_shift,0], callback=__change, default_value="ROI")
                    dpg.add_button(label="START", callback=__start_movies_and_stimuli)
                    
                    path = Path(os.getcwd())

                    with dpg.group(label="roi buttons", pos=[right_shift,25]) as roi: #ROI Mode Buttons
                        dpg.add_button(label="New ROI", callback=state_manager.new_roi)
                        
                        curr_dir = path.parent
                        curr_name = str(path.stem)
                        
                        with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.load_rois, id="roi_load_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                            dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                            
                        dpg.add_button(label="Load ROI File", callback=lambda: dpg.show_item("roi_load_file"))
                
                        with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois, id="roi_save_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                            dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                        
                        dpg.add_button(label="Save ROIs", callback=lambda: dpg.show_item("roi_save_file"))
                        
                        dpg.add_button(label="Auto Generate ROIs", callback=state_manager.auto_gen_rois)
                        
                        dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(right_shift+5, 125), wrap=150)
                    
        
                    with dpg.group(label="line buttons", pos=[right_shift, 25]) as line:
                        dpg.add_button(label="Vertical Line", callback=state_manager.line_interface.vertical_line_callback)
                        vert = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[right_shift+104,25], callback=state_manager.line_interface.num_of_vert_lines_changer)
                        dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface.horizontal_line_callback)
                        hor = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[right_shift+118,48], callback=state_manager.line_interface.num_of_hor_lines_changer)
                        dpg.add_button(label="Generate ROIs", callback=__generate_rois)
        
                        with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.save_lines, id="line_save_file", width=700 ,height=400):
                            dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                            
                        dpg.add_button(label="Save Line Configuration", callback="line_save_file")
        
                        with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.load_lines, id="line_load_file", width=700 ,height=400):
                            dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                        
                        dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))
        
                        dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(right_shift+5, 195), wrap=150)
        
                    dpg.hide_item(line)
                
                with dpg.group(label="post line buttons", pos=[right_shift,0]) as post_line:
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois, id="roi_gen_save_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                        dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                        
                    dpg.add_button(label="Save ROIs",  callback=lambda: dpg.show_item("roi_gen_save_file"))#callback=state_manager.roi_interface.save_rois)
                    dpg.add_button(label="Clear Screen and Start Over", callback=__restart)
        
                dpg.hide_item(post_line)

            dpg.add_image("texture_tag", pos=(std_shift, std_shift+23))
        with dpg.tab(label='Contour Overlay'):
            dpg.add_image("texture_tag")
            
            with dpg.group(pos = [right_shift, std_shift+23]):
                with dpg.file_dialog(directory_selector=False, show=False, callback=set_cells, id="set_rois", width=0 ,height=0):
                    dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                
                #dpg.add_checkbox(label = "Show ROIs", callback=):
                
                with dpg.tree_node(label="Basic", default_open=True):
                    dpg.add_combo(("No Contours", "Structural Similarity", "Real Time"), label="Contour Detecting Algorithm", callback=c.cv_alg_change, default_value="No Contours", width=180)
                    with dpg.group(width = 300):
                        dpg.add_slider_float(label="Threshold", callback=c.threshold_change, min_value=0, max_value=255, default_value=c.thresh)            
                        dpg.add_slider_float(label="Centroid Size", callback=c.centroid_change, max_value=1000, default_value=c.centroid_size)
                
    
dpg.set_primary_window(window, True)
dpg.create_viewport(width=int(FRAME_WIDTH*1.5), height=FRAME_HEIGHT+20, title="ROI Selector")
dpg.setup_dearpygui()
dpg.show_viewport()

"""
def CV(image, image_data, frame_counter):
    if len(moviedeq) < 50:
        moviedeq.append(image)
    elif len(moviedeq) >= 50 and run_once == True:
        #mode_thread = Thread(target=calc_mode, args=(moviedeq, FRAME_HEIGHT, FRAME_WIDTH))
        #mode_thread.start()
        print("starting")
        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(calc_mode, (moviedeq, FRAME_HEIGHT, FRAME_WIDTH))
        #mode_noblur_img = calc_mode(moviedeq, FRAME_HEIGHT, FRAME_WIDTH)
        
        run_once = False
        
        if not async_result.is_alive():
            print("!")
            mode_noblur_img = async_result.get()
            
    #if len(moviedeq) >= 50:
    #   

    match c.cv_method:
        case "Structural Similarity":            
            rt_tracker.MIN_AREA = c.centroid_size
            print(mode_noblur_img)
            contours, diff = rt_tracker.structural_sim_contours(image, mode_noblur_img, min_thresh = c.thresh)     
            cv2.drawContours(image_data, contours, -1, (255, 0, 0), 1)
        case "Real Time":
            
            #if frame_counter % 50 == 0:
            #    f.standard_image_noise = f.find_image_noise(vidcap, 50)
            
            rt_tracker.MIN_AREA = c.centroid_size
            sharpened_contours, contour_img = rt_tracker.new_CV(image, min_thresh = c.thresh)
            cv2.drawContours(image_data, sharpened_contours, -1, (0, 255, 0), 1)  # RED
"""

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
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        try:
            nodemap, nodemap_tldevice = setup_nodemap(cam)
            print('*** IMAGE ACQUISITION ***\n')
            try:
                set_node_acquisition_mode(nodemap)

                cam.BeginAcquisition()
        
                print('Acquiring images...')
        
                device_serial_number = ''
                node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(node_device_serial_number):
                    device_serial_number = node_device_serial_number.GetValue()
                    print('Device serial number retrieved as %s...' % device_serial_number)
        
                # Close program
               # print('Press enter to close the program..')
                timer = PreciseTime()
                # Retrieve and display images
                pr = cProfile.Profile()
                pr.enable()
                
                counter = 0
                schedule_times = pd.read_csv(os.getcwd() + "\\scheduled-events", sep="\t", header=None)
                num_of_instructions = schedule_times.shape[0]
                first_time = True
                end_time = np.inf
                
                mode_noblur_img = np.zeros((FRAME_WIDTH, FRAME_HEIGHT))
                
                
                frame_counter = 0
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
                    
                    moviedeq.append(image)
 
                    #CV(image, image_data, frame_counter)
                    
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
        
            except Exception as e:
                print(e)
                print("failed during image GUI capture")
                
            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

            print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

video()
#video_thread = Thread(target=video)
#video_thread.start()
