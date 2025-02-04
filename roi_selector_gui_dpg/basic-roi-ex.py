import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from pathlib import Path

from statemanager import StateManager

import subprocess

dpg.create_context()

fp = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/"
#ex_fn = fp + "fc2_save_2025-01-31-125230-0000.mp4"
ex_fn = fp + "fc2_save_2025-01-31-125333-0000.mp4"

vidcap = cv2.VideoCapture(ex_fn)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

raw_data = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

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
    subprocess.call("arduino filepath")

def setup_elements():
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(frame_width, frame_height, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
        
    with dpg.theme() as canvas_theme, dpg.theme_component():
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)
    
    with dpg.window(label="Video player", pos=(50,50), width = frame_width, height=frame_height) as window:
        state_manager = StateManager(frame_width, frame_height, window)
    
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
                shift = frame_width+10
                dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[shift,0], callback=__change, default_value="ROI")
                dpg.add_button(label="START", callback=__start_movies_and_stimuli)
                
                path = Path(ex_fn)
                with dpg.group(label="roi buttons", pos=[shift,25]) as roi: #ROI Mode Buttons
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
                    
                    
                    dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 125), wrap=150)
    
                with dpg.group(label="line buttons", pos=[shift, 25]) as line:
                    dpg.add_button(label="Vertical Line", callback=state_manager.line_interface.vertical_line_callback)
                    vert = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+104,25], callback=state_manager.line_interface.num_of_vert_lines_changer)
                    dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface.horizontal_line_callback)
                    hor = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+118,48], callback=state_manager.line_interface.num_of_hor_lines_changer)
                    dpg.add_button(label="Generate ROIs", callback=__generate_rois)
    
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.save_lines, id="line_save_file", width=700 ,height=400):
                        dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                        
                    dpg.add_button(label="Save Line Configuration", callback="line_save_file")
    
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.load_lines, id="line_load_file", width=700 ,height=400):
                        dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                    
                    dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))
    
                    dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 195), wrap=150)
    
                dpg.hide_item(line)
            
            with dpg.group(label="post line buttons", pos=[shift,0]) as post_line:
                dpg.add_button(label="Save ROIs", callback=state_manager.roi_interface.save_rois)
                dpg.add_button(label="Clear Screen and Start Over", callback=__restart)
    
            dpg.hide_item(post_line)
        
        dpg.add_image("texture_tag", pos=[8,8])
    
    dpg.set_primary_window(window, True)
    dpg.create_viewport(width=int(frame_width*1.2), height=frame_height+20, title="ROI Selector")
    dpg.setup_dearpygui()
    dpg.show_viewport()

    return roi, line, state_manager, roi_and_line_selection, post_line

roi, line, state_manager, roi_and_line_selection, post_line = setup_elements()
while dpg.is_dearpygui_running():
    cont, curr_img = vidcap.read()
    
        
    if not cont:
        _ = vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cont, curr_img = vidcap.read()
    
    data = np.flip(curr_img, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)
    
    dpg.set_value("texture_tag", texture_data)
    
    dpg.render_dearpygui_frame()
dpg.destroy_context()

