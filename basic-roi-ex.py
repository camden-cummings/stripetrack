import dearpygui.dearpygui as dpg

from roipoly import RoiPoly
from roiinterface import ROIInterface
from lineinterface import LineInterface

import pickle

import cv2

import numpy as np

dpg.create_context()

ex_fn = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/testlog_277_ubi-kcn.avi"

vidcap = cv2.VideoCapture(ex_fn)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

raw_data = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

class StateManager:
    def __init__(self):
        self.inactive = True
        self.current_roi = None
        self.ROI = True
        self.roi_interface = ROIInterface(frame_height, frame_width)
        self.line_interface = LineInterface(frame_height, frame_width, window)
        
    def __left_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI:
            self.roi_interface._ROIInterface__left_button_press_callback()
        elif not self.ROI:
            self.line_interface._LineInterface__left_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__left_button_press_callback()
            
    def __right_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI:
            self.roi_interface._ROIInterface__right_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__right_button_press_callback()      
            
            if self.current_roi.completed:
                self.roi_interface.rois.append(self.current_roi)
                self.inactive = True
                self.current_roi = None
                                        
    def __motion_notify_callback(self):
        if self.inactive and self.ROI and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__motion_notify_callback()
        elif not self.ROI:
            self.line_interface._LineInterface__motion_notify_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__motion_notify_callback()
    
    def __new_roi(self):
        if self.current_roi != None and self.current_roi.completed != True:
            dpg.configure_item(self.current_roi, points=[])
                            
        self.current_roi = RoiPoly(window, frame_width, frame_height)
        self.inactive = False

    def __release(self):
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI:
            self.roi_interface._ROIInterface__left_release_callback()
        elif not self.ROI:
            self.line_interface._LineInterface__left_release_callback()
                
    def __change(self, e, data):
        if data == "ROI":
            dpg.show_item(roi)
            dpg.hide_item(line)
            
            self.ROI = True
        elif data == "Line":
            dpg.show_item(line)
            dpg.hide_item(roi)
            self.ROI = False

    def __save_rois(self):
        with open(ex_fn[:-4] + ".cells", 'wb') as filename:
            pickle.dump(self.roi_interface.rois, filename)

    def __load_rois(self, _, app_data: dict):        
        with open(app_data["file_path_name"], 'rb') as filename:
            rois = pickle.load(filename)
            
        new_rois = []
        for roi in rois:
            new_rois.append(RoiPoly(window, frame_width, frame_height, roi.lines, roi.poly))
            
        self.roi_interface.rois.extend(new_rois)
            
    def __load_line_config(self, _, app_data: dict):
        with open(app_data["file_path_name"], 'rb') as filename:
            lines = pickle.load(filename)
            self.line_interface.load_lines(lines)
    
    def __save_line_config(self):
        with open(ex_fn[:-4] + ".lines", 'wb') as filename:
            new_lines = []
            for line in self.line_interface.lines:
                config_dict = dpg.get_item_configuration(line)
                new_lines.append([config_dict["p1"], config_dict["p2"]])
            pickle.dump(new_lines, filename)    
        
    def __auto_gen_rois(self):
        pass
 

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(frame_width, frame_height, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
    
with dpg.theme() as canvas_theme, dpg.theme_component():
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)

with dpg.window(label="Video player", pos=(50,50), width = frame_width, height=frame_height) as window:
    state_manager = StateManager()

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=state_manager._StateManager__motion_notify_callback)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__release)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__left_button_press_callback)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=state_manager._StateManager__right_button_press_callback)
    
    with dpg.child_window(border=False):
        with dpg.group():
            shift = frame_width+10
            dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[shift,0], callback=state_manager._StateManager__change, default_value="ROI")
            
            with dpg.group(label="roi buttons", pos=[shift,25]) as roi: #ROI Mode Buttons
                dpg.add_button(label="New ROI", callback=state_manager._StateManager__new_roi)
                
                with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager._StateManager__load_rois, id="roi_load_file", width=700 ,height=400):
                    dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                    
                dpg.add_button(label="Load ROI File", callback=lambda: dpg.show_item("roi_load_file"))
        
                dpg.add_button(label="Save ROIs", callback=state_manager._StateManager__save_rois)
                dpg.add_button(label="Auto Generate ROIs", callback=state_manager._StateManager__auto_gen_rois)
                
                dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 125), wrap=150)

            with dpg.group(label="line buttons", pos=[shift, 25]) as line:
                dpg.add_button(label="Vertical Line", callback=state_manager.line_interface._LineInterface__vertical_line)
                vert = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+104,25], callback=state_manager.line_interface._LineInterface__num_of_vert_lines_changer)
                dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface._LineInterface__horizontal_line)
                hor = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+118,48], callback=state_manager.line_interface._LineInterface__num_of_hor_lines_changer)
                dpg.add_button(label="Generate ROIs", callback=state_manager.line_interface._LineInterface__generate_rois)
                
                dpg.add_button(label="Save Line Configuration", callback=state_manager._StateManager__save_line_config)

                with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager._StateManager__load_line_config, id="line_load_file", width=700 ,height=400):
                    dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                
                dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))

                dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 195), wrap=150)

            dpg.hide_item(line)
    
    dpg.add_image("texture_tag", pos=[8,8])

dpg.set_primary_window(window, True)
dpg.create_viewport(width=int(frame_width*1.2), height=frame_height+20, title="ROI Selector")
dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    cont, curr_img = vidcap.read()
    
    data = np.flip(curr_img, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)
    
    dpg.set_value("texture_tag", texture_data)
    
    dpg.render_dearpygui_frame()
dpg.destroy_context()

