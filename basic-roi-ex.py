import dearpygui.dearpygui as dpg

from roipoly import RoiPoly
from roiinterface import ROIInterface

import pickle

import os

import cv2

import numpy as np

dpg.create_context()

ex_fn = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/testlog_277_ubi-kcn.avi"

vidcap = cv2.VideoCapture(ex_fn)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

raw_data = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

class LineInterface:
    def __init__(self):
        self.lines = []
        self.multi_roi = None
        self.drag_point = None
        self.drag_line = None
        self.hover_line = None
    
    def __vertical_line(self):
        pass
        
    def __horizontal_line(self):
        pass
    
    def __generate_rois(self):
        
        pass
    
    def __load_line_config(self):
        pass
    
    def __save_line_config(self):
        pass
    

class StateManager:
    def __init__(self):
        self.inactive = True
        self.current_roi = None
        self.roi_interface = ROIInterface(frame_height, frame_width)
        self.line_interface = LineInterface()
        
    def __left_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__left_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__left_button_press_callback()
            
    def __right_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__right_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__right_button_press_callback()      
            
            if self.current_roi.completed:
                self.roi_interface.rois.append(self.current_roi)
                self.inactive = True
                self.current_roi = None
                                        
    def __motion_notify_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__motion_notify_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__motion_notify_callback()
    
    def __new_roi(self):
        if self.current_roi != None and self.current_roi.completed != True:
            dpg.configure_item(self.current_roi, points=[])
                            
        self.current_roi = RoiPoly(window, frame_width, frame_height)
        self.inactive = False

    def __release(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__left_release_callback()
        
    def __change(self, e, data):
        if data == "ROI":
            dpg.show_item(roi)
            dpg.hide_item(line)
        elif data == "Line":
            dpg.show_item(line)
            dpg.hide_item(roi)
    
    def __save_rois(self):
        with open(os.getcwd() + "/cells.save", 'wb') as filename:
            pickle.dump(self.roi_interface.rois, filename)
                
    def __auto_gen_rois(self):
        pass
 
    def __callback(self, _, app_data: dict):        
        with open(app_data["file_path_name"], 'rb') as filename:
            rois = pickle.load(filename)
            
        new_rois = []
        for roi in rois:
            new_rois.append(RoiPoly(window, frame_width, frame_height, roi.lines, roi.poly))
            
        self.roi_interface.rois.extend(new_rois)


state_manager = StateManager()


with dpg.texture_registry(show=False):
    dpg.add_raw_texture(frame_width, frame_height, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
    
with dpg.theme() as canvas_theme, dpg.theme_component():
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)

with dpg.window(label="Video player", pos=(50,50), width = frame_width, height=frame_height+100) as window:

    #dpg.bind_item_theme(canvas, canvas_theme)
    #drawlist = dpg.add_drawlist(width=500, height=500)
    dpg.add_image("texture_tag")

    #with dpg.handler_registry() as registry:
    
    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=state_manager._StateManager__motion_notify_callback)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__release)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__left_button_press_callback)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=state_manager._StateManager__right_button_press_callback)
            
        #     dpg.bind_item_handler_registry(drawlist, registry)

    
    with dpg.child_window(border=False):
        dpg.add_combo(("ROI", "Line"), label="Mode", width=50, callback=state_manager._StateManager__change, default_value="ROI")
    
        with dpg.group(label="roi buttons", horizontal=True, pos=[105,0]) as roi: #ROI Mode Buttons
            dpg.add_button(label="New ROI", callback=state_manager._StateManager__new_roi)
            
            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager._StateManager__callback, id="roi_load_file", width=700 ,height=400):
                dpg.add_file_extension(".save", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                
            dpg.add_button(label="Load ROI File", callback=lambda: dpg.show_item("roi_load_file"))
    
            dpg.add_button(label="Save ROIs", callback=state_manager._StateManager__save_rois)
            dpg.add_button(label="Auto Generate ROIs", callback=state_manager._StateManager__auto_gen_rois)
        
        with dpg.group(label="line buttons", horizontal=True, pos=[105,0]) as line:
            dpg.add_button(label="Vertical Line", callback=state_manager.line_interface._LineInterface__vertical_line)
            dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface._LineInterface__horizontal_line)
            dpg.add_button(label="Generate ROIs", callback=state_manager.line_interface._LineInterface__generate_rois)
            dpg.add_button(label="Load Line Configuration", callback=state_manager.line_interface._LineInterface__load_line_config)
            dpg.add_button(label="Save Line Configuration", callback=state_manager.line_interface._LineInterface__save_line_config)
        
        dpg.hide_item(line)
            
dpg.set_primary_window(window, True)
dpg.create_viewport(width=int(frame_width*1.2), height=frame_height+100, title="ROI Selector")
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

