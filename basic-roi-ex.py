import dearpygui.dearpygui as dpg

from roipoly import RoiPoly
from roiinterface import ROIInterface
from lineinterface import LineInterface

import pickle

import cv2

import numpy as np

import math

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
        self.ctrl_has_been_pressed = False
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
            new_rois.append(RoiPoly(window, frame_width, frame_height, roi.lines))
            
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
 
    def __generate_rois(self):
        """Creates ROIs from set of lines."""
        lines = []

        # four corners
        corners = [[0, 0], 
                   [0, int(frame_height)], 
                   [int(frame_width), 0], 
                   [int(frame_width), int(frame_height)]]

        for c1 in corners:
            for c2 in corners:
                if c1 != c2 and (c1[0] == c2[0] or c1[1] == c2[1]) and [c2, c1] not in lines: # all possible borders of screen
                    lines.append([tuple(c1), tuple(c2)])
        
        for line in self.line_interface.lines:
            config_dict = dpg.get_item_configuration(line)
            p1 = [int(i) for i in config_dict["p1"]]
            p2 = [int(i) for i in config_dict["p2"]]
            lines.append([tuple(p1), tuple(p2)])
        
        img = np.zeros((int(frame_height),
                       int(frame_width), 3), dtype=np.uint8)

        for line in lines:
            point_1, point_2 = line
            cv2.line(img, point_1, point_2, (255, 255, 255), 3)

        contours, _ = cv2.findContours(
            np.array(img)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shortened_contours = self.trim(contours)
        self.roi_interface.rois = self.make_rois(shortened_contours)

        for line in self.line_interface.lines:
            dpg.delete_item(line)
    
        self.line_interface.lines.clear()
        
        dpg.hide_item(roi_and_line_selection)
        dpg.show_item(post_line)
        self.ROI=True
        
    def make_rois(self, shortened_contours):
        """Make ROIs."""
        rois = []
        for i in range(len(shortened_contours)):
            n_l = [[int(j[0]), int(j[1])] for j in shortened_contours[i]]
            roi = RoiPoly(window, frame_width, frame_height, lines=n_l)
            rois.append(roi)
        return rois

    def trim(self, contours):
        """
        Trim.

        Parameters
        ----------
        contours : TYPE
            DESCRIPTION.

        Returns
        -------
        shortened_contours : TYPE
            DESCRIPTION.

        """
        shortened_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_height*frame_width*0.995:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

                M = cv2.moments(contour)

                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])

                single = []

                for point in approx:
                    exp_cx = point[0][0] + 1 * \
                        (centroid_x - point[0][0]) / \
                        math.dist(point[0], (centroid_x, centroid_y))
                    exp_cy = point[0][1] + 1 * \
                        (centroid_y - point[0][1]) / \
                        math.dist(point[0], (centroid_x, centroid_y))
                    single.append((exp_cx, exp_cy))
                    
                shortened_contours.append(np.array(single))

        return shortened_contours

    def __restart(self):
        dpg.show_item(roi_and_line_selection)
        dpg.hide_item(post_line)

        self.clear_screen()
        
    def clear_screen(self):    
        for line in self.line_interface.lines:
            dpg.delete_item(line)
        
        self.line_interface.lines.clear()

        for roi in self.roi_interface.rois:
            dpg.delete_item(roi.poly)
        
        self.roi_interface.rois.clear()

    def __copy(self, _, app_data):            
        if self.ROI and self.ctrl_has_been_pressed:
            
            roi = self.roi_interface.check_for_hover()
            if roi is not None:
                new_roi = RoiPoly(window, frame_width, frame_height, lines=roi.lines)
                self.roi_interface.rois.append(new_roi)
        else:
            self.line_interface.copy()
            
        self.ctrl_has_been_pressed=False
        
    def __control(self, _, __): # janky soln to key press check
        self.ctrl_has_been_pressed = True
        
    def __delete(self, _, appdata):
        if self.ROI:
            roi = self.roi_interface.check_for_hover()
            dpg.delete_item(roi.poly)
            self.roi_interface.rois.remove(roi)

        else:
            self.line_interface.delete()
        
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
        dpg.add_key_press_handler(key=dpg.mvKey_C, callback=state_manager._StateManager__copy)
        dpg.add_key_press_handler(key=dpg.mvKey_LControl, callback=state_manager._StateManager__control)
        dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=state_manager._StateManager__delete)
        
    with dpg.child_window(border=False):
        with dpg.group() as roi_and_line_selection:
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
                dpg.add_button(label="Generate ROIs", callback=state_manager._StateManager__generate_rois)
                
                dpg.add_button(label="Save Line Configuration", callback=state_manager._StateManager__save_line_config)

                with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager._StateManager__load_line_config, id="line_load_file", width=700 ,height=400):
                    dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                
                dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))

                dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 195), wrap=150)

            dpg.hide_item(line)
        
        with dpg.group(label="post line buttons", pos=[shift,0]) as post_line:
            dpg.add_button(label="Save ROIs", callback=state_manager._StateManager__save_rois)
            dpg.add_button(label="Clear Screen and Start Over", callback=state_manager._StateManager__restart)

        dpg.hide_item(post_line)
    
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

