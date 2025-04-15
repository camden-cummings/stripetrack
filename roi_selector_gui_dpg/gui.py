#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:30:38 2025

@author: chamomile
"""
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from helpers import update_frame_shape, get_shape
from statemanager import StateManager

import cv2

class GUI:
    """Sets and manages GUI elements."""

    def __init__(self, filename: str, window, vidcap, frame_width, frame_height):
        ext = filename.split('.')[-1] 
        
        if vidcap is not None:
            self.vidcap = cv2.VideoCapture(filename)
            
            self.frame_width = frame_width
            self.frame_height = frame_height
                    
        self.roi, self.line, self.roi_and_line_selection, self.post_line, self.state_manager = self.setup_elements(filename, window)

    def change_selection_mode(self, _, data):
        """Changes set of buttons based on mode selected."""
        if data == "ROI":
            dpg.show_item(self.roi)
            dpg.hide_item(self.line)

            self.state_manager.ROI_mode_selected = True

        elif data == "Line":
            dpg.show_item(self.line)
            dpg.hide_item(self.roi)
            self.state_manager.ROI_mode_selected = False

    def generate_rois(self):
        """Generates ROIs and changes mode to show post-line generation GUI."""
        dpg.hide_item(self.roi_and_line_selection)
        dpg.show_item(self.post_line)

        self.state_manager.generate_rois_callback()

    def restart(self):
        """Resets everything to original."""
        dpg.show_item(self.roi_and_line_selection)
        dpg.hide_item(self.post_line)
        self.state_manager.ROI_mode_selected = False

        self.state_manager.clear_window()

    def open_file_callback(self, _, appdata: dict):
        self.vidcap = cv2.VideoCapture(appdata["file_path_name"])
        
        self.frame_width = int(self.vidcap.get(3))
        self.frame_height = int(self.vidcap.get(4))
        
        update_frame_shape(self.state_manager, self.frame_width, self.frame_height)
        update_frame_shape(self.state_manager.roi_interface, self.frame_width, self.frame_height)
        update_frame_shape(self.state_manager.line_interface, self.frame_width, self.frame_height)

        #file_path_name = appdata["file_path_name"]
        
    @staticmethod
    def setup_roi_buttons(shift, down_shift, curr_dir, curr_name, state_manager):
        """ROI Mode Buttons."""
        with dpg.group(label="roi buttons", pos=[shift, down_shift]) as roi:
            dpg.add_button(label="New ROI", callback=state_manager.new_roi)

            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.load_rois_callback,
                                 id="roi_load_file", width=700, height=400, default_path=curr_dir, default_filename=curr_name):
                dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")

            dpg.add_button(label="Load ROI File",
                           callback=lambda: dpg.show_item("roi_load_file"))

            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois_callback,
                                 id="roi_save_file", width=700, height=400, default_path=curr_dir, default_filename=curr_name):
                dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")

            dpg.add_button(label="Save ROIs",
                           callback=lambda: dpg.show_item("roi_save_file"))

            dpg.add_button(label="Clear Screen and Start Over", pos=[
                           shift, down_shift+95], callback=state_manager.clear_window)
            dpg.add_text(
                "NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete",
                pos=(shift+5, down_shift+115), wrap=150)

        return roi

    def setup_line_buttons(self, shift, down_shift, state_manager):
        """Line Mode Buttons."""
        with dpg.group(label="line buttons", pos=[shift, down_shift]) as line:
            dpg.add_button(
                label="Vertical Line", callback=state_manager.line_interface.vertical_line_callback)
            dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[
                shift+104, down_shift], callback=state_manager.line_interface.num_of_vert_lines_changer)
            dpg.add_button(label="Horizontal Line",
                           callback=state_manager.line_interface.horizontal_line_callback)
            dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[
                shift+118, down_shift+23], callback=state_manager.line_interface.num_of_hor_lines_changer)
            dpg.add_button(label="Generate ROIs", callback=self.generate_rois)

            with dpg.file_dialog(directory_selector=False, show=False, 
                                 callback=state_manager.line_interface.save_lines, id="line_save_file", width=700, height=400):
                dpg.add_file_extension(".lines", color=(
                    0, 255, 0, 255), custom_text="[Line Save File]")

            dpg.add_button(label="Save Line Configuration",
                           callback="line_save_file")

            with dpg.file_dialog(directory_selector=False, show=False, 
                                 callback=state_manager.line_interface.load_lines, id="line_load_file", width=700, height=400):
                dpg.add_file_extension(".lines", color=(
                    0, 255, 0, 255), custom_text="[Line Save File]")

            dpg.add_button(label="Load Line Configuration",
                           callback=lambda: dpg.show_item("line_load_file"))

            dpg.add_button(label="Clear Screen and Start Over", pos=[
                           shift, down_shift+123], callback=state_manager.clear_window)
            dpg.add_text(
                "NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete \n WASD: move all lines", 
                pos=(shift+5, down_shift+140), wrap=150)

        return line

    def setup_post_line_buttons(self, shift, down_shift, state_manager, curr_dir, curr_name):
        """Post Line ROI Generation Buttons."""
        with dpg.group(label="post line buttons", pos=[shift, down_shift]) as post_line:
            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois_callback, id="roi_post_save_file", width=700, height=400, default_path=curr_dir, default_filename=curr_name):
                dpg.add_file_extension(".cells", color=(
                    0, 255, 0, 255), custom_text="[ROI Save File]")

            dpg.add_button(label="Save ROIs",
                           callback=lambda: dpg.show_item("roi_post_save_file"))

            dpg.add_button(label="Clear Screen and Start Over",
                           callback=self.restart)

            dpg.add_slider_double(
                height=150,
                width=50,
                pos=[shift+10, down_shift+50],
                vertical=True,
                min_value=0.0,
                max_value=self.frame_width*self.frame_height,
                default_value=0.0,
                callback=state_manager.roi_slider_size_callback_min
            )

            dpg.add_slider_double(
                height=150,
                width=50,
                pos=[shift+70, down_shift+50],
                vertical=True,
                min_value=0.0,
                max_value=self.frame_width*self.frame_height,
                default_value=self.frame_width*self.frame_height,
                callback=state_manager.roi_slider_size_callback_max
            )

            dpg.add_text("MIN SIZE", pos=(shift+10, down_shift+210))
            dpg.add_text("MAX SIZE", pos=(shift+70, down_shift+210))
        return post_line

    @staticmethod
    def setup_keypress(state_manager):
        """Keypresses and mouse handlers."""
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(
                callback=state_manager.motion_notify_callback)
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Left, callback=state_manager.release_callback)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left, callback=state_manager.left_mouse_press_callback)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right, callback=state_manager.right_mouse_press_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_C, callback=state_manager.copy_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_LControl, callback=state_manager.control_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_Delete, callback=state_manager.delete_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_W, callback=state_manager.line_interface.up_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_A, callback=state_manager.line_interface.left_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_S, callback=state_manager.line_interface.down_callback)
            dpg.add_key_press_handler(
                key=dpg.mvKey_D, callback=state_manager.line_interface.right_callback)

    def setup_elements(self, filename, window):
        state_manager = StateManager(window, self.frame_width, self.frame_height)
        self.setup_keypress(state_manager)
        
        with dpg.child_window(border=False, parent=window):
            with dpg.group() as roi_and_line_selection:
                path = Path(filename)
                curr_dir = path.parent
                curr_name = str(path.stem)

                shift = self.frame_width+10
                
                with dpg.file_dialog(directory_selector=False, show=False, callback=self.open_file_callback, id="open_file", width=800, height=400, default_path=curr_dir, default_filename=curr_name):
                    for vid_ext in [".mp4", ".avi"]:
                        dpg.add_file_extension(vid_ext, color=(
                            0, 255, 0, 255), custom_text="[Video File]")
                    for img_ext in [".png"]:
                        dpg.add_file_extension(vid_ext, color=(
                            0, 255, 0, 255), custom_text="[Image File]")                            
                          
                dpg.add_button(label="Choose Video/Image", callback=lambda: dpg.show_item("open_file"), pos=[shift, 2])

                dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[
                              shift, 27], callback=self.change_selection_mode, default_value="ROI")

                roi = self.setup_roi_buttons(
                    shift, 50, curr_dir, curr_name, state_manager)
                line = self.setup_line_buttons(shift, 50, state_manager)
                dpg.hide_item(line)

            post_line = self.setup_post_line_buttons(shift, 0, state_manager, curr_dir, curr_name)
            dpg.hide_item(post_line)

        return roi, line, roi_and_line_selection, post_line, state_manager

    def start(self, window):
        raw_data = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.frame_width, self.frame_height, raw_data,
                                format=dpg.mvFormat_Float_rgb, tag="texture_tag")
    
        with dpg.theme(), dpg.theme_component():
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
        
        dpg.add_image("texture_tag", pos=[8, 8], parent=window)
        
        dpg.set_primary_window(window, True)
    
        dpg.create_viewport(width=int(self.frame_width+275),
                            height=self.frame_height+20, title="ROI Selector")
    
        dpg.setup_dearpygui()
        dpg.show_viewport()
            
        while dpg.is_dearpygui_running():
            if self.vidcap is not None:
                cont, curr_img = self.vidcap.read()
    
                if not cont:
                    _ = self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cont, curr_img = self.vidcap.read()
            
            data = np.flip(curr_img, 2)
            data = data.ravel()
            data = np.asfarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
            
            dpg.set_value("texture_tag", texture_data)
            
            dpg.render_dearpygui_frame()