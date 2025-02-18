#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:30:38 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg
from pathlib import Path

from statemanager import StateManager

import subprocess

import numpy as np

class VisibilityManager:
    def __init__(self, filename, frame_width, frame_height):
        self.roi, self.line, self.roi_and_line_selection, self.post_line, self.state_manager = self.setup_elements(filename, frame_width, frame_height)
        
    def __change(self, e, data):
        if data == "ROI":
            dpg.show_item(self.roi)
            dpg.hide_item(self.line)
            
            self.state_manager.ROI = True
            
        elif data == "Line":
            dpg.show_item(self.line)
            dpg.hide_item(self.roi)
            self.state_manager.ROI = False
    
    def __generate_rois(self):
        dpg.hide_item(self.roi_and_line_selection)
        dpg.show_item(self.post_line)
        
        self.state_manager.generate_rois()
    
    def __restart(self):
        dpg.show_item(self.roi_and_line_selection)
        dpg.hide_item(self.post_line)
    
        self.state_manager.clear_window()
        
    def __start_movies_and_stimuli():
        print("go to arduino script here")
        subprocess.call("arduino filepath")

    
    def setup_roi_buttons(self, shift, path, state_manager):
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

        return roi
    
    def setup_line_buttons(self, shift, state_manager):
        with dpg.group(label="line buttons", pos=[shift, 25]) as line:
            dpg.add_button(label="Vertical Line", callback=state_manager.line_interface.vertical_line_callback)
            vert = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+104,25], callback=state_manager.line_interface.num_of_vert_lines_changer)
            dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface.horizontal_line_callback)
            hor = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+118,48], callback=state_manager.line_interface.num_of_hor_lines_changer)
            dpg.add_button(label="Generate ROIs", callback=self.__generate_rois)

            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.save_lines, id="line_save_file", width=700 ,height=400):
                dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                
            dpg.add_button(label="Save Line Configuration", callback="line_save_file")

            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.load_lines, id="line_load_file", width=700 ,height=400):
                dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
            
            dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))

            dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 195), wrap=150)
            
        return line
    
    def setup_post_line_buttons(self, shift, state_manager, path, frame_width, frame_height):
        with dpg.group(label="post line buttons", pos=[shift,0]) as post_line:
            curr_dir = path.parent
            curr_name = str(path.stem)
            
            with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois, id="roi_post_save_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
            
            dpg.add_button(label="Save ROIs", callback=lambda: dpg.show_item("roi_post_save_file"))

            dpg.add_button(label="Clear Screen and Start Over", callback=self.__restart)
            
            roi_slider = dpg.add_slider_double(
                label="Allowed ROI Area",
                width=100,
                min_value=0.0,
                max_value=frame_width*frame_height,
                default_value=0.0,
                callback=state_manager.roi_slider_size
            )
        return post_line
    
    #TODO fill out description
    def setup_elements(self, filename, frame_width, frame_height):
        raw_data = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

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
                dpg.add_key_press_handler(key=dpg.mvKey_W, callback=state_manager.w)
                dpg.add_key_press_handler(key=dpg.mvKey_A, callback=state_manager.a)
                dpg.add_key_press_handler(key=dpg.mvKey_S, callback=state_manager.s)
                dpg.add_key_press_handler(key=dpg.mvKey_D, callback=state_manager.d)
                
            with dpg.child_window(border=False):
                with dpg.group() as roi_and_line_selection:
                    shift = frame_width+10
                    dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[shift,0], callback=self.__change, default_value="ROI")
                    dpg.add_button(label="START", callback=self.__start_movies_and_stimuli)
                    
                    path = Path(filename)
                    roi = self.setup_roi_buttons(shift, path, state_manager)
                    line = self.setup_line_buttons(shift, state_manager)
                    dpg.hide_item(line)
                
                post_line = self.setup_post_line_buttons(shift, state_manager, path, frame_width, frame_height)
                dpg.hide_item(post_line)
            
            dpg.add_image("texture_tag", pos=[8,8])
        
        dpg.set_primary_window(window, True)
        dpg.create_viewport(width=int(frame_width+275), height=frame_height+20, title="ROI Selector")
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
        return roi, line, roi_and_line_selection, post_line, state_manager
