#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:30:38 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg

class GUI:
    """Sets and manages GUI elements - set up for line interface and ROI interface. """

    def __init__(self, window, frame_width=500, frame_height=500): #TODO investigate if this works for basic ROI
        self.roi, self.line, self.roi_and_line_selection, self.post_line, self.state_manager = self.setup_elements(window)
        self.frame_width = frame_width
        self.frame_height = frame_height

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

    def setup_elements(self, window):
        return None, None, None, None, None