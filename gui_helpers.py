# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:57:57 2025

@author: ThymeLab
"""
from pathlib import Path
from roi_selector_gui_dpg.statemanager import StateManager
from roi_selector_gui_dpg.gui import GUI
import dearpygui.dearpygui as dpg
import os
from contour_definer import ContourDefiner
from tracker.roi_manip import convert_to_contours
import numpy as np

# TODO make play nice with newly redone GUI

class GUIHelpers(GUI):
    """"""

    def __init__(self, window, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.roi, self.line, self.roi_and_line_selection, self.post_line, self.state_manager, self.status = self.setup_elements(
            window)

        self.contour_definer = ContourDefiner()

        # TODO: check how being used actually
        # min and max allowed centroid area to be considered a potential fish
        self.min_area = 40
        self.max_area = 300

        # min length of any contour to be considered a potential fish
        self.length_req = 40

        self.cell_contours = []
        self.cell_centers = []
        self.shape_of_rows = []
        
        self.show_only_inside_contours = False
        self.contour_overlay = False
        self.contours_updated = False
        self.start_recording = False
        self.mode_calculated = False


    def start_recording_callback(self, _, __):
        # TODO check here if GUI save file exists
        if self.mode_calculated:
            self.start_recording = True

    def set_cells(self, _, appdata):
        self.cell_contours, contour_mask, self.cell_centers, self.shape_of_rows = convert_to_contours(appdata["filepathname"],
                                                                                       self.frame_width,
                                                                                       self.frame_height)

        self.contours_updated = True

    def tab_callback(self, _, tab_id):
        match dpg.get_item_configuration(tab_id)["label"]:
            case "ROI Selection":
                self.contour_overlay = False
            case "Contour Overlay":
                self.contour_overlay = True
                self.contours_updated = True

                print("Contour Overlay")
                self.cell_contours, contour_mask, self.cell_centers, self.shape_of_rows = convert_to_contours(
                    self.state_manager.roi_interface.convert_rois_to_np_array(self.state_manager.roi_interface.rois),
                    self.frame_width, self.frame_height)

    def only_selected_contours(self, _, show_only_inside_contours):
        self.show_only_inside_contours = show_only_inside_contours

    def setup_elements(self, window):
        raw_data = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.frame_width, self.frame_height, raw_data, format=dpg.mvFormat_Float_rgb,
                                tag="texture_tag")

        with dpg.theme() as canvas_theme, dpg.theme_component():
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)

        right_shift = self.frame_width+10
        std_shift = 8

        #        with dpg.window(label="Video player", pos=(0,0), width = self.FRAME_WIDTH, height = self.FRAME_HEIGHT+150) as window:
        with dpg.tab_bar(label="Select", callback=self.tab_callback, parent=window):
            with dpg.tab(label='ROI Selection'):
                state_manager = StateManager(window, self.frame_width, self.frame_height, shift=(0, 23))
                #horizontal_button_posn = self.frame_width + 10
                self.setup_keypress(state_manager)

                with dpg.child_window(border=False):
                    with dpg.group() as roi_and_line_selection:
                        dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[right_shift, 0],
                                      callback=self.change_selection_mode, default_value="ROI")

                        path = Path(os.getcwd())
                        curr_dir = path.parent
                        curr_name = str(path.stem)

                        shift = self.frame_width + 10

                        roi = self.setup_roi_buttons(shift, 50, curr_dir, curr_name, state_manager)
                        line = self.setup_line_buttons(shift, 50, state_manager)

                        dpg.hide_item(line)

                    post_line = self.setup_post_line_buttons(shift, 0, state_manager, curr_dir, curr_name, self.frame_width*self.frame_height)

                    dpg.hide_item(post_line)

                dpg.add_image("texture_tag", pos=(std_shift, std_shift + 23))
            with dpg.tab(label='Contour Overlay'):
                dpg.add_image("texture_tag")

                with dpg.group(pos=[right_shift, std_shift + 23]):
                    with dpg.file_dialog(directory_selector=False, show=False, callback=self.set_cells, id="set_rois",
                                         width=0, height=0):
                        dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")

                    with dpg.tree_node(label="Computer Vision Options", default_open=True):
                        dpg.add_combo(("No Contours", "Structural Similarity", "Real Time"),
                                      label="Contour Detecting Algorithm", callback=self.contour_definer.cv_alg_change,
                                      default_value="No Contours", width=180)
                        with dpg.group(width=300):
                            dpg.add_slider_float(label="Threshold", callback=self.contour_definer.threshold_change,
                                                 min_value=0, max_value=255, default_value=self.contour_definer.thresh)
                            dpg.add_slider_float(label="Centroid Size", callback=self.contour_definer.centroid_change,
                                                 max_value=1000, default_value=self.contour_definer.centroid_size)
                        status = dpg.add_text("Status: Calculating Mode / Not Ready")

                        dpg.add_checkbox(label="Only Show ROIs Inside Selected Contours",
                                         callback=self.only_selected_contours)

                    dpg.add_button(label="START RECORDING", callback=self.start_recording_callback)

        return roi, line, roi_and_line_selection, post_line, state_manager, status
