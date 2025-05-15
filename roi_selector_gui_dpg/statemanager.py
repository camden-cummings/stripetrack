#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:55:53 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg

from roi_selector_gui_dpg.roipoly import RoiPoly
from roi_selector_gui_dpg.roiinterface import ROIInterface
from roi_selector_gui_dpg.lineinterface import LineInterface
from roi_selector_gui_dpg.roi_generation import generate_rois


class StateManager:
    """Makes sure in correct state when necessary."""

    def __init__(self, window, frame_width, frame_height, shift=(0, 0)):
        self.inactive = True
        self.current_roi = None
        self.ROI_mode_selected = True
        self.ctrl_has_been_pressed = False
        
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.roi_interface = ROIInterface(window, frame_width, frame_height, shift)
        self.line_interface = LineInterface(window, frame_width, frame_height, shift)

        self.shift = shift

        self.window = window

    def left_mouse_press_callback(self):
        """When left mouse pressed, sends to appropriate method."""
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI_mode_selected:
            # selecting completed ROIs
            self.roi_interface.left_mouse_press_callback()
        elif not self.ROI_mode_selected:
            # selecting lines
            self.line_interface.left_mouse_press_callback()
        elif not self.inactive:
            # there's an ROI that has not been completed,
            # so the left mouse press means a new vertex for the ROI
            self.current_roi.left_mouse_press_callback()

    def right_mouse_press_callback(self):
        """Finishes the currently unfinished ROI, if there is one."""
        if not self.inactive:
            self.current_roi.right_mouse_press_callback()

            if self.current_roi.completed:
                self.roi_interface.rois.append(self.current_roi)
                self.inactive = True
                self.current_roi = None

    def motion_notify_callback(self):
        """When mouse is moving, sends to appropriate method."""
        if self.inactive and self.ROI_mode_selected and len(self.roi_interface.rois) > 0:
            self.roi_interface.motion_notify_callback()
        elif not self.ROI_mode_selected:
            self.line_interface.motion_notify_callback()
        elif not self.inactive:
            self.current_roi.motion_notify_callback()

    def new_roi(self):
        """Create new ROI and add to ROIInterface."""
        if self.current_roi is not None and not self.current_roi.completed:
            dpg.configure_item(self.current_roi, points=[])

        self.current_roi = RoiPoly(
            self.window, self.frame_width, self.frame_height, self.shift)
        self.inactive = False

    def release_callback(self):
        """When mouse is released, send to either ROI or line interface based on mode."""
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI_mode_selected:
            self.roi_interface.left_mouse_release_callback()
        elif not self.ROI_mode_selected:
            self.line_interface.left_mouse_release_callback()

    def generate_rois_callback(self):
        """Creates list of ROIs based on lines."""
        shortened_contours = generate_rois(
            self.line_interface.lines, self.frame_height, self.frame_width, self.shift)

        for line in self.line_interface.lines:
            dpg.delete_item(line)

        self.line_interface.lines.clear()

        self.roi_interface.rois.extend(self.make_rois(shortened_contours))
        self.ROI_mode_selected = True

    def make_rois(self, shortened_contours: list):
        """Make an ROIPoly object for each contour in shortened_contours."""
        rois = []
        for i in range(len(shortened_contours)):
            n_l = [[int(j[0]+self.shift[0]), int(j[1]+self.shift[1])]
                   for j in shortened_contours[i]]
            roi = RoiPoly(self.window, self.frame_width,
                          self.frame_height, self.shift, lines=n_l)
            roi.finish_roi()
            rois.append(roi)
        return rois

    def clear_window(self):
        """Clears window of all ROIs and lines."""
        for line in self.line_interface.lines:
            dpg.delete_item(line)

        self.line_interface.lines.clear()

        for roi in self.roi_interface.rois:
            dpg.delete_item(roi.poly)

        self.roi_interface.rois.clear()

    def roi_slider_size_callback_min(self, _, allowed_area):
        """Shows or hides ROIs based on value of allowed area."""
        self.roi_interface.allowed_area_min = allowed_area

        for roi in self.roi_interface.rois:
            if roi.area < allowed_area or roi.area > self.roi_interface.allowed_area_max:
                dpg.hide_item(roi.poly)
            else:
                dpg.show_item(roi.poly)

    def roi_slider_size_callback_max(self, _, allowed_area):
        """Shows or hides ROIs based on value of allowed area."""
        self.roi_interface.allowed_area_max = allowed_area

        for roi in self.roi_interface.rois:
            if roi.area > allowed_area or roi.area < self.roi_interface.allowed_area_min:
                dpg.hide_item(roi.poly)
            else:
                dpg.show_item(roi.poly)

    def copy_callback(self, _, __):
        """Callback from copy (ctrl+c) pressed, decides if a line or ROI should be copied."""
        if self.ROI_mode_selected and self.ctrl_has_been_pressed:
            self.roi_interface.copy()
        else:
            self.line_interface.copy()

        self.ctrl_has_been_pressed = False

    def control_callback(self, _, __):  # janky soln to key press check
        self.ctrl_has_been_pressed = True

    def delete_callback(self, _, __):
        """If delete button pressed, delete either hovered ROI or line, depending on mode."""
        if self.ROI_mode_selected:
            self.roi_interface.delete()
        else:
            self.line_interface.delete()
