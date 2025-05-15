#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:22:20 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg

from shapely.geometry import Polygon

from roi_selector_gui_dpg.helpers import get_mouse_pos


class RoiPoly:
    """Defines point-by-point selected polygon."""

    def __init__(self, window, frame_width, frame_height, shift=(0, 0), lines=None):
        self.line = None
        self.lines = []
        self.poly = None

        self.area = None

        self.completed = False

        self.previous_point = []

        self.window = window

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.shift = shift

        if lines is not None:
            self.lines = lines
            self.completed = True
            self.finish_roi()
            self.poly = dpg.draw_polyline(points=self.lines, color=(
                255, 0, 0, 255), thickness=1, parent=self.window, closed=True)

    def finish_roi(self):
        """Last steps to finish ROI."""
        self.area = Polygon(self.lines).area

    def left_mouse_press_callback(self):
        """Add new vertex to the polygon."""
        x, y = get_mouse_pos(self.shift)
        if self.completed is False and self.shift[0] <= x <= self.shift[0]+self.frame_width and self.shift[1] <= y <= self.shift[1]+self.frame_height:
            if self.line is None:
                self.line = [x, y], [x, y]

                self.lines.extend(self.line)

                self.previous_point = [x, y]

                self.poly = dpg.draw_polyline(points=self.lines, color=(
                    255, 0, 0, 255), thickness=1, parent=self.window)

            else:
                self.line = self.previous_point, [x, y]

                self.lines.append([x, y])
                self.previous_point = [x, y]

                dpg.configure_item(self.poly, points=self.lines)

    def right_mouse_press_callback(self):
        """Complete polygon if right mouse is clicked and there are enough points to make one."""
        if len(self.lines) > 3:
            print("right press")
            self.lines = self.lines[:-1]
            dpg.configure_item(self.poly, points=self.lines, closed=True)
            self.finish_roi()
            print(self.area)
            self.completed = True

    def motion_notify_callback(self):
        """When mouse is moving, update position of currently selected line."""
        if self.line is not None and self.completed is False:
            x, y = get_mouse_pos(self.shift)

            self.line = self.previous_point, [
                max(min(x, self.shift[0]), self.shift[0]+self.frame_width), max(min(y, self.shift[1]), self.shift[1]+self.frame_height)]
            self.lines[-1] = [min(max(x, self.shift[0]), self.shift[0]+self.frame_width),
                              min(max(y, self.shift[1]), self.shift[1]+self.frame_height)]

            dpg.configure_item(self.poly, points=self.lines)
