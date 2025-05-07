#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:24:21 2025

@author: chamomile
"""
import math
import pickle

import numpy as np
import dearpygui.dearpygui as dpg
from shapely.geometry import Point, Polygon

from roipoly import RoiPoly
from helpers import get_mouse_pos


class ROIInterface:
    """Defines useful methods for interacting with (moving, rotating) polygons."""

    def __init__(self, window, frame_width, frame_height, shift=(0, 0)):
        self.rois = []
        self.selected_polygon = None
        self.selected_polygon_vert = None
        self.prev = None
        self.drag_polygon = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.window = window
        self.allowed_area_min = 0.0
        self.allowed_area_max = self.frame_width*self.frame_height
        self.shift = shift

    def left_mouse_press_callback(self):
        """When mouse clicked, checks if current mouse position is near to a polygon or a polygon vertex."""
        x, y = get_mouse_pos(self.shift)

        if self.drag_polygon is None and self.selected_polygon_vert is None:
            self.check_for_selection((x, y))

    def motion_notify_callback(self):
        """When mouse moving, if drag polygon, moves, if selected polygon vertex, rotates."""
        x, y = get_mouse_pos(self.shift)

        if self.drag_polygon is not None:
            self.move()
        elif self.selected_polygon is not None:
            centr = Polygon(self.selected_polygon.lines).centroid

            mouse_posn_on_circle = self.find_future_posn((x, y), centr)
            theta = math.atan2(
                mouse_posn_on_circle[1]-centr.y, mouse_posn_on_circle[0]-centr.x)

            if self.prev is not None:
                self.rotate(self.prev - theta)

            self.prev = theta

    def left_mouse_release_callback(self):
        """Reset all because mouse is no longer down."""
        self.drag_polygon = None
        self.selected_polygon_vert = None
        self.selected_polygon = None
        self.prev = None

    def check_for_selection(self, mouse_pos):
        """Check if mouse down on poly or poly vertex."""
        closest = ()
        for poly in self.rois:
            mouse_pt = Point(mouse_pos)
            n_poly = Polygon(poly.lines)

            if mouse_pt.within(n_poly):
                self.drag_polygon = poly

            for point in poly.lines:
                dist = math.dist(point, mouse_pos)

                if dist < 40:  # if
                    if closest:
                        if dist < closest[1]:
                            closest = ((poly, point), dist)
                    else:
                        closest = ((poly, point), dist)

        if closest:
            self.selected_polygon = closest[0][0]
            self.selected_polygon_vert = closest[0][1]

    def check_for_hover(self):
        """Check if mouse hovering over poly or poly vertex."""
        mouse_pos = get_mouse_pos(self.shift)
        for roi in self.rois:
            mouse_pt = Point(mouse_pos)
            n_poly = Polygon(roi.lines)

            if mouse_pt.within(n_poly):
                return roi
        return None

    def copy(self):
        """Copy hovered ROI."""
        roi = self.check_for_hover()

        if roi is not None:
            new_roi = RoiPoly(self.window, self.frame_width,
                              self.frame_height, lines=roi.lines)
            self.rois.append(new_roi)

    def delete(self):
        """Delete hovered ROI."""
        roi = self.check_for_hover()

        if roi is not None:
            dpg.delete_item(roi.poly)
            self.rois.remove(roi)

    def move(self):
        """Move polygon to current mouse position."""
        x, y = get_mouse_pos(self.shift)
        poly = self.drag_polygon.lines
        centr = Polygon(poly).centroid

        translated_poly = [[p[0] - centr.x + x, p[1] - centr.y + y]
                           for p in poly]

        for point in translated_poly:
            if point[0] > self.shift[0]+self.frame_width or self.shift[0] > point[0]:
                break
            elif point[1] > self.shift[1]+self.frame_height or self.shift[1] > point[1]:
                break
        else:
            self.drag_polygon.lines = translated_poly
            dpg.configure_item(self.drag_polygon.poly,
                               points=self.drag_polygon.lines)

    def rotate(self, angle):
        """Rotate polygon to angle."""
        poly = self.selected_polygon.lines
        centr = Polygon(poly).centroid

        rot_matrix = [[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]]
        rotated_poly = np.array([(p[0]-centr.x, p[1]-centr.y) for p in poly]).dot(
            rot_matrix) + [(centr.x, centr.y) for x in range(len(poly))]

        rotated_poly = rotated_poly.tolist()

        self.selected_polygon.lines = rotated_poly
        dpg.configure_item(self.selected_polygon.poly,
                           points=self.selected_polygon.lines)

    def convert_rois_to_np_array(self, rois):
        """Converts all ROIPolys to numpy arrays."""
        lines = []
        for roi in rois:
            l = roi.lines
            poly = []

            for point in l:
                poly.append([point[0]-self.shift[0], point[1]-self.shift[1]])

            lines.append(poly)
        return lines

    def convert_np_array_to_rois(self, lines):
        """Converts numpy arrays to ROIPolys."""
        new_rois = []
        for line in lines:
            poly_lines = []

            for point in line:
                poly_lines.append(
                    [point[0]+self.shift[0], point[1]+self.shift[1]])

            new_rois.append(
                RoiPoly(self.window, self.frame_width, self.frame_height, self.shift, poly_lines))

        self.rois.extend(new_rois)

    def save_rois_callback(self, _, app_data: dict):
        """Saves all ROIs to a file."""
        with open(app_data["file_path_name"], 'wb') as filename:
            allowed_rois = []

            for roi in self.rois:
                if self.allowed_area_max > roi.area > self.allowed_area_min:
                    allowed_rois.append(roi)

            lines = self.convert_rois_to_np_array(allowed_rois)
            pickle.dump(lines, filename)

    def load_rois_callback(self, _, app_data: dict):
        """Loads all ROIs into current canvas."""
        with open(app_data["file_path_name"], 'rb') as filename:
            lines = pickle.load(filename)

        self.convert_np_array_to_rois(lines)

    def find_future_posn(self, cursor_posn: tuple[int, int], centr: tuple[int, int]) \
            -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Intermediate calculations for finding the next point the shape will
        be rotated to on the circle of positions that the currently selected
        point could go to.

        Parameters
        ----------
        cursor_posn : current mouse position
        centr : center of the polygon being rotated

        Returns
        -------
        future_v : future position on the circle of the selected vertex


        """

        # find closest point on the circle of potential vertex positions
        curr_v = (cursor_posn[0]-centr.x, cursor_posn[1]-centr.y)
        curr_mag_v = math.sqrt(
            curr_v[0]*curr_v[0] + curr_v[1]*curr_v[1])
        radius = math.dist((centr.x, centr.y), self.selected_polygon_vert)
        future_v = (centr.x + curr_v[0] / curr_mag_v * radius,
                    centr.y + curr_v[1] / curr_mag_v * radius)

        return future_v

    def introduce_dir(self, theta: int, future_v: tuple[int, int], centr: tuple[int, int]) -> int:
        """
        Chooses direction for theta.

        Parameters
        ----------
        theta : angle used for rotation
        future_v : vertex
        centr : center of the polygon being rotated

        Returns
        -------
        theta : angle used for rotation, now with pos/neg value indicating direction

        """
        d_x = (
            (self.selected_polygon_vert[0]-centr.x) - (future_v[0]-centr.x)) > 0
        d_y = (
            (self.selected_polygon_vert[1]-centr.y) - (future_v[1]-centr.y)) > 0

        if self.selected_polygon_vert[1]-centr.y > 0:
            if self.selected_polygon_vert[0]-centr.x > 0:
                if d_x and not d_y:
                    theta = -theta
            else:
                if d_x and d_y:
                    theta = -theta

        else:

            if self.selected_polygon_vert[0]-centr.x > 0:
                if not d_x and not d_y:
                    theta = -theta
            else:
                if d_y and not d_x:
                    theta = -theta

        return theta
