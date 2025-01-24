#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:24:21 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg
from shapely.geometry import Point, Polygon
import math

import numpy as np

class ROIInterface:
    def __init__(self):
        self.rois = []
        self.selected_polygon = None
        self.selected_polygon_vert = None
        self.drag_polygon = None

    def __left_button_press_callback(self):
        x,y = dpg.get_mouse_pos(local=True)

        if self.drag_polygon is None and self.selected_polygon_vert is None:
            self.check_for_selection((x,y))
        
    def __right_button_press_callback(self):
        pass
            
    def __motion_notify_callback(self):
        x,y = dpg.get_mouse_pos(local=True)

        if self.drag_polygon is not None:
            self.move()
        elif self.selected_polygon is not None:
            centr = Polygon(self.selected_polygon.lines).centroid
    
            v, future_v = self.find_future_posn((x,y), centr)
    
            if -1 <= v <= 1:
                # doing it this way means we need to reintroduce directionality / pos neg
                theta = math.acos(v)
                theta = self.introduce_dir(theta, future_v, centr)
                
                self.rotate(theta)
                        
    def __left_release_callback(self):
        self.drag_polygon = None
        self.selected_polygon_vert = None
        self.selected_polygon = None
        
    def check_for_selection(self, mouse_pos):
        for poly in self.rois:
            mouse_pt = Point(mouse_pos)
            n_poly = Polygon(poly.lines)

            if mouse_pt.within(n_poly):
                self.drag_polygon = poly
                
            for point in poly.lines:
                if math.dist(point, mouse_pos) < 40:  # if
                    self.selected_polygon_vert = point
                    self.selected_polygon = poly

    def move(self):
        x,y = dpg.get_mouse_pos(local=True)
        
        poly = self.drag_polygon.lines
        centr = Polygon(poly).centroid

        translated_poly = [[p[0] - centr.x + x, p[1] - centr.y + y] for p in poly]
        
        self.drag_polygon.lines = translated_poly
        dpg.configure_item(self.drag_polygon.poly, points=self.drag_polygon.lines)
        
    def rotate(self, angle):
        poly = self.selected_polygon.lines
        centr = Polygon(poly).centroid
        
        """
        r = dpg.create_rotation_matrix(angle, [centr.x, centr.y])
        
        dpg.apply_transform(self.selected_polygon.poly, r)
        """
        
        rot_matrix = [[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]]
        rotated_poly = np.array([(p[0]-centr.x, p[1]-centr.y) for p in poly]).dot(
            rot_matrix) + [(centr.x, centr.y) for x in range(len(poly))]
        
        rotated_poly = rotated_poly.tolist()
        
        self.selected_polygon.lines = rotated_poly
        dpg.configure_item(self.selected_polygon.poly, points=self.selected_polygon.lines)
        
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
        v : current position on the circle of the selected vertex
        future_v : future position on the circle of the selected vertex


        """


        # find closest point on the circle of potential vertex positions
        curr_v = (cursor_posn[0]-centr.x, cursor_posn[1]-centr.y)
        curr_mag_v = math.sqrt(
            curr_v[0]*curr_v[0] + curr_v[1]*curr_v[1])
        radius = math.dist((centr.x, centr.y), self.selected_polygon_vert)
        future_v = (centr.x + curr_v[0] / curr_mag_v * radius,
                    centr.y + curr_v[1] / curr_mag_v * radius)

        v = (2 * (radius**2) - math.dist(future_v,
                                         cursor_posn) ** 2) / (2*radius*radius)

        return v, future_v
    
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