#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:47:51 2025

@author: chamomile
"""
import math
import re
import pickle

import cv2
import numpy as np
import dearpygui.dearpygui as dpg


class LineInterface:
    """Allows manipulating of lines going from top to bottom of given window, and creation of poly based on that."""

    def __init__(self, frame_height, frame_width, window):
        self.lines = []
        self.drag_point = None
        self.drag_line = None
        self.middle_drag_line = None
        
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.window = window

        self.last = (False, False)
        self.last_mouse_posn = None
        
        self.hor_lines = 1
        self.vert_lines = 1

        self.hypotenuse = math.sqrt(
            math.pow(self.frame_width, 2) + math.pow(self.frame_height, 2))

    def vertical_line_callback(self):
        """Makes a vertical line and adds to current window."""
        for i in range(1, self.vert_lines+1):
            pixel_val = int(i*(self.frame_width/(self.vert_lines+1)))
            point_i, point_j = [pixel_val, 0], [pixel_val, self.frame_height]

            vert_line = dpg.draw_line(point_i, point_j, color=(
                255, 0, 0, 255), parent=self.window)
            self.lines.append(vert_line)

    def horizontal_line_callback(self):
        """Makes a horizontal line and adds to current window."""
        for i in range(1, self.hor_lines+1):
            pixel_val = int(i*(self.frame_height/(self.hor_lines+1)))
            point_i, point_j = [0, pixel_val], [self.frame_width, pixel_val]

            hor_line = dpg.draw_line(point_i, point_j, color=(
                255, 0, 0, 255), parent=self.window)

            self.lines.append(hor_line)

    def motion_notify_callback(self):
        """When mouse moves and user is currently dragging a line, update line's position."""
        if self.drag_line is not None:
            mouse_posn = dpg.get_mouse_pos()

            if (0 <= mouse_posn[0] <= self.frame_width) and (0 <= mouse_posn[1] <= self.frame_height):
                self.move_line(mouse_posn)
        
        elif self.middle_drag_line is not None:
            mouse_posn = dpg.get_mouse_pos()

            if (0 <= mouse_posn[0] <= self.frame_width) and (0 <= mouse_posn[1] <= self.frame_height):
                self.move_line_by_middle(mouse_posn)
        
    def left_mouse_press_callback(self):
        """When user clicks, check if there's a line nearby to begin dragging."""
        mouse_posn = dpg.get_mouse_pos()

        if self.drag_line is None and self.middle_drag_line is None:  # we haven't selected a point to drag
            self.check_for_selection(mouse_posn)

    def left_mouse_release_callback(self):
        """When mouse is released, no longer dragging line."""
        self.drag_line = None
        self.middle_drag_line = None

    def num_of_vert_lines_changer(self, _, data):
        """Keep desired number of vertical lines updated as it is changed in text box."""
        self.vert_lines = int(data) if re.search(
            r'^\d*$', data) and len(data) > 0 else self.vert_lines

    def num_of_hor_lines_changer(self, _, data):
        """Keep desired number of vertical lines updated as it is changed in text box."""
        self.hor_lines = int(data) if re.search(
            r'^\d*$', data) and len(data) > 0 else self.hor_lines
        
    def move_line(self, mouse_posn: tuple[int, int]):
        """
        Calculates where to move the line and sets it to the value.

        Parameters
        ----------
        mouse_posn :

        """
        config_dict = dpg.get_item_configuration(self.drag_line)

        dr_x = config_dict["p"+str(self.drag_point)][0]
        dr_y = config_dict["p"+str(self.drag_point)][1]

        moving_vert = dr_y < 5 or abs(dr_y - abs(self.frame_height)) < 5
        moving_hor = dr_x < 5 or abs(dr_x - self.frame_width) < 5

        if moving_vert and moving_hor:
            moving_hor, moving_vert = self.last
        else:
            self.last = (moving_vert, moving_hor)

        if moving_vert:
            dr_y = self.frame_height if dr_y > self.frame_height/2 else 0.0

            if self.drag_point == 1:
                dpg.configure_item(self.drag_line, p1=[mouse_posn[0], dr_y])
            else:
                dpg.configure_item(self.drag_line, p2=[mouse_posn[0], dr_y])

        elif moving_hor:
            dr_x = self.frame_width if dr_x > self.frame_width/2 else 0.0

            if self.drag_point == 1:
                dpg.configure_item(self.drag_line, p1=[dr_x, mouse_posn[1]])
            else:
                dpg.configure_item(self.drag_line, p2=[dr_x, mouse_posn[1]])
                
    def move_line_by_middle(self, mouse_posn: tuple[int, int]):
        """Moves selected line around using middle point as handle."""
        dx = self.last_mouse_posn[0] - mouse_posn[0]
        dy = self.last_mouse_posn[1] - mouse_posn[1]

        config_dict = dpg.get_item_configuration(self.middle_drag_line)
        p1 = config_dict["p1"]
        p2 = config_dict["p2"]
        
        #if p1[0] == 0.0 and p1[1] == self.
        if abs(dx) > abs(dy) and p1[0] != 0.0 and p1[0] != self.frame_width and p2[0] != 0.0 and p2[0] != self.frame_width:
            dpg.configure_item(self.middle_drag_line, p1=[p1[0]-dx, p1[1]], p2 = [p2[0]-dx, p2[1]])
        elif abs(dy) > abs(dx) and p1[1] != 0.0 and p1[1] != self.frame_height and p2[1] != 0.0 and p2[1] != self.frame_height:
            dpg.configure_item(self.middle_drag_line, p1=[p1[0], p1[1]-dy], p2 = [p2[0], p2[1]-dy])

        self.last_mouse_posn = mouse_posn
        
    
    def check_for_selection(self, mouse_posn: tuple[int, int]):
        """
        Checks if one of the vertices of the line is being selected.

        Parameters
        ----------
        mouse_posn :


        """
        closest = ()
        for line in self.lines:  # for each line on the screen
            # get x and y data points
            config_dict = dpg.get_item_configuration(line)
            
            x_ = 0
            y_ = 0
            for point_num in range(1, 3, 1):
                x, y = config_dict["p"+str(point_num)]
                x_ += x
                y_ += y
                dist = math.dist((x, y), mouse_posn)
                if dist < self.hypotenuse/16:
                    if closest != ():
                        if dist < closest[1]:
                            closest = ((line, point_num), dist)
                    else:
                        closest = ((line, point_num), dist)
                        
            middle_point = [int(x_/2), int(y_/2)]
            
            if math.dist(middle_point, mouse_posn) < self.hypotenuse/16:
                self.middle_drag_line = line
                self.last_mouse_posn = mouse_posn


        if closest != ():
            self.drag_line = closest[0][0]
            self.drag_point = closest[0][1]
    
    def check_for_hover(self):
        """Check if mouse hovering over poly or poly vertex."""
        mouse_posn = dpg.get_mouse_pos()
        points = []
        line_num = -1
        for line in self.lines:
            config_dict = dpg.get_item_configuration(line)
            p1 = config_dict["p1"]
            p2 = config_dict["p2"]

            line_centr = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

            if math.dist(line_centr, mouse_posn) < self.hypotenuse/16:
                points = [p1, p2]
                line_num = line

        return points, line_num

    def copy(self):
        """Copy line being hovered over."""
        points, _ = self.check_for_hover()
        if len(points) > 0:
            copied_line = dpg.draw_line(points[0], points[1], color=(
                255, 0, 0, 255), parent=self.window)
            self.lines.append(copied_line)

    def delete(self):
        """Delete line being hovered over."""
        _, hovered_line = self.check_for_hover()
        if hovered_line != -1:  # if no line being hovered, we can't delete
            dpg.delete_item(hovered_line)
            self.lines.remove(hovered_line)

    def undo(self): #TODO: complete
        """Undo last action taken."""
        pass
    
    def load_lines(self, _, app_data: dict):
        with open(app_data["file_path_name"], 'rb') as filename:
            lines = pickle.load(filename)
            
            for line in lines:
                loaded_line = dpg.draw_line(line[0], line[1], color=(
                    255, 0, 0, 255), parent=self.window)
                self.lines.append(loaded_line)

    def save_lines(self, path):
        with open(path[:-4] + ".lines", 'wb') as filename:
            new_lines = []
            for line in self.lines:
                config_dict = dpg.get_item_configuration(line)
                new_lines.append([config_dict["p1"], config_dict["p2"]])
                
            pickle.dump(new_lines, filename)
            
    def generate_rois(self):
        """Creates ROIs from set of lines."""
        lines = []

        # four corners
        corners = [[0, 0],
                   [0, int(self.frame_height)],
                   [int(self.frame_width), 0],
                   [int(self.frame_width), int(self.frame_height)]]

        for c1 in corners:
            for c2 in corners:
                # all possible borders of screen
                if c1 != c2 and (c1[0] == c2[0] or c1[1] == c2[1]) and [c2, c1] not in lines:
                    lines.append([tuple(c1), tuple(c2)])

        for line in self.lines:
            config_dict = dpg.get_item_configuration(line)
            p1 = [int(i) for i in config_dict["p1"]]
            p2 = [int(i) for i in config_dict["p2"]]
            lines.append([tuple(p1), tuple(p2)])

        img = np.zeros((int(self.frame_height),
                       int(self.frame_width), 3), dtype=np.uint8)

        for line in lines:
            point_1, point_2 = line
            cv2.line(img, point_1, point_2, (255, 255, 255), 3)

        contours, _ = cv2.findContours(
            np.array(img)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shortened_contours = self.trim(contours)

        for line in self.lines:
            dpg.delete_item(line)

        self.lines.clear()

        return shortened_contours

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
            if area < self.frame_height*self.frame_width*0.995:
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
