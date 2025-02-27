#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:47:51 2025

@author: chamomile
"""
import math
import re
import pickle

import dearpygui.dearpygui as dpg

from helpers import get_mouse_pos

class LineInterface:
    """Allows manipulating of lines going from top to bottom of given window, and creation of poly based on that."""

    def __init__(self, frame_height, frame_width, window, shift=(0, 0)):
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

        self.shift = shift

        self.hypotenuse = math.sqrt(
            math.pow(self.frame_width, 2) + math.pow(self.frame_height, 2))

    def vertical_line_callback(self):
        """Makes a vertical line and adds to current window."""
        for i in range(1, self.vert_lines+1):
            pixel_val = int(i*(self.frame_width/(self.vert_lines+1)))
            point_i, point_j = [self.shift[0]+pixel_val, self.shift[1]
                                ], [self.shift[0]+pixel_val, self.shift[1]+self.frame_height]

            vert_line = dpg.draw_line(point_i, point_j, color=(
                255, 0, 0, 255), parent=self.window)
            self.lines.append(vert_line)

    def horizontal_line_callback(self):
        """Makes a horizontal line and adds to current window."""
        for i in range(1, self.hor_lines+1):
            pixel_val = int(i*(self.frame_height/(self.hor_lines+1)))
            point_i, point_j = [self.shift[0], self.shift[1]+pixel_val], [
                self.shift[0]+self.frame_width, self.shift[1]+pixel_val]
            hor_line = dpg.draw_line(point_i, point_j, color=(
                255, 0, 0, 255), parent=self.window)

            self.lines.append(hor_line)

    def motion_notify_callback(self):
        """When mouse moves and user is currently dragging a line, update line's position."""
        if self.drag_line is not None:
            mouse_posn = get_mouse_pos(self.shift)

            if (self.shift[0] <= mouse_posn[0] <= self.shift[0]+self.frame_width) and (self.shift[1] <= mouse_posn[1] <= self.shift[1]+self.frame_height):
                self.move_line(mouse_posn)

        elif self.middle_drag_line is not None:
            mouse_posn = get_mouse_pos(self.shift)

            if (self.shift[0] <= mouse_posn[0] <= self.shift[0]+self.frame_width) and (self.shift[1] <= mouse_posn[1] <= self.shift[1]+self.frame_height):
                self.move_line_by_middle(mouse_posn)

    def left_mouse_press_callback(self):
        """When user clicks, check if there's a line nearby to begin dragging."""
        mouse_posn = get_mouse_pos(self.shift)

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

    def up_callback(self, _, __):
        """Moves all lines up."""
        self.move_lines_incremented(0, 1)

    def left_callback(self, _, __):
        """Moves all lines left."""
        self.move_lines_incremented(1, 0)

    def down_callback(self, _, __):
        """Move all lines down."""
        self.move_lines_incremented(0, -1)

    def right_callback(self, _, __):
        """Move all lines right."""
        self.move_lines_incremented(-1, 0)

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

        moving_vert = dr_y < self.shift[1] + \
            5 or abs(dr_y - abs(self.frame_height+self.shift[1])) < 5
        moving_hor = dr_x < self.shift[0] + \
            5 or abs(dr_x - self.frame_width+self.shift[0]) < 5

        if moving_vert and moving_hor:
            moving_hor, moving_vert = self.last
        else:
            self.last = (moving_vert, moving_hor)

        if moving_vert:
            dr_y = self.frame_height + \
                self.shift[1] if dr_y > (
                    self.frame_height/2)+self.shift[1] else self.shift[1]

            if self.drag_point == 1:
                dpg.configure_item(self.drag_line, p1=[mouse_posn[0], dr_y])
            else:
                dpg.configure_item(self.drag_line, p2=[mouse_posn[0], dr_y])

        elif moving_hor:
            dr_x = self.frame_width + \
                self.shift[0] if dr_x > (
                    self.frame_width/2)+self.shift[0] else self.shift[0]

            if self.drag_point == 1:
                dpg.configure_item(self.drag_line, p1=[dr_x, mouse_posn[1]])
            else:
                dpg.configure_item(self.drag_line, p2=[dr_x, mouse_posn[1]])

    def move_line_by_middle(self, mouse_posn: tuple[int, int]):
        """Moves selected line around using middle point as handle."""
        # TODO adjust s.t. still works when not on opposite walls, fix movement issue where escapes bounds
        dx = self.last_mouse_posn[0] - mouse_posn[0]
        dy = self.last_mouse_posn[1] - mouse_posn[1]

        config_dict = dpg.get_item_configuration(self.middle_drag_line)
        p1 = config_dict["p1"]
        p2 = config_dict["p2"]

        # if p1[0] == 0.0 and p1[1] == self.
        if abs(dx) > abs(dy) and p1[0] != self.shift[0] and p1[0] != self.shift[0]+self.frame_width and p2[0] != self.shift[0] and p2[0] != self.shift[0]+self.frame_width:
            dpg.configure_item(self.middle_drag_line, p1=[
                               p1[0]-dx, p1[1]], p2=[p2[0]-dx, p2[1]])
        elif abs(dy) > abs(dx) and p1[1] != self.shift[1] and p1[1] != self.shift[1]+self.frame_height and p2[1] != self.shift[1] and p2[1] != self.shift[1]+self.frame_height:
            dpg.configure_item(self.middle_drag_line, p1=[
                               p1[0], p1[1]-dy], p2=[p2[0], p2[1]-dy])

        self.last_mouse_posn = mouse_posn

    def check_for_selection(self, mouse_posn: tuple[int, int]):
        """
        Checks if one of the vertices of the line is being selected.

        Parameters
        ----------
        mouse_posn :


        """
        closest = ()
        closest_middle = ()
        for line in self.lines:  # for each line on the screen
            # get x and y data points
            config_dict = dpg.get_item_configuration(line)

            all_xs = 0
            all_ys = 0
            for point_num in range(1, 3, 1):
                x, y = config_dict["p"+str(point_num)]
                all_xs += x
                all_ys += y
                dist = math.dist((x, y), mouse_posn)
                if dist < self.hypotenuse/16:
                    if closest:
                        if dist < closest[1]:
                            closest = ((line, point_num), dist)
                    else:
                        closest = ((line, point_num), dist)

            middle_point = [int(all_xs/2), int(all_ys/2)]
            mid_dist = math.dist(middle_point, mouse_posn)
            if mid_dist < self.hypotenuse/16:
                if closest_middle:
                    if mid_dist < closest_middle[1]:
                        closest_middle = ((line, mouse_posn), mid_dist)
                else:
                    closest_middle = ((line, mouse_posn), mid_dist)

        if closest:
            self.drag_line = closest[0][0]
            self.drag_point = closest[0][1]

        if closest_middle:
            self.middle_drag_line = closest_middle[0][0]
            self.last_mouse_posn = closest_middle[0][1]

    def check_for_hover(self):
        """Check if mouse hovering over poly or poly vertex."""
        mouse_posn = get_mouse_pos(self.shift)
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

    def load_lines(self, _, app_data: dict):
        """Load lines onto existing canvas."""
        with open(app_data["file_path_name"], 'rb') as filename:
            lines = pickle.load(filename)

            for line in lines:
                loaded_line = dpg.draw_line(line[0], line[1], color=(
                    255, 0, 0, 255), parent=self.window)
                self.lines.append(loaded_line)

    def save_lines(self, path):
        """Saves all current lines."""
        with open(path[:-4] + ".lines", 'wb') as filename:
            new_lines = []
            for line in self.lines:
                config_dict = dpg.get_item_configuration(line)
                new_lines.append([config_dict["p1"], config_dict["p2"]])

            pickle.dump(new_lines, filename)

    def move_lines_incremented(self, dx, dy):
        """Moves every line by dx, dy, while keeping within bounds of set canvas."""
        for line in self.lines:
            config_dict = dpg.get_item_configuration(line)

            for point_num in range(1, 3, 1):
                x, y = config_dict["p"+str(point_num)]

                if x in [0.0, self.frame_width]:
                    if point_num == 1:
                        dpg.configure_item(line, p1=[x, y-dy])
                    else:
                        dpg.configure_item(line, p2=[x, y-dy])

                elif y in [0.0, self.frame_height]:
                    if point_num == 1:
                        dpg.configure_item(line, p1=[x-dx, y])
                    else:
                        dpg.configure_item(line, p2=[x-dx, y])
