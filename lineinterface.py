#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:47:51 2025

@author: chamomile
"""
import math

import dearpygui.dearpygui as dpg

import re

class LineInterface:
    def __init__(self, frame_height, frame_width, window):
        self.lines = []
        self.multi_roi = None
        self.drag_point = None
        self.drag_line = None
        self.hover_line = None
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.last = (False, False)
        self.window = window

        self.hor_lines = 1
        self.vert_lines = 1
        
        self.hypotenuse = math.sqrt(math.pow(self.frame_width, 2) + math.pow(self.frame_height, 2))


    def __vertical_line(self):        
        for i in range(1, self.vert_lines+1):    
            pixel_val = int(i*(self.frame_width/(self.vert_lines+1)))
            point_i, point_j = [pixel_val, 0], [pixel_val, self.frame_height]
            
            l = dpg.draw_line(point_i, point_j, color=(255, 0, 0, 255), parent=self.window)
            self.lines.append(l)

    def __horizontal_line(self):        
        for i in range(1, self.hor_lines+1):    
            pixel_val = int(i*(self.frame_height/(self.hor_lines+1)))
            point_i, point_j = [0, pixel_val], [self.frame_width, pixel_val]
            
            l = dpg.draw_line(point_i, point_j, color=(255, 0, 0, 255), parent=self.window)

            self.lines.append(l)

        pass
    
    def __generate_rois(self):
        pass
    
    def __motion_notify_callback(self):
        if self.drag_line is not None:
            cursor_posn = dpg.get_mouse_pos()
            
            if (0 <= cursor_posn[0] <= self.frame_width) and (0 <= cursor_posn[1] <= self.frame_height):
                self.move_line(cursor_posn)
                    
    def __left_button_press_callback(self):
        cursor_posn = dpg.get_mouse_pos()

        if self.drag_line is None:  # we haven't selected a point to drag
            self.check_for_selection(cursor_posn)
    
    def __left_release_callback(self):
        self.drag_line = None
    
    def __num_of_vert_lines_changer(self, _, data):
        self.vert_lines = int(data) if re.search('^\d*$', data) else self.vert_lines

    def __num_of_hor_lines_changer(self, _, data):
        self.hor_lines = int(data) if re.search('^\d*$', data) else self.hor_lines
        
    def load_lines(self, lines):
        for line in lines:
            l = dpg.draw_line(line[0], line[1], color=(255, 0, 0, 255), parent=self.window)    
            self.lines.append(l)
        
    def move_line(self, cursor_posn: tuple[int,int]):
        """
        Calculates where to move the line and sets it to the value.

        Parameters
        ----------
        cursor_posn :
            

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
                dpg.configure_item(self.drag_line, p1=[cursor_posn[0], dr_y])
            else:
                dpg.configure_item(self.drag_line, p2=[cursor_posn[0], dr_y])
                
        elif moving_hor:
            dr_x = self.frame_width if dr_x > self.frame_width/2 else 0.0

            if self.drag_point == 1:
                dpg.configure_item(self.drag_line, p1=[dr_x, cursor_posn[1]])
            else:
                dpg.configure_item(self.drag_line, p2=[dr_x, cursor_posn[1]])
                
    def check_for_selection(self, cursor_posn: tuple[int,int]):
        """
        Checks if one of the vertices of the line is being selected.

        Parameters
        ----------
        cursor_posn :
            
            
        """
        for line_num in range(len(self.lines)):  # for each line on the screen
            # get x and y data points
            line = self.lines[line_num]
            config_dict = dpg.get_item_configuration(line)
            
            for v in range(1,3,1):
                x,y = config_dict["p"+str(v)]
                if math.dist((x,y), cursor_posn) < self.hypotenuse/16:
                    self.drag_line = line
                    self.drag_point = v
                                        
    def copy(): #TODO finish
        pass
    
    def delete(): #TODO finish
        pass

    def clear_all_lines(self):  #TODO finish
        pass
    