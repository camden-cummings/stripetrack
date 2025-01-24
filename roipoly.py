#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:22:20 2025

@author: chamomile
"""
import dearpygui.dearpygui as dpg

class RoiPoly:
    def __init__(self, window):
        self.line = None
        self.lines = []
        self.poly = None
        
        self.completed = False
        
        self.previous_point = []
        
        self.window = window
        
    def __left_button_press_callback(self):
        x,y = dpg.get_mouse_pos(local=True)
        
        if self.completed is False:
            if self.line == None:
                self.line = [x,y], [x,y]
                
                [self.lines.append(i) for i in self.line]
                
                self.previous_point = [x, y]

                self.poly = dpg.draw_polyline(points=self.lines, color=(255, 0, 0, 255), thickness=1, parent=self.window)
    
            else:
                self.line = self.previous_point, [x,y]
                
                self.lines.append([x,y])
                self.previous_point = [x, y]
                
                dpg.configure_item(self.poly, points=self.lines)

    def __right_button_press_callback(self):
        self.lines = self.lines[:-1]
        dpg.configure_item(self.poly, points=self.lines, closed=True)
        self.completed = True

    def __motion_notify_callback(self):
        if self.line is not None and self.completed is False:
            x,y = dpg.get_mouse_pos(local=True)
    
            self.line = self.previous_point, [x,y]
            self.lines[-1] = [x,y]
            
            dpg.configure_item(self.poly, points=self.lines)
