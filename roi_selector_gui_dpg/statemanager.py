#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:55:53 2025

@author: chamomile
"""
import pickle

import math

import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from roipoly import RoiPoly
from roiinterface import ROIInterface
from lineinterface import LineInterface

#TODO clean up statemanager
class StateManager:
    def __init__(self, frame_width, frame_height, window):
        self.inactive = True
        self.current_roi = None
        self.ROI = True
        self.ctrl_has_been_pressed = False
        
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.roi_interface = ROIInterface(frame_height, frame_width)
        self.line_interface = LineInterface(frame_height, frame_width, window)
        
        self.window = window
        
    def left_mouse_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI:
            self.roi_interface.left_mouse_press_callback()
        elif not self.ROI:
            self.line_interface.left_mouse_press_callback()
        elif not self.inactive:
            self.current_roi.left_mouse_press_callback()
            
    def right_mouse_press_callback(self):
        if not self.inactive:
            self.current_roi.right_mouse_press_callback()      
            
            if self.current_roi.completed:
                self.roi_interface.rois.append(self.current_roi)
                self.inactive = True
                self.current_roi = None
                                        
    def motion_notify_callback(self):
        if self.inactive and self.ROI and len(self.roi_interface.rois) > 0:
            self.roi_interface.motion_notify_callback()
        elif not self.ROI:
            self.line_interface.motion_notify_callback()
        elif not self.inactive:
            self.current_roi.motion_notify_callback()
    
    def new_roi(self):
        if self.current_roi != None and self.current_roi.completed != True:
            dpg.configure_item(self.current_roi, points=[])
                            
        self.current_roi = RoiPoly(self.window, self.frame_width, self.frame_height)
        self.inactive = False

    def release(self):
        if self.inactive and len(self.roi_interface.rois) > 0 and self.ROI:
            self.roi_interface.left_mouse_release_callback()
        elif not self.ROI:
            self.line_interface.left_mouse_release_callback()

    def save_rois(self, path): #TODO change to being able to save to filename specified
        with open(path[:-4] + ".cells", 'wb') as filename:
            pickle.dump(self.roi_interface.rois, filename)

    def load_rois(self, _, app_data: dict):        
        with open(app_data["file_path_name"], 'rb') as filename:
            rois = pickle.load(filename)
            
        new_rois = []
        for roi in rois:
            new_rois.append(RoiPoly(self.window, self.frame_width, self.frame_height, roi.lines))
            
        self.roi_interface.rois.extend(new_rois)
            
    def load_line_config(self, _, app_data: dict):
        with open(app_data["file_path_name"], 'rb') as filename:
            lines = pickle.load(filename)
            self.line_interface.load_lines(lines)
    
    def save_line_config(self, path): #TODO change to being able to save to filename specified
        with open(path[:-4] + ".lines", 'wb') as filename:
            new_lines = []
            for line in self.line_interface.lines:
                config_dict = dpg.get_item_configuration(line)
                new_lines.append([config_dict["p1"], config_dict["p2"]])
            pickle.dump(new_lines, filename)    
        
    def auto_gen_rois(self):
        pass
 
    def generate_rois(self):
        shortened_contours = self.line_interface.generate_rois()
        self.roi_interface.rois = self.make_rois(shortened_contours)        
        self.ROI=True
        
    def make_rois(self, shortened_contours):
        """Make ROIs."""
        rois = []
        for i in range(len(shortened_contours)):
            n_l = [[int(j[0]), int(j[1])] for j in shortened_contours[i]]
            roi = RoiPoly(self.window, self.frame_width, self.frame_height, lines=n_l)
            rois.append(roi)
        return rois
        
    def clear_screen(self):    
        for line in self.line_interface.lines:
            dpg.delete_item(line)
        
        self.line_interface.lines.clear()

        for roi in self.roi_interface.rois:
            dpg.delete_item(roi.poly)
        
        self.roi_interface.rois.clear()

    def copy(self, _, app_data):            
        if self.ROI and self.ctrl_has_been_pressed:
            
            roi = self.roi_interface.check_for_hover()
            if roi is not None:
                new_roi = RoiPoly(self.window, self.frame_width, self.frame_height, lines=roi.lines)
                self.roi_interface.rois.append(new_roi)
        else:
            self.line_interface.copy()
            
        self.ctrl_has_been_pressed=False
        
    def control(self, _, __): # janky soln to key press check
        self.ctrl_has_been_pressed = True
        
    def delete(self, _, appdata):
        if self.ROI:
            roi = self.roi_interface.check_for_hover()
            dpg.delete_item(roi.poly)
            self.roi_interface.rois.remove(roi)

        else:
            self.line_interface.delete()