#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:12:36 2025

@author: chamomile
"""

import dearpygui.dearpygui as dpg

#TODO: make both inherited functions

def get_mouse_pos(shift):
    """Shifts mouse position by shift to allow canvas to be put anywhere in a window."""
    x, y = dpg.get_mouse_pos()

    return x+shift[0], y+shift[1]

def convert_to_in_bounds(point: tuple[int, int], frame_width, frame_height, shift):
    """Return a point that is in bounds based on given point."""
    return [max(min(frame_width+shift[0], point[0]), 0), max(min(frame_height+shift[1], point[1]), 0)]

def update_frame_shape(obj, new_frame_width, new_frame_height):
    obj.frame_width = new_frame_width
    obj.frame_height = new_frame_height

def get_shape(shape):
    return shape[1], shape[0]