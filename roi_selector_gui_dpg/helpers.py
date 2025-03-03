#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:12:36 2025

@author: chamomile
"""

import dearpygui.dearpygui as dpg

def get_mouse_pos(shift):
    """Shifts mouse position by shift to allow canvas to be put anywhere in a window."""
    x, y = dpg.get_mouse_pos()

    return x+shift[0], y+shift[1]
