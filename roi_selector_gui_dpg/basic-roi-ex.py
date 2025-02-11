import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from visibility_manager import VisibilityManager

dpg.create_context()

frame_width = 500
frame_height = 500

m = VisibilityManager("file_save_path", frame_width, frame_height)
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()
    
dpg.destroy_context()

