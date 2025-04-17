import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from gui import GUI

from pathlib import Path

import os

from helpers import get_shape

fp = "/home/chamomile/Thyme-lab/data/vids/apr-run/"
         
def find_all_videos_for_tracking(path=None, ext="avi"):
    """Finds all avi files in current working directory, if path given finds all
    avi files in path that aren't output of the algorithm."""
    if path is None:
        path = os.getcwd()

    files_to_read = []
    for ex in os.walk(path):
        for filename in ex[2]:
            if filename[-3:] == ext and not "tracked" in filename and not "checking" in filename \
            and not "examine" in filename and not "output" in filename:
                files_to_read.append(ex[0] + "/" + filename)

    return files_to_read

all_files = find_all_videos_for_tracking(fp, ext="mp4")
#all_files = ["/home/chamomile/Thyme-lab/data/vids/brandon-data/brandon_y_mazes/4-3-25/box1/fc2_save_2025-04-03-132533-0000.mp4"]

for filename in all_files:    
    dpg.create_context()

    ext = filename.split('.')[-1] 
    vidcap = None
    
    if ext in ["avi", "mp4"]:
        vidcap = cv2.VideoCapture(filename)
        
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
                
    elif ext in ["png"]:
        curr_img = cv2.imread(filename)
        shape = curr_img.shape
        frame_width, frame_height = get_shape(shape)
                    
    window = dpg.add_window(label="Video player", pos=(50, 50), width=frame_width, height=frame_height) 
    
    path = Path(filename)
    curr_dir = path.parent
    curr_name = str(path.stem)

    #dpg.add_button(label="Choose Video/Image", callback=lambda: dpg.show_item("open_file"), pos=[1700, 0], parent = window)
    m = GUI(filename, window, vidcap, frame_width, frame_height)
    
    m.start(window)
        
    dpg.destroy_context()

