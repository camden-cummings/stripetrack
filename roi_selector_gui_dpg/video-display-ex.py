import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from gui import GUI

import os

fp = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/to-be-processed/march/"

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

all_files = find_all_videos_for_tracking(fp, ext="png")
for filename in all_files:    
    dpg.create_context()

    ext = filename.split('.')[-1] 
    video = False
    
    if ext in ["avi", "mp4"]:
        vidcap = cv2.VideoCapture(filename)
        
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        
        video = True
        
    elif ext in ["png"]:
        curr_img = cv2.imread(filename)
        shape = curr_img.shape
        
        if len(shape) == 2:
            frame_width, frame_height = curr_img.shape
        elif len(shape) == 3:
            frame_height, frame_width, _ = curr_img.shape
        
    m = GUI(filename, frame_width, frame_height)
    
    while dpg.is_dearpygui_running():
        if video:
            cont, curr_img = vidcap.read()
                
            if not cont:
                _ = vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cont, curr_img = vidcap.read()
        
        data = np.flip(curr_img, 2)
        data = data.ravel()
        data = np.asfarray(data, dtype='f')
        texture_data = np.true_divide(data, 255.0)
        
        dpg.set_value("texture_tag", texture_data)
        
        dpg.render_dearpygui_frame()
        
    dpg.destroy_context()

