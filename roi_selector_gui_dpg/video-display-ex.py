import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from gui import GUI

dpg.create_context()

fp = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/"
ex_fn = fp + "fc2_save_2025-02-06-151155-0000.mp4"

vidcap = cv2.VideoCapture(ex_fn)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

m = GUI(ex_fn, frame_width, frame_height)
while dpg.is_dearpygui_running():
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

