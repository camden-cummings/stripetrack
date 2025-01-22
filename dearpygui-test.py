import array
import cv2
import numpy as np

import dearpygui.dearpygui as dpg

from tracker.full_process_tester_helpers import FullProcessTesterHelpers

ex_fn = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/testlog_277_ubi-kcn.avi"

def next_frame(fn):
    vidcap = cv2.VideoCapture(fn)
    cont, curr_img = vidcap.read()

    while cont:
        cont, curr_img = vidcap.read()

        yield curr_img
    
vidcap = cv2.VideoCapture(ex_fn)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

raw_data = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

dpg.create_context()

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(frame_width, frame_height, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
    
def update_dynamic_texture(new_frame):
    global raw_data
    h2, w2, _ = new_frame.shape
    raw_data[:h2, :w2] = new_frame[:,:] / 255

cell_contours = []

num_rows, num_cols = 1, 1
f = FullProcessTesterHelpers(ex_fn, cell_contours, num_rows, num_cols, vidcap)

mode_noblur_path = ex_fn[:-4] + "-mode.png"
mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

class ContourDefiner:
    def __init__(self):
        self.cv_method = ""
        self.thresh = 155
        self.centroid_size = 70
        
    def cv_alg_change(self, sender_id, data):
        self.cv_method = data
        
    def threshold_change(self, sender_id, data):
        self.thresh = data
        
    def centroid_change(self, sender_id, data):
        self.centroid_size = data
     
c = ContourDefiner()
with dpg.window(label="Video player", pos=(50,50), width = frame_width, height=frame_height+150):
    dpg.add_image("texture_tag")
    
    with dpg.tree_node(label="Basic", default_open=True):

        with dpg.group():
            dpg.add_combo(("No Contours", "Structural Similarity", "Real Time"), label="Contour Detecting Algorithm", callback=c.cv_alg_change, default_value="No Contours")
            dpg.add_slider_float(label="Threshold", callback=c.threshold_change, min_value=0, max_value=255, default_value=c.thresh)            
            dpg.add_slider_float(label="Centroid Size", callback=c.centroid_change, max_value=1000, default_value=c.centroid_size)
        
dpg.create_viewport(title='Dashboard', width=int(frame_width+100), height=int(frame_height+250))
dpg.setup_dearpygui()
dpg.show_viewport()

frame_counter = 0
video_gen = next_frame(ex_fn)
for curr_img in video_gen:
    if dpg.is_dearpygui_running():
        match c.cv_method:
            case "Structural Similarity":
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)            

                f.MIN_AREA = c.centroid_size
                contours, diff = f.structural_sim_contours(curr_img_gray, mode_noblur_img, min_thresh = c.thresh)     
                cv2.drawContours(curr_img, contours, -1, (255, 0, 0), 1)
            case "Real Time":
                if frame_counter % 50 == 0:
                    f.standard_image_noise = f.find_image_noise(vidcap, 50)
                    
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)            
                f.MIN_AREA = c.centroid_size
                sharpened_contours, contour_img = f.new_CV(curr_img_gray, min_thresh = c.thresh)
                cv2.drawContours(curr_img, sharpened_contours, -1, (0, 255, 0), 1)  # RED
                
        update_dynamic_texture(curr_img)
        frame_counter += 1

                
        dpg.render_dearpygui_frame()
    else:
        break  # deal with this scenario appropriately

dpg.destroy_context()