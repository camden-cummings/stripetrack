from multiprocessing import Process
import multiprocessing
import argparse

import PySpin
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from live_tracker.camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image
from live_tracker.gui_helpers import GUIHelpers
from live_tracker.config import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, FPS

import keyboard 


class PoolRun:
    def __init__(self, frame_width, frame_height, fps):
        self.FRAME_HEIGHT, self.FRAME_WIDTH = frame_height, frame_width
        self.FPS = fps
        self.image_data = np.zeros((self.FRAME_WIDTH, self.FRAME_HEIGHT, 3))
    
    def video_pool(self, queue, done, start_recording):
        print("start video pool")
        try:
            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            cam_list = setup(system)

            # Run example on each camera
            cam = cam_list[0]
                
            nodemap, nodemap_tldevice = setup_nodemap(cam)
            print('*** IMAGE ACQUISITION ***\n')

            set_node_acquisition_mode(nodemap)

            node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            node_fps.SetValue(self.FPS)

            width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            
            try:
                width.SetValue(self.FRAME_WIDTH)
                height.SetValue(self.FRAME_HEIGHT)
            except Exception as e:
                print(f'Failed to set width and height to the values given, {self.FRAME_WIDTH}, {self.FRAME_HEIGHT}, with following error:')    
                print(e)
            
            cam.BeginAcquisition()
            
            while not done.is_set():
                image = get_image(cam)
                image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                queue.put(image_data)
                
                if keyboard.is_pressed('ENTER'):
                    done.set()
                    
            cam.EndAcquisition()
                
            # Deinitialize camera
            cam.DeInit()
    
        except Exception as ex:
            print('Error: %s' % ex)
    
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam
    
        # Clear camera list before releasing system
        cam_list.Clear()
    
        # Release system instance
        system.ReleaseInstance()
    
    def gui_pool(self, queue, done, start_recording):
        print("start gui pool")
        dpg.create_context()
    
        window = dpg.add_window(label="Video player", pos=(50, 50), width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT) 
        gui = GUIHelpers(window, self.FRAME_WIDTH, self.FRAME_HEIGHT)
        
        dpg.set_primary_window(window, True)
        dpg.create_viewport(width=int(self.FRAME_WIDTH*1.5), height=self.FRAME_HEIGHT+20, title="ROI Selector")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        while not done.is_set():
            image_data = queue.get()
            data = np.asarray(image_data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
                        
            dpg.set_value("texture_tag", texture_data)
            dpg.render_dearpygui_frame()
              
        dpg.destroy_context()

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-frame_width",
        "--frame_width",
        default=DEFAULT_FRAME_WIDTH
    )
    
    parser.add_argument(
        "-frame_height",
        "--frame_height",
        default=DEFAULT_FRAME_HEIGHT
    )
    
    parser.add_argument(
        "-fps",
        "--fps",
        default=FPS
    )
    
    args = parser.parse_args()

    frame_width = args.frame_width
    frame_height = args.frame_height
    fps = args.fps

    poolrun = PoolRun(frame_width, frame_height, fps)

    print('Acquiring images...')
    
    queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    start_recording = multiprocessing.Event()
    vid_p = Process(target=poolrun.video_pool, args=(queue, done, start_recording, ))
    gui_p = Process(target=poolrun.gui_pool, args=(queue, done, start_recording, ))
    
    vid_p.start()    
    gui_p.start()
    vid_p.join()
    gui_p.join()
    