from multiprocessing import Pool, Process
import multiprocessing
import cProfile, pstats, io
from pstats import SortKey

import PySpin
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from camera_helpers import setup, setup_nodemap, set_node_acquisition_mode, get_image
from gui_helpers import GUIHelpers

from precise_time import PreciseTime

from command_reader import process_command_string

import pandas as pd
import os

import keyboard 
import serial
     
from memory_profiler import profile

class PoolRun:
    def __init__(self):
        self.FRAME_HEIGHT, self.FRAME_WIDTH = 660, 992
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

            #node_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))
            #node_enable.SetValue(True)

            node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            node_fps.SetValue(50.0)   
            
            cam.BeginAcquisition()
            
            while not done.is_set():
                image = get_image(cam)
                image_data = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                queue.put(image_data)
                
                #if start_recording.is_set(): # should really be a pipe with commands
                #    node_fps.SetValue(285.0)   

                    
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
    
    @profile
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
            #print(queue.qsize())
            image_data = queue.get()
            data = np.flip(image_data, 2)
            data = data.ravel()
            data = np.asarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
                        
            dpg.set_value("texture_tag", texture_data)
            dpg.render_dearpygui_frame()
            
            if gui.start_recording:
                start_recording.set()
                    
        dpg.destroy_context()
    
    def printer_pool(self, queue, done, start_recording):
        timer = PreciseTime()

        counter = 0
        schedule_times = pd.read_csv(os.getcwd() + "\\shortened-schedule", sep="\t", header=None)
        num_of_instructions = schedule_times.shape[0]
        first_time = True
        end_time = np.inf
        dev = serial.Serial(port='COM7', baudrate=115200, timeout=.1)
        
        while counter < num_of_instructions:
            if start_recording.is_set():
                if first_time: #setup all necessary pieces    
                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    print("COMMANDS: ", at_time, command_string, type_of_video)
                    if type_of_video == 0:
                        duration = 0
                    elif type_of_video == 1:
                        duration = 1
                    elif type_of_video == 2:
                        duration = 1800
                    
                    if duration != 0:                        
                        vid_name = "_".join(str(at_time).strip("[]").split(", ")) + "-" + str(counter)

                        if type_of_video == 1:
                            #node_fps.SetValue(285.0)
                            result = cv2.VideoWriter(f'{vid_name}.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                     285, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)
                        else:
                            #node_fps.SetValue(30.0)
                            result = cv2.VideoWriter(f'{vid_name}-long.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                         30, (self.FRAME_WIDTH, self.FRAME_HEIGHT), False)
                        
                    first_time = False
                    not_written_to_arduino = True
                    
                if (timer.formatted_time(timer.now()) == at_time and (type_of_video == 1 or type_of_video == 0)) or (timer.formatted_time(timer.now()) >= at_time and type_of_video == 2):
                    if end_time == np.inf:
                        end_time = int(timer.now()) + duration
                        
                    if duration != 0:
                        image = cv2.cvtColor(queue.get(), cv2.COLOR_BGR2GRAY)
                        result.write(image)
                    
                    if not_written_to_arduino:
                        dev.write(bytes(command_string, 'utf-8'))
                        not_written_to_arduino = False
                    
                if (timer.now() >= end_time):
                    counter += 1    
                    first_time = True
                    end_time = np.inf        
        #while not event.is_set():
        #    print(pt.now())
            
if __name__ == '__main__':   
    poolrun = PoolRun()

    print('Acquiring images...')
    
    pr = cProfile.Profile()
    pr.enable()
    
    queue = multiprocessing.Queue()
    done = multiprocessing.Event()
    start_recording = multiprocessing.Event()
    vid_p = Process(target=poolrun.video_pool, args=(queue, done, start_recording, ))
    gui_p = Process(target=poolrun.gui_pool, args=(queue, done, start_recording, ))
    p = Process(target=poolrun.printer_pool, args=(queue, done, start_recording, ))
    
    vid_p.start()    
    gui_p.start()
    p.start()
    vid_p.join()
    gui_p.join()
    p.join()
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    
    #pool.apply_async(video_pool)
    #video_thread = Thread(target=video)
    #video_thread.start()
