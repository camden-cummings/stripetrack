import pandas as pd
from time import time, perf_counter, get_clock_info, ctime, gmtime
from threading import Thread

import cv2

import serial 

import numpy as np

from pathlib import PurePath

import subprocess

#import PySpin

#from AcquireAndDisplay import setup, save_images

import os

schedule_times = pd.read_csv(os.getcwd() + "/scheduled-events", sep="\t", header=None)

#dev = serial.Serial(port='COM11', baudrate=115200, timeout=.1) 

high_speed_path_template = PurePath(".../hsmovie")
movie_index = 0

keep_on = True

class PreciseTime: 
    # being paranoid and pulling some nice time code from here: https://github.com/hardbyte/python-can/pull/936
    # to double check that time stamps will be as correct as possible

    def __init__(self): 
        # use this if the resolution is higher than 10us 
        if get_clock_info("time").resolution > 1e-5: 
            t0 = time() 
            while True: 
                t1, performance_counter = time(), perf_counter() 
                if t1 != t0: 
                    break 
            self.time = t1 
            self.perfcounter = performance_counter 
        else: 
            self.time = time() 
            self.perfcounter = None # not needed 
    
    # returns in [H, M, S] format
    def formatted_time(self, time):
        return [int(c) for c in ctime(time).split()[3].split(":")]

    def now(self): 
        if self.perfcounter is None: 
            return time()
        return self.time + (perf_counter() - self.perfcounter)

def simple_time(time):
    return time[0]*3600+time[1]*60+time[2]

def subtr_seconds(tm):
    pt = PreciseTime()
    
    precise_seconds = pt.now()
    
    hour, minute, second = pt.formatted_time(precise_seconds)

    s_from_start_of_day = hour * 3600 + minute * 60 + precise_seconds%60
    
    return simple_time(tm) - s_from_start_of_day

class Server:
    def __init__(self):
        self.counter = 0
        self.at_time, self.command_string, self.type_of_video = self.process_command_string(schedule_times.iloc[self.counter]) 
        self.num_of_instructions = schedule_times.shape[0]
        self.pt = PreciseTime()
        
    def take_video(self):
        print("taking video of kind", self.type_of_video)
        print(self.num_of_instructions)
        while self.counter < self.num_of_instructions:
            self.at_time, self.command_string, self.type_of_video = self.process_command_string(schedule_times.iloc[self.counter]) 
            print(f"entering video function (but still in take video) {subtr_seconds(self.at_time)} seconds before time")
            
            if self.type_of_video == 0:
                pass #no video (?)
            elif self.type_of_video == 1:
                self.video(1)   
            elif self.type_of_video == 2:
                self.video(108000)   
                
            self.counter += 1
     
    def send_to_arduino(self):
        prev = -1

        while self.counter < self.num_of_instructions:
            #print(f"{self.counter}, {self.num_of_instructions}")
            #if self.counter == self.num_of_instructions: 
            #   #here to make sure that if self.num_of_instructions changed midway through loop and while check doesn't run, still doesn't enter next check
            #    print("!")
            #    break
            if self.counter != prev and self.counter < self.num_of_instructions:
                print(self.counter, self.num_of_instructions, "arduino waiting for time {}, current time {}".format(self.at_time, self.pt.formatted_time(self.pt.now())))
                
                while self.pt.formatted_time(self.pt.now()) != self.at_time:
#                    print(self.pt.now() % 60)
                    if self.counter >= self.num_of_instructions:
                        break
                    pass
                    #print(pt.formatted_time(pt.now()), at_time)

                print(f"error in arduino sending is {subtr_seconds(self.at_time)} seconds")
#                dev.write(bytes(self.command_string, 'utf-8')) 

                prev = self.counter

    
    def video(self, duration):
        print(f"video function starts {subtr_seconds(self.at_time)} seconds before time")
        
        while self.pt.formatted_time(self.pt.now()) != self.at_time:
            pass
        
        print(f"error in video start is {subtr_seconds(self.at_time)} seconds")
        
        end_time = simple_time(self.pt.formatted_time(self.pt.now())) + duration
        
        while(simple_time(self.pt.formatted_time(self.pt.now())) < end_time):
            pass
        
        print(f"error in video end is {subtr_seconds(self.at_time) + duration} seconds")

    """        
    def video(self, duration: int):
        
        Example entry point; notice the volume of data that the logging event handler
        prints out on debug despite the fact that very little really happens in this
        example. Because of this, it may be better to have the logger set to lower
        level in order to provide a more concise, focused log.
        
        duration: in seconds 
        :return: True if successful, False otherwise.
        :rtype: bool
        

        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()
    
        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
    
        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()
    
        num_cameras = cam_list.GetSize()
    
        print('Number of cameras detected: %d' % num_cameras)
    
        # Finish if there are no cameras
        if num_cameras == 0:
    
            # Clear camera list before releasing system
            cam_list.Clear()
    
            # Release system instance
            system.ReleaseInstance()
    
            print('Not enough cameras!')
            #input('Done! Press Enter to exit...')
            return False
    
        # Run example on each camera
        for i, cam in enumerate(cam_list):
    
            print('Running example for camera %d...' % i)
    
            self.run_single_camera(cam, duration)
            print('Camera %d example complete... \n' % i)
    
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam
    
        # Clear camera list before releasing system
        cam_list.Clear()
    
        # Release system instance
        system.ReleaseInstance()
    
        print("done")
            
    def run_single_camera(self, cam, duration):
        
        This function acts as the body of the example; please see NodeMapInfo example
        for more in-depth comments on setting up cameras.
    
        :param cam: Camera to run on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        
        print("enters run with {} seconds".format(self.pt.now() % 60 - self.at_time[2]))
    
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
    
            # Initialize camera
            cam.Init()
    
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()
            
            setup(cam, nodemap, nodemap_tldevice)
            print("setup is done.")
            #image = save_images(cam)
    
            i = 0
            
            while self.pt.formatted_time(self.pt.now()) != self.at_time:
                pass
            
            end_time = simple_time(self.pt.formatted_time(self.pt.now())) + duration
            
            print("error in video start is {} seconds".format(self.pt.now() % 60 - self.at_time[2]))
    
            while(simple_time(self.pt.formatted_time(self.pt.now())) < end_time):
                image = save_images(cam)
            #    result.write(image)
                
                #print(image.shape[0] ==)
                cv2.imwrite(f"images/im-{i}.jpg", image)
                i += 1
            
            print("error in video end is {} seconds".format(self.pt.now() % 60 - self.at_time[2] - duration))
    
            cam.EndAcquisition()
    
            # Deinitialize camera
            cam.DeInit()
    
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
    """

    def process_command_string(self, cmd_string):
        at_time = [int(r) for r in cmd_string.iloc[0].split(":")]
    
        arduino_command = cmd_string.iloc[3]
    
        if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
            at_time[0] += 12
        
        video_type = cmd_string.iloc[2]
        
        return at_time, arduino_command, video_type
    
    #if __name__ == "__main__":
 
s = Server()
print("spinning up threads")

t1 = Thread(target=s.take_video)
t2 = Thread(target=s.send_to_arduino)

print("starting")

t1.start()
t2.start()

print("started")

#t1.join()
#t2.join()
print("joined")
    
        #for index, row in schedule_times.iterrows():
        #    print("index", index)
"""
        time_diff = simple_time(at_time) - simple_time(pt.formatted_time(pt.now()))
        prev_time_diff = 0
        
        while not time_diff <= 60 or time_diff < 0:
            time_diff = simple_time(at_time) - simple_time(pt.formatted_time(pt.now()))

            if time_diff != prev_time_diff:
                print(time_diff)
                
            prev_time_diff = time_diff
            pass
        
        #else:
        #    print("time {} has already passed".format(at_time))

    
    now = pt.now()
      """  