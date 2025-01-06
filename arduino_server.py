import pandas as pd
from time import time, perf_counter, get_clock_info, ctime, gmtime
from threading import Thread

import cv2

import serial 

import numpy as np

from pathlib import PurePath

schedule_times = pd.read_csv("C:\\Users\\ThymeLab\\Desktop\\scheduled-events", sep="\t", header=None)

dev = serial.Serial(port='COM5', baudrate=115200, timeout=.1) 

high_speed_path_template = PurePath(".../hsmovie")
movie_index = 0

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


def track_fish(viz, what_kind_of_movie):
    """
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not (cam.isOpened()):
        print("could not open given camera")
    else:
        while True:
            result, image = cam.read()
            cv2.imshow("frame", image)
            cv2.waitKey(0)

        cv2.destroyWindows()
    """

    while pt.formatted_time(pt.now()) != at_time:
        pass
        
    # find fish in image
    im = 0

    print("error in fish tracking is {} seconds".format(pt.now() % 60 - at_time[2]))
    
    """
    if what_kind_of_movie == 1:
        high_speed_path = high_speed_path_template.root + "_" + str(movie_index) + ".avi"
        #result = (high_speed_path)
        #write 285 frames over the next second 
        movie_index += 1
    elif what_kind_of_movie == 2: 
	long_speed_path = 
	desired_min = 1
	#write desired number of frames for next desired_min
    """

    if viz == True:
        # show image post tracking
        im += 1

def less_than(at_time):
    return (pt.formatted_time(pt.now())[0] <= at_time[0] and pt.formatted_time(pt.now())[1] <= at_time[1] and pt.formatted_time(pt.now())[2] <= at_time[2]) or (pt.formatted_time(pt.now())[0] <= at_time[0] and pt.formatted_time(pt.now())[1] <= at_time[1]) or pt.formatted_time(pt.now())[0] <= at_time[0]

def simple_time(time):
    return time[0]*3600+time[1]*60+time[2]
    
def send_to_arduino(at_time, command_string):
    print("waiting for time {}, current time {}".format(at_time, pt.formatted_time(pt.now())))
    
    while pt.formatted_time(pt.now()) != at_time:
        pass

    print("error in arduino sending is {} seconds".format(pt.now() % 60 - at_time[2]))
    dev.write(bytes(command_string, 'utf-8')) 

if __name__ == "__main__":
    pt = PreciseTime()

    for index, row in schedule_times.iterrows():
        at_time = [int(r) for r in row.iloc[0].split(":")]

        command_string = row.iloc[3]
    
        if row.iloc[1] == "PM" and at_time[0] != 12:
            at_time[0] += 12
            
        if less_than(at_time):
            print("waiting for time", at_time, "starting at time", pt.formatted_time(pt.now()))
            while not simple_time(at_time) - simple_time(pt.formatted_time(pt.now())) <= 60:
                pass
            
            print("spinning up threads")
            
            t1 = Thread(target=track_fish, args=(True, row.iloc[1], ))
            t2 = Thread(target=send_to_arduino, args=(at_time, command_string, ))
                
            t1.start()
            t2.start()
                
            t1.join()
            t2.join()
            
        else:
            print("time {} has already passed".format(at_time))

    
    now = pt.now()
    
