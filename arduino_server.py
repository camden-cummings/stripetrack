from time import time, perf_counter, get_clock_info, ctime
import os
from threading import Thread

import serial
import cv2
import pandas as pd
import PySpin

from AcquireAndDisplayClass import setup, get_image

dev = serial.Serial(port='COM11', baudrate=115200, timeout=.1)

FRAME_HEIGHT, FRAME_WIDTH = 660, 1088

class PreciseTime:
    """Timer which tries to find most precise method of timing."""
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
            self.perfcounter = None  # not needed

    @staticmethod
    def formatted_time(input_time) -> list[int, int, int]:
        """Returns time in [H, M, S] format."""
        return [int(c) for c in ctime(input_time).split()[3].split(":")]

    def now(self) -> float:
        """Finds current time according to best timer."""
        if self.perfcounter is None:
            return time()
        return self.time + (perf_counter() - self.perfcounter)


def simple_time(input_time):
    return input_time[0]*3600+input_time[1]*60+input_time[2]


def subtr_seconds(input_time):
    """Finds precise different between current time and given time"""
    timer = PreciseTime()

    current_time_precise = timer.now()

    hour, minute, _ = timer.formatted_time(current_time_precise)

    s_from_start_of_day = hour * 3600 + minute * 60 + current_time_precise % 60

    return simple_time(input_time) - s_from_start_of_day


class Server:
    """Manages sending series of commands to arduino and taking videos at specified times in given schedule_times file."""

    def __init__(self, schedule_times):
        self.counter = 0
        self.at_time, self.command_string, self.type_of_video = self.process_command_string(
            schedule_times.iloc[self.counter])
        self.num_of_instructions = schedule_times.shape[0]
        self.timer = PreciseTime()
        self.schedule_times = schedule_times

    def send_to_arduino(self):
        """Sends command string to arduino."""
        prev = -1

        while self.counter < self.num_of_instructions:
            if self.counter != prev and self.counter < self.num_of_instructions:
                print(
                    f"arduino waiting for time {self.at_time}, current time {self.timer.formatted_time(self.timer.now())}")

                while self.timer.formatted_time(self.timer.now()) != self.at_time:
                    if self.counter >= self.num_of_instructions:
                        break

                print(
                    f"error in arduino sending is {subtr_seconds(self.at_time)} seconds")
                dev.write(bytes(self.command_string, 'utf-8'))

                prev = self.counter

    def video(self):
        """
        Example entry point; notice the volume of data that the logging event handler
        prints out on debug despite the fact that very little really happens in this
        example. Because of this, it may be better to have the logger set to lower
        level in order to provide a more concise, focused log.

        duration: in seconds
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print(
            f"video function starts {subtr_seconds(self.at_time)} seconds before time")

        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print(
            f'Library version: {version.major}.{version.minor}.{version.type}.{version.build}')

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print(f'Number of cameras detected: {num_cameras}')

        # Finish if there are no cameras
        if num_cameras == 0:

            # Clear camera list before releasing system
            cam_list.Clear()

            # Release system instance
            system.ReleaseInstance()

            print('Not enough cameras!')

            return False

        cam = cam_list[0]
        self.run_single_camera(cam)

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

    def run_single_camera(self, cam):
        """
        This function acts as the body of the example; please see NodeMapInfo example
        for more in-depth comments on setting up cameras.

        :param cam: Camera to run on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print(f"enters run with {subtr_seconds(self.at_time)} seconds")

        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()
            nodemap = cam.GetNodeMap()

            """

            cam.BeginAcquisition()
            cam.EndAcquisition()
            cam.DeInit()
            cam.Init()
            cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
            cam.UserSetLoad()
            # Retrieve GenICam nodemap
            #print(PySpin.IsWritable(nodemap.GetNode("AcquisitionFrameRateControlEnabled")))
            #print(PySpin.IsWritable(nodemap.GetNode("AcquisitionFrameRate")))
            #print(PySpin.IsWritable(nodemap.GetNode("randomstringofchars")))
            node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            framerate_to_set = node_acquisition_framerate.GetValue()
            node_acquisition_framerate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))
            enable = node_acquisition_framerate_enable.GetValue()
            node_acquisition_framerate_enable.SetValue(True)
            print(framerate_to_set, enable)
            node_acquisition_framerate.SetValue(10.0)
            #node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            #node_fps.SetValue(100.0)
            """
            
            setup(cam, nodemap, nodemap_tldevice)
            print("setup is done.")

            while self.counter < self.num_of_instructions:
                self.at_time, self.command_string, self.type_of_video = self.process_command_string(
                    self.schedule_times.iloc[self.counter])
                print(
                    f"entering video function (but still in take video) {subtr_seconds(self.at_time)} seconds before time")

                if self.type_of_video == 0:
                    duration = None
                elif self.type_of_video == 1:
                    duration = 1
                elif self.type_of_video == 2:
                    duration = 30

                i = 0
                
                while self.timer.formatted_time(self.timer.now()) != self.at_time:
                    pass

                if duration != None:
                    end_time = simple_time(
                        self.timer.formatted_time(self.timer.now())) + duration
    
                    print(
                        f"error in video start is {subtr_seconds(self.at_time)} seconds")
                    vid_name = "-".join(str(self.timer.formatted_time(self.timer.now())
                                            ).strip("[]").split(", "))
                    
                    if self.type_of_video == 1:
                        result = cv2.VideoWriter(f'{vid_name}.avi',
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                 285, (FRAME_WIDTH, FRAME_HEIGHT), False)
                    else:
                        result = cv2.VideoWriter(f'{vid_name}.avi',
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                 30, (FRAME_WIDTH, FRAME_HEIGHT), False)
                    
                    print(self.timer.formatted_time(self.timer.now()))

                    while (simple_time(self.timer.formatted_time(self.timer.now())) < end_time):
                        image = get_image(cam)
                        result.write(image)
    
                        i += 1
                    print(self.timer.formatted_time(self.timer.now()))

                    print(i)
    
                    print(
                        f"error in video end is {subtr_seconds(self.at_time) + duration} seconds")
    
                self.counter += 1

            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')

    @staticmethod
    def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
        """Converts a command into separate pieces."""

        at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

        arduino_command = cmd_string.iloc[3]

        if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
            at_time[0] += 12

        video_type = cmd_string.iloc[2]

        return at_time, arduino_command, video_type


events = pd.read_csv(os.getcwd() + "\\scheduled-events", sep="\t", header=None)
s = Server(events)
print("spinning up threads")

video_thread = Thread(target=s.video)
arduino_thread = Thread(target=s.send_to_arduino)

print("starting")

video_thread.start()
arduino_thread.start()

print("started")
