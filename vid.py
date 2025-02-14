import PySpin
from AcquireAndDisplayClass import get_image
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pyplot as plt

import cv2
import numpy as np
global continue_recording

from arduino_server import PreciseTime
import pandas as pd
import os

continue_recording = True
FRAME_HEIGHT, FRAME_WIDTH = 660, 1088

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
        at_time[0] += 12

    video_type = cmd_string.iloc[2]

    return at_time, arduino_command, video_type

def setup(system):
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
        input('Done! Press Enter to exit...')
        return None
    
    return cam_list

def setup_nodemap(cam):
    nodemap_tldevice = cam.GetTLDeviceNodeMap()

    # Initialize camera
    cam.Init()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
    
    return nodemap, nodemap_tldevice

def set_node_acquisition_mode(nodemap):
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsReadable(node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False

    # Retrieve integer value from entry node
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    print('Acquisition mode set to continuous...')
    
def run():
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    cam_list = setup(system)


    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        try:
            nodemap, nodemap_tldevice = setup_nodemap(cam)
            print('*** IMAGE ACQUISITION ***\n')
            try:
                set_node_acquisition_mode(nodemap)

                cam.BeginAcquisition()
        
                print('Acquiring images...')
        
                device_serial_number = ''
                node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(node_device_serial_number):
                    device_serial_number = node_device_serial_number.GetValue()
                    print('Device serial number retrieved as %s...' % device_serial_number)
        
                # Close program
               # print('Press enter to close the program..')
                global continue_recording
                timer = PreciseTime()
                # Retrieve and display images
                pr = cProfile.Profile()
                pr.enable()
                
                counter = 0
                schedule_times = pd.read_csv(os.getcwd() + "\\scheduled-events", sep="\t", header=None)
                num_of_instructions = schedule_times.shape[0]
                
                while counter < num_of_instructions:
                    print("counter", counter)
                    continue_recording = True

                    at_time, command_string, type_of_video = process_command_string(
                        schedule_times.iloc[counter])
                    
                    print(timer.formatted_time(timer.now()), at_time, command_string, type_of_video)

                    if type_of_video == 0:
                        duration = None
                    elif type_of_video == 1:
                        duration = 1
                    elif type_of_video == 2:
                        duration = 108000
    
                    i = 0
                    
                    print(duration)
                    if duration != None:
                        vid_name = "-".join(str(at_time).strip("[]").split(", "))
                 
                        if type_of_video == 1:
                            result = cv2.VideoWriter(f'{vid_name}.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                     285, (FRAME_WIDTH, FRAME_HEIGHT), False)
                        else:
                            result = cv2.VideoWriter(f'{vid_name}.avi',
                                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                                         30, (FRAME_WIDTH, FRAME_HEIGHT), False)
                            
                        while timer.formatted_time(timer.now()) != at_time:
                            pass
                        
                        end_time = int(timer.now()) + duration
                        c = 0
                        while(continue_recording):# and dpg.is_dearpygui_running():
                            try:
                                image = get_image(cam)
                                                       
                                result.write(image)
                                c += 1
                                if (timer.now() >= end_time):
                                    continue_recording=False  
                                    
                            except PySpin.SpinnakerException as ex:
                                print('Error: %s' % ex)
                                return False
                            
                        print(c)

                        counter += 1
                        print("reached the end")

                pr.disable()
                s = io.StringIO()
                sortby = SortKey.CUMULATIVE
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
        
                cam.EndAcquisition()
        
            except Exception as e:
                print(e)
                print("failed during image GUI capture")
                
            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

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

run()