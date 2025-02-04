# coding=utf-8
# =============================================================================
# Copyright (c) 2024 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# This AcquireAndDisplay.py shows how to get the image data, and then display images in a GUI.
# This example relies on information provided in the ImageChannelStatistics.py example.
#
# This example demonstrates how to display images represented as numpy arrays.
# Currently, this program is limited to single camera use.
# NOTE: keyboard and matplotlib must be installed on Python interpreter prior to running this example.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics


import os
import PySpin
import sys
import keyboard
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from pathlib import Path

from pynput.keyboard import Key, Controller

import matplotlib.pyplot as plt

from roi_selector_gui_dpg.statemanager import StateManager

import subprocess


global continue_recording
continue_recording = True

FRAME_HEIGHT, FRAME_WIDTH = 1200, 1920

raw_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

ex_fn = os.getcwd()

dpg.create_context()

def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """

    global continue_recording
    continue_recording = False


def __change(e, data):
    if data == "ROI":
        dpg.show_item(roi)
        dpg.hide_item(line)
        
        state_manager.ROI = True
    elif data == "Line":
        dpg.show_item(line)
        dpg.hide_item(roi)
        state_manager.ROI = False

def __generate_rois():
    dpg.hide_item(roi_and_line_selection)
    dpg.show_item(post_line)
    
    state_manager.generate_rois()

def __restart():
    dpg.show_item(roi_and_line_selection)
    dpg.hide_item(post_line)

    state_manager.clear_window()
    print(state_manager.roi_interface.rois)
    
def __start_movies_and_stimuli():
    print("go to arduino script here")
    keyboard = Controller()
    keyboard.press(Key.enter)
    subprocess.call(['python.exe',"arduino_server.py"])

def setup_elements():
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(FRAME_WIDTH, FRAME_HEIGHT, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
        
    with dpg.theme() as canvas_theme, dpg.theme_component():
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)
    
    with dpg.window(label="Video player", pos=(50,50), width = FRAME_WIDTH, height=FRAME_HEIGHT) as window:
        state_manager = StateManager(FRAME_WIDTH, FRAME_HEIGHT, window)
    
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=state_manager.motion_notify_callback)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=state_manager.release)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=state_manager.left_mouse_press_callback)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=state_manager.right_mouse_press_callback)
            dpg.add_key_press_handler(key=dpg.mvKey_C, callback=state_manager.copy)
            dpg.add_key_press_handler(key=dpg.mvKey_LControl, callback=state_manager.control)
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=state_manager.delete)
            
        with dpg.child_window(border=False):
            with dpg.group() as roi_and_line_selection:
                shift = FRAME_WIDTH+10
                dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[shift,0], callback=__change, default_value="ROI")
                dpg.add_button(label="START", callback=__start_movies_and_stimuli)
                
                path = Path(ex_fn)
                with dpg.group(label="roi buttons", pos=[shift,25]) as roi: #ROI Mode Buttons
                    dpg.add_button(label="New ROI", callback=state_manager.new_roi)
                    
                    curr_dir = path.parent
                    curr_name = str(path.stem)
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.load_rois, id="roi_load_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                        dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                        
                    dpg.add_button(label="Load ROI File", callback=lambda: dpg.show_item("roi_load_file"))
            
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.roi_interface.save_rois, id="roi_save_file", width=700 ,height=400, default_path = curr_dir, default_filename = curr_name):
                        dpg.add_file_extension(".cells", color=(0, 255, 0, 255), custom_text="[ROI Save File]")
                        
                    dpg.add_button(label="Save ROIs", callback=lambda: dpg.show_item("roi_save_file"))
                    dpg.add_button(label="Auto Generate ROIs", callback=state_manager.auto_gen_rois)
                    
                    
                    dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 125), wrap=150)
    
                with dpg.group(label="line buttons", pos=[shift, 25]) as line:
                    dpg.add_button(label="Vertical Line", callback=state_manager.line_interface.vertical_line_callback)
                    vert = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+104,25], callback=state_manager.line_interface.num_of_vert_lines_changer)
                    dpg.add_button(label="Horizontal Line", callback=state_manager.line_interface.horizontal_line_callback)
                    hor = dpg.add_input_text(width=15, source="int_value", default_value=1, pos=[shift+118,48], callback=state_manager.line_interface.num_of_hor_lines_changer)
                    dpg.add_button(label="Generate ROIs", callback=__generate_rois)
    
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.save_lines, id="line_save_file", width=700 ,height=400):
                        dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                        
                    dpg.add_button(label="Save Line Configuration", callback="line_save_file")
    
                    with dpg.file_dialog(directory_selector=False, show=False, callback=state_manager.line_interface.load_lines, id="line_load_file", width=700 ,height=400):
                        dpg.add_file_extension(".lines", color=(0, 255, 0, 255), custom_text="[Line Save File]")
                    
                    dpg.add_button(label="Load Line Configuration", callback=lambda: dpg.show_item("line_load_file"))
    
                    dpg.add_text("NOTES \nclick and hold the edge of a ROI to rotate it \n\nSHORTCUTS \n ctrl+c: copy \n del: delete", pos=(shift+5, 195), wrap=150)
    
                dpg.hide_item(line)
            
            with dpg.group(label="post line buttons", pos=[shift,0]) as post_line:
                dpg.add_button(label="Save ROIs", callback=state_manager.roi_interface.save_rois)
                dpg.add_button(label="Clear Screen and Start Over", callback=__restart)
    
            dpg.hide_item(post_line)
        
        dpg.add_image("texture_tag", pos=[8,8])
    
    dpg.set_primary_window(window, True)
    dpg.create_viewport(width=int(FRAME_WIDTH*1.2), height=FRAME_HEIGHT+20, title="ROI Selector")
    dpg.setup_dearpygui()
    dpg.show_viewport()

    return roi, line, state_manager, roi_and_line_selection, post_line

roi, line, state_manager, roi_and_line_selection, post_line = setup_elements()

def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    """
    This function continuously acquires images from a device and display them in a GUI.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global continue_recording

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

    print('*** IMAGE ACQUISITION ***\n')
    try:
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

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Close program
        print('Press enter to close the program..')
        
        # Retrieve and display images
        while(continue_recording) and dpg.is_dearpygui_running():
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.

                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:                    

                    # Getting the image data as a numpy array
                    old_image_data = image_result.GetNDArray()
                    image_data = cv2.cvtColor(old_image_data, cv2.COLOR_GRAY2BGR)
                    # Draws an image on the current figure
                    #im.set_data(cv2.cvtColor(cv2.pyrDown(image_data), cv2.COLOR_BGR2RGB))
   
                    data = np.flip(image_data, 2)
                    data = data.ravel()
                    data = np.asfarray(data, dtype='f')
                    texture_data = np.true_divide(data, 255.0)
                    
                   # print(texture_data.shape)
                    dpg.set_value("texture_tag", texture_data)
                        
                    dpg.render_dearpygui_frame()
                    plt.pause(0.001)
                    # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # Interval is in seconds.
        
                    # If user presses enter, close the program
                    if keyboard.is_pressed('ENTER'):
                        print('Program is closing...')
                        dpg.destroy_context()

                        # Close figure
                        input('Done! Press Enter to exit...')
                        continue_recording=False                        

                #  Release image
                #
                #  *** NOTES ***
                #  Images retrieved directly from the camera (i.e. non-converted
                #  images) need to be released in order to keep from filling the
                #  buffer.
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except Exception as e:
        print(e)
        print("failed during image GUI capture")

def save_images(cam, nodemap, nodemap_tldevice, vid_name):
    """
    This function continuously acquires images from a device and display them in a GUI.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global continue_recording

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

    print('*** IMAGE ACQUISITION ***\n')
    try:
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

        cam.BeginAcquisition()

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)


        result = cv2.VideoWriter(vid_name,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (FRAME_WIDTH, FRAME_HEIGHT))    
    
        while(continue_recording):
            try:
                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:                    

                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()

                    result.write(image_data)                    

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()
    
    except Exception as e:
        print(e)

def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; notice the volume of data that the logging event handler
    prints out on debug despite the fact that very little really happens in this
    example. Because of this, it may be better to have the logger set to lower
    level in order to provide a more concise, focused log.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

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
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam)
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

    input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
