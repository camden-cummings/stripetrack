from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
import cv2

from helpers import update_frame_shape, get_shape
from statemanager import StateManager
from gui import GUI


class VideoGUI(GUI):
    def __init__(self, window, frame_width, frame_height, filename: str):
        self.vidcap = cv2.VideoCapture(filename)

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.roi, self.line, self.roi_and_line_selection, self.post_line, self.state_manager = self.setup_elements(
            filename, window)

    def open_file_callback(self, _, appdata: dict):
        self.vidcap = cv2.VideoCapture(appdata["file_path_name"])

        self.frame_width = int(self.vidcap.get(3))
        self.frame_height = int(self.vidcap.get(4))

        update_frame_shape(self.state_manager, self.frame_width, self.frame_height)
        update_frame_shape(self.state_manager.roi_interface, self.frame_width, self.frame_height)
        update_frame_shape(self.state_manager.line_interface, self.frame_width, self.frame_height)

        # file_path_name = appdata["file_path_name"]

    def start(self, window):
        raw_data = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.frame_width, self.frame_height, raw_data,
                                format=dpg.mvFormat_Float_rgb, tag="texture_tag")

        with dpg.theme(), dpg.theme_component():
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)

        dpg.add_image("texture_tag", pos=[8, 8], parent=window)

        dpg.set_primary_window(window, True)

        dpg.create_viewport(width=int(self.frame_width + 275),
                            height=self.frame_height + 20, title="ROI Selector")

        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self.vidcap is not None:
                cont, curr_img = self.vidcap.read()

                if not cont:
                    _ = self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cont, curr_img = self.vidcap.read()

            data = np.flip(curr_img, 2)
            data = data.ravel()
            data = np.asfarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)

            dpg.set_value("texture_tag", texture_data)

            dpg.render_dearpygui_frame()

    def setup_elements(self, filename, window):
        state_manager = StateManager(window, self.frame_width, self.frame_height)
        self.setup_keypress(state_manager)

        with dpg.child_window(border=False, parent=window):
            with dpg.group() as roi_and_line_selection:
                path = Path(filename)
                curr_dir = path.parent
                curr_name = str(path.stem)

                shift = self.frame_width + 10

                with dpg.file_dialog(directory_selector=False, show=False, callback=self.open_file_callback,
                                     id="open_file", width=800, height=400, default_path=curr_dir,
                                     default_filename=curr_name):
                    for vid_ext in [".mp4", ".avi"]:
                        dpg.add_file_extension(vid_ext, color=(
                            0, 255, 0, 255), custom_text="[Video File]")
                    for img_ext in [".png"]:
                        dpg.add_file_extension(vid_ext, color=(
                            0, 255, 0, 255), custom_text="[Image File]")

                dpg.add_button(label="Choose Video/Image", callback=lambda: dpg.show_item("open_file"), pos=[shift, 2])

                dpg.add_combo(("ROI", "Line"), label="Mode", width=50, pos=[
                    shift, 27], callback=self.change_selection_mode, default_value="ROI")

                roi = self.setup_roi_buttons(
                    shift, 50, curr_dir, curr_name, state_manager)
                line = self.setup_line_buttons(shift, 50, state_manager)
                dpg.hide_item(line)

            post_line = self.setup_post_line_buttons(shift, 0, state_manager, curr_dir, curr_name, self.frame_width*self.frame_height)
            dpg.hide_item(post_line)

        return roi, line, roi_and_line_selection, post_line, state_manager