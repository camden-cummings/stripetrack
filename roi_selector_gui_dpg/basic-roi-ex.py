import dearpygui.dearpygui as dpg
from gui import GUI

dpg.create_context()

frame_width = 500
frame_height = 500

m = GUI("file_save_path", frame_width, frame_height)

while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()
    
dpg.destroy_context()