import dearpygui.dearpygui as dpg

from roipoly import RoiPoly
from roiinterface import ROIInterface
dpg.create_context()

class LineInterface:
    def __init__(self):
        self.lines = []
        self.multi_roi = None
        self.drag_point = None
        self.drag_line = None
        self.hover_line = None

class StateManager:
    def __init__(self):
        self.inactive = True
        self.current_roi = None
        self.roi_interface = ROIInterface()
        
    def __left_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__left_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__left_button_press_callback()
            
    def __right_button_press_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__right_button_press_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__right_button_press_callback()      
            
            if self.current_roi.completed:
                self.roi_interface.rois.append(self.current_roi)
                self.inactive = True
                self.current_roi = None
                                        
    def __motion_notify_callback(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__motion_notify_callback()
        elif not self.inactive:
            self.current_roi._RoiPoly__motion_notify_callback()
    
    def __new_roi(self):
        if self.current_roi != None and self.current_roi.completed != True:
            dpg.configure_item(self.current_roi, points=[])
                            
        self.current_roi = RoiPoly(window)
        self.inactive = False

    def __release(self):
        if self.inactive and len(self.roi_interface.rois) > 0:
            self.roi_interface._ROIInterface__left_release_callback()
        
state_manager = StateManager()

with dpg.theme() as canvas_theme, dpg.theme_component():
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)

with dpg.window() as window:
    
    with dpg.group(horizontal=True):

        with dpg.child_window(width=500, height=500) as canvas:
            dpg.bind_item_theme(canvas, canvas_theme)
            drawlist = dpg.add_drawlist(width=500, height=500)
        
            with dpg.item_handler_registry() as registry:
                dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__left_button_press_callback)
                dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=state_manager._StateManager__right_button_press_callback)
                    
            with dpg.handler_registry():
                dpg.add_mouse_move_handler(callback=state_manager._StateManager__motion_notify_callback)
                dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__release)

            dpg.bind_item_handler_registry(drawlist, registry)

        with dpg.child_window(border=False):
            dpg.add_button(label="New ROI", callback=state_manager._StateManager__new_roi)
            
dpg.set_primary_window(window, True)
dpg.create_viewport(width=900, height=600, title="Raindrops")
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    #r.lines = r.update_lines(r.lines)
    dpg.render_dearpygui_frame()
dpg.destroy_context()

