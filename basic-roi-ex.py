from dataclasses import dataclass
import dearpygui.dearpygui as dpg
dpg.create_context()

@dataclass
class Raindrop:
    x: int
    y: int
    size: int
    color: tuple
    
class RoiPoly:
    def __init__(self):
        self.line = None
        self.lines = []
        self.poly = None
        
        self.completed = False
        
        self.previous_point = []
        
    def __left_button_press_callback(self):
        x,y = dpg.get_mouse_pos(local=True)
        
        if self.completed is False:
            if self.line == None:
                self.line = [x,y], [x,y]
                
                [self.lines.append(i) for i in self.line]
                
                self.previous_point = [x, y]

                self.poly = dpg.draw_polyline(points=self.lines, color=(255, 0, 0, 255), thickness=1, parent=window)
    
            else:
                self.line = self.previous_point, [x,y]
                
                self.lines.append([x,y])
                self.previous_point = [x, y]
                
                dpg.configure_item(self.poly, points=self.lines)

    def __right_button_press_callback(self):
        dpg.configure_item(self.poly, points=self.lines[:-1], closed=True)
        self.completed = True

    def __motion_notify_callback(self):
        if self.line is not None and self.completed is False:
            x,y = dpg.get_mouse_pos(local=True)
    
            self.line = self.previous_point, [x,y]
            self.lines[-1] = [x,y]
            
            dpg.configure_item(self.poly, points=self.lines)

class Inactive:
    def __left_button_press_callback(self):
        pass

    def __right_button_press_callback(self):
        pass
    
    def __motion_notify_callback(self):
        pass

inactive_state = Inactive()

class StateManager:
    def __init__(self):
        self.inactive = True
        self.current_roi = None
        
    def __left_button_press_callback(self):
        if self.inactive:
            inactive_state._Inactive__left_button_press_callback()
        else:
            self.current_roi._RoiPoly__left_button_press_callback()
            
    def __right_button_press_callback(self):
        if self.inactive:
            inactive_state._Inactive__right_button_press_callback()
        else:
            self.current_roi._RoiPoly__right_button_press_callback()      
            
    def __motion_notify_callback(self):
        if self.inactive:
            inactive_state._Inactive__motion_notify_callback()
        else:
            self.current_roi._RoiPoly__motion_notify_callback()
    
    def __new_roi(self):
        self.current_roi = RoiPoly()
        self.inactive = False

state_manager = StateManager()

with dpg.theme() as canvas_theme, dpg.theme_component():
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0,0)

with dpg.window() as window:
    dpg.add_text("Click to create raindrops.")

    with dpg.group(horizontal=True):

        with dpg.child_window(width=500, height=500) as canvas:
            dpg.bind_item_theme(canvas, canvas_theme)
            drawlist = dpg.add_drawlist(width=500, height=500)
        
            with dpg.item_handler_registry() as registry:
                dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=state_manager._StateManager__left_button_press_callback)
                dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=state_manager._StateManager__right_button_press_callback)
                    
            with dpg.handler_registry():
                dpg.add_mouse_move_handler(callback=state_manager._StateManager__motion_notify_callback)

            dpg.bind_item_handler_registry(drawlist, registry)

        with dpg.child_window(border=False):
            speed = dpg.add_slider_int(label="Speed", width=200, default_value=1, min_value=1, max_value=10)
            color_picker = dpg.add_color_picker(width=200, default_value=(0,0,255,255))
            dpg.add_button(label="New ROI", callback=state_manager._StateManager__new_roi)
            
dpg.set_primary_window(window, True)
dpg.create_viewport(width=900, height=600, title="Raindrops")
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    #r.lines = r.update_lines(r.lines)
    dpg.render_dearpygui_frame()
dpg.destroy_context()

