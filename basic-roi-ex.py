from dataclasses import dataclass
import dearpygui.dearpygui as dpg
from shapely.geometry import Point, Polygon

import math

import numpy as np

dpg.create_context()

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
        self.lines = self.lines[:-1]
        dpg.configure_item(self.poly, points=self.lines, closed=True)
        self.completed = True

    def __motion_notify_callback(self):
        if self.line is not None and self.completed is False:
            x,y = dpg.get_mouse_pos(local=True)
    
            self.line = self.previous_point, [x,y]
            self.lines[-1] = [x,y]
            
            dpg.configure_item(self.poly, points=self.lines)

class ROIInterface:
    def __init__(self):
        self.rois = []
        self.selected_polygon = None
        self.selected_polygon_vert = None
        self.drag_polygon = None

    def __left_button_press_callback(self):
        x,y = dpg.get_mouse_pos(local=True)

        if self.drag_polygon is None and self.selected_polygon_vert is None:
            self.check_for_selection((x,y))
        
    def __right_button_press_callback(self):
        pass
            
    def check_for_selection(self, mouse_pos):
        for poly in self.rois:
            mouse_pt = Point(mouse_pos)
            n_poly = Polygon(poly.lines)

            if mouse_pt.within(n_poly):
                self.drag_polygon = poly
                
            for point in poly.lines:
                if math.dist(point, mouse_pos) < 40:  # if
                    self.selected_polygon_vert = point
                    self.selected_polygon = poly

    def move(self):
        x,y = dpg.get_mouse_pos(local=True)
        
        poly = self.drag_polygon.lines
        centr = Polygon(poly).centroid

        translated_poly = [[p[0] - centr.x + x, p[1] - centr.y + y] for p in poly]
        
        self.drag_polygon.lines = translated_poly
        dpg.configure_item(self.drag_polygon.poly, points=self.drag_polygon.lines)
        
    def rotate(self, angle):
        poly = self.selected_polygon.lines
        centr = Polygon(poly).centroid
        
        """
        r = dpg.create_rotation_matrix(angle, [centr.x, centr.y])
        
        dpg.apply_transform(self.selected_polygon.poly, r)
        """
        
        rot_matrix = [[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]]
        rotated_poly = np.array([(p[0]-centr.x, p[1]-centr.y) for p in poly]).dot(
            rot_matrix) + [(centr.x, centr.y) for x in range(len(poly))]
        
        rotated_poly = rotated_poly.tolist()
        
        self.selected_polygon.lines = rotated_poly
        dpg.configure_item(self.selected_polygon.poly, points=self.selected_polygon.lines)
        
    def find_future_posn(self, cursor_posn: tuple[int, int], centr: tuple[int, int]) \
        -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Intermediate calculations for finding the next point the shape will
        be rotated to on the circle of positions that the currently selected
        point could go to.

        Parameters
        ----------
        cursor_posn : current mouse position
        centr : center of the polygon being rotated

        Returns
        -------
        v : current position on the circle of the selected vertex
        future_v : future position on the circle of the selected vertex


        """


        # find closest point on the circle of potential vertex positions
        curr_v = (cursor_posn[0]-centr.x, cursor_posn[1]-centr.y)
        curr_mag_v = math.sqrt(
            curr_v[0]*curr_v[0] + curr_v[1]*curr_v[1])
        radius = math.dist((centr.x, centr.y), self.selected_polygon_vert)
        future_v = (centr.x + curr_v[0] / curr_mag_v * radius,
                    centr.y + curr_v[1] / curr_mag_v * radius)

        v = (2 * (radius**2) - math.dist(future_v,
                                         cursor_posn) ** 2) / (2*radius*radius)

        return v, future_v
    
    def introduce_dir(self, theta: int, future_v: tuple[int, int], centr: tuple[int, int]) -> int:
        """
        Chooses direction for theta.

        Parameters
        ----------
        theta : angle used for rotation
        future_v : vertex
        centr : center of the polygon being rotated

        Returns
        -------
        theta : angle used for rotation, now with pos/neg value indicating direction
        
        """
        d_x = (
            (self.selected_polygon_vert[0]-centr.x) - (future_v[0]-centr.x)) > 0
        d_y = (
            (self.selected_polygon_vert[1]-centr.y) - (future_v[1]-centr.y)) > 0

        if self.selected_polygon_vert[1]-centr.y > 0:
            if self.selected_polygon_vert[0]-centr.x > 0:
                if d_x and not d_y:
                    theta = -theta
            else:
                if d_x and d_y:
                    theta = -theta

        else:

            if self.selected_polygon_vert[0]-centr.x > 0:
                if not d_x and not d_y:
                    theta = -theta
            else:
                if d_y and not d_x:
                    theta = -theta

        return theta
    
    def __motion_notify_callback(self):
        x,y = dpg.get_mouse_pos(local=True)

        if self.drag_polygon is not None:
            self.move()
        elif self.selected_polygon is not None:
            centr = Polygon(self.selected_polygon.lines).centroid
    
            v, future_v = self.find_future_posn((x,y), centr)
    
            if -1 <= v <= 1:
                # doing it this way means we need to reintroduce directionality / pos neg
                theta = math.acos(v)
                theta = self.introduce_dir(theta, future_v, centr)
                
                self.rotate(theta)

                        
    def __left_release_callback(self):
        self.drag_polygon = None
        self.selected_polygon_vert = None
        self.selected_polygon = None

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
                            
        self.current_roi = RoiPoly()
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

