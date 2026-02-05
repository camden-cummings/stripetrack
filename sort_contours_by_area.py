import math

import cProfile
import io
import pstats
from pstats import SortKey
from precise_time import PreciseTime

import cv2

timer = PreciseTime()
        
def check_masked_image(centroid, cen_mask):
    if cen_mask[centroid[1]][centroid[0]] == 255.0:
        return False
    else:
        return True

def generate_row_col(shape_of_rows):
    for row_num, item in enumerate(shape_of_rows):
        for col_num in range(item):
            yield row_num, col_num

class SortContours:
    def __init__(self, mask, shape_of_rows, cell_contours, cell_centers, cell_bounds, min_thresh=50):
        self.mask = mask
        self.shape_of_rows = shape_of_rows
        self.cell_contours = cell_contours
        self.cell_centers = cell_centers
        self.cell_bounds = cell_bounds
        
        self.min_thresh = min_thresh
        
        self.last_confident_centroid = [[cell_centers[i][j] for j in range(shape_of_rows[i])] for i in range(len(shape_of_rows))]
        self.sorted_contours = []
        
        self.contours = []
        
    def set_contours(self, contours):
        self.contours = contours
    
    def set_diff(self, diff):
        self.diff = diff
        
    def sort_contours_by_area(self, contours, frame_count, curr_time, diff, dpix):
        # darkest_pixel_val = 255
        
        #pr = cProfile.Profile()
        #pr.enable()
        #t1 = timer.now()
        
        self.sorted_contours.clear()
        posns = [[[] for j in range(self.shape_of_rows[i])] for i in
                 range(len(self.shape_of_rows))]

    
        # print(shape_of_rows)
        for c in contours:
            # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
            # a well, and that we know which one it is in)
    
            # contours = [c for c in contours if self.gui.contour_definer.centroid_size < cv2.contourArea(c) < 500000]
            if self.min_thresh < cv2.contourArea(c) < 500000:
                x, y, w, h = cv2.boundingRect(c)
    
                point_x, point_y = (int(x + w / 2), int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster
    
                
                #if not check_masked_image((point_x, point_y), self.mask):
                for row, col in generate_row_col(self.shape_of_rows):
                        #if math.dist(self.cell_centers[row][col], (point_x, point_y)) < 50:
                    minx, miny, maxx, maxy = self.cell_bounds[row][col]
                    if minx < point_x < maxx and miny < point_y < maxy:
                
                #if not check_masked_image((point_x, point_y), self.mask):
                #    for row, col in generate_row_col(self.shape_of_rows):
                #        if math.dist(self.cell_centers[row][col], (point_x, point_y)) < 50:
                        cell_count = row * self.shape_of_rows[row] + col

                        in_polygon = cv2.pointPolygonTest(self.cell_contours[cell_count], (point_x, point_y),
                                                          False)
                        # print(in_polygon)
                        if in_polygon >= 0:
                            # print(diff_box)
                            # R, G, B, _ = np.array(cv2.mean(diff_box[y:y + h, x:x + w]), np.uint8)
                            n = cv2.mean(diff[y:y + h, x:x + w])[0]
                            # gray_avg = 0.299 * R + 0.587 * G + 0.114 * B
                            # print(R, G, B, n, gray_avg)
                            posns[row][col].append(((point_x, point_y), n))

                            break
                            # if gray_avg < darkest_pixel_val:
                            #   darkest_pixel_val = gray_avg

        for row, col in generate_row_col(self.shape_of_rows):
            ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]
            
            x, y, x2, y2 = self.cell_bounds[row][col]
            cell = dpix[y:y2, x:x2]
            dpix_count = len(cell[cell>0])
            
            if len(ten_darkest_centroids) > 0:
                self.sorted_contours.append(
                    [curr_time, frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1], dpix_count])
                self.last_confident_centroid[row][col] = ten_darkest_centroids[0][0]
            else:
                self.sorted_contours.append(
                    [curr_time, frame_count, row, col, self.last_confident_centroid[row][col][0],
                     self.last_confident_centroid[row][col][1], dpix_count]
                )
    
                # for c in ten_darkest_centroids:
                #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
        
        #print(timer.now() - t1)
        #pr.disable()
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print("tracker pool", s.getvalue())
        
        return self.sorted_contours