import cv2
from .precise_time import PreciseTime

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
    def __init__(self, shape_of_rows, cell_contours, cell_centers, cell_bounds, min_thresh=50):
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
        self.sorted_contours.clear()
        posns = [[[] for j in range(self.shape_of_rows[i])] for i in
                 range(len(self.shape_of_rows))]

        for c in contours:
            # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
            # a well, and that we know which one it is in)

            if self.min_thresh < cv2.contourArea(c) < 500000:
                x, y, w, h = cv2.boundingRect(c)

                point_x, point_y = (int(x + w / 2), int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster

                for row, col in generate_row_col(self.shape_of_rows):
                    cell_count = row * self.shape_of_rows[row] + col

                    minx, miny, maxx, maxy = self.cell_bounds[cell_count]
                    if minx < point_x < maxx and miny < point_y < maxy:

                        in_polygon = cv2.pointPolygonTest(self.cell_contours[cell_count], (point_x, point_y),
                                                          False)
                        if in_polygon >= 0:
                            n = cv2.mean(diff[y:y + h, x:x + w])[0]

                            posns[row][col].append(((point_x, point_y), n))

                            break

        for row, col in generate_row_col(self.shape_of_rows):
            ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]

            cell_count = row * self.shape_of_rows[row] + col
            x, y, x2, y2 = self.cell_bounds[cell_count]

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

        return self.sorted_contours