import math

import cv2


def check_masked_image(centroid, cen_mask):
    if cen_mask[centroid[1]][centroid[0]] == 255.0:
        return False
    else:
        return True

def generate_row_col(shape_of_rows):
    for row_num, item in enumerate(shape_of_rows):
        for col_num in range(item):
            yield row_num, col_num

def sort_contours_by_area(contours, last_confident_centroid, frame_count, time, diff, mask, shape_of_rows, cell_contours, cell_centers, min_thresh):
    # darkest_pixel_val = 255
    posns = [[[] for j in range(shape_of_rows[i])] for i in
             range(len(shape_of_rows))]
    sorted_contours = []

    # print(shape_of_rows)
    for c in contours:
        # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
        # a well, and that we know which one it is in)

        # contours = [c for c in contours if self.gui.contour_definer.centroid_size < cv2.contourArea(c) < 500000]
        if min_thresh < cv2.contourArea(c) < 500000:
            x, y, w, h = cv2.boundingRect(c)

            point_x, point_y = (int(x + w / 2), int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster

            if not check_masked_image((point_x, point_y), mask):
                for row, col in generate_row_col(shape_of_rows):
                    if math.dist(cell_centers[row][col], (point_x, point_y)) < 50:
                        cell_count = row * shape_of_rows[row] + col

                        in_polygon = cv2.pointPolygonTest(cell_contours[cell_count], (point_x, point_y),
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

    for row, col in generate_row_col(shape_of_rows):
        ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]
        if len(ten_darkest_centroids) > 0:
            sorted_contours.append(
                [time, frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1]])
            last_confident_centroid[row][col] = ten_darkest_centroids[0][0]
        else:
            sorted_contours.append(
                [time, frame_count, row, col, last_confident_centroid[row][col][0],
                 last_confident_centroid[row][col][1]]
            )

            # for c in ten_darkest_centroids:
            #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
    return sorted_contours