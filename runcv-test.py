from speedy_str_sim import correlate1d, run_math #structural_similarity
import cv2
import numpy as np
from tracker.helpers.centroid_manip import find_centroid_of_contour, check_masked_image, generate_row_col
import pickle

# fuck this
# is not working

weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
 0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]
np_weights = np.asarray(weights)

FRAMES_TO_SAVE_AFTER = 100
sigma = 1.5
truncate = 3.5
r = int(truncate * sigma + 0.5)  # radius as in ndimage
win_size = 2 * r + 1

fn = "/home/chamomile/Thyme-lab/data/vids/smart-dumb-run-fc2_save_2025-02-06-151144-0000.mp4"
vidcap = cv2.VideoCapture(fn)
with open(fn.split(".")[0] + ".cells", 'rb') as filename:
    cell_contours = pickle.load(filename)

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

contour_mask = np.zeros((frame_height, frame_width, 3), dtype=np.int32)

for c in cell_contours:
    non_int = np.array(c, np.int32)
    contour_mask = cv2.drawContours(contour_mask, [non_int],
                                    -1, (255, 255, 255), thickness=cv2.FILLED)

mask = cv2.cvtColor(np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
cont, curr_img = vidcap.read()
masked_mode_noblur_img = cv2.bitwise_and(curr_img, curr_img, mask=mask)
masked_mode_noblur_img = cv2.cvtColor(masked_mode_noblur_img.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
shape_of_rows = [4,4,4]

def run_CV(masked_curr_img, frame_count, time, ux, uy, uxx, uyy, uxy):
    contours, diff_box = find_centroids(masked_curr_img, ux, uy, uxx, uyy, uxy)
    sorted_contours = sort_contours_by_area(contours, frame_count, time, diff_box, shape_of_rows, cell_contours, mask)

    for i in contours:
        center = find_centroid_of_contour(i)
        cv2.circle(masked_curr_img, center, 1, (0, 255, 0), 1)

def find_centroids(curr_img_gray, ux, uy, uxx, uyy, uxy):
    ndim = masked_mode_noblur_img.ndim

    # ndimage filters need floating point data
    curr_img_gray = curr_img_gray.astype(np.uint8, copy=False)

    correlate1d(curr_img_gray, weights, ux)
    correlate1d(masked_mode_noblur_img, weights, uy)

    correlate1d(curr_img_gray * curr_img_gray, weights, uxx)
    correlate1d(masked_mode_noblur_img * masked_mode_noblur_img, weights, uyy)
    correlate1d(curr_img_gray * masked_mode_noblur_img, weights, uxy)

    cov_norm = win_size ** ndim / (win_size ** ndim - 1)  # sample covariance
    diff = run_math(cov_norm, 255, ux, uy, uxx, uyy, uxy)

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # thresh_img = cv2.threshold(diff, gui.contour_definer.thresh, 255, cv2.THRESH_BINARY)[1]

    thresh_img = cv2.threshold(diff, 155, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #    cv2.drawContours(diff, contours, -1, (0,255,0), 1)
    # contours = [c for c in contours if gui.contour_definer.centroid_size < cv2.contourArea(c) < 500000]
    contours = [c for c in contours if 70 < cv2.contourArea(c) < 500000]

    return contours, diff_box

def sort_contours_by_area(contours, frame_count, time, diff_box, shape_of_rows, cell_contours, mask):
    posns = [[[] for j in range(shape_of_rows[i])] for i in
             range(len(shape_of_rows))]

    sorted_contours = []

    for c in contours:
        # we want to make sure that each contour we find is in a masked section of the image (i.e. relevant because it's in
        # a well, and that we know which one it is in)

        x, y, w, h = cv2.boundingRect(c)

        point_x, point_y = (int(x + w / 2), int(y + h / 2))  # not as exact as find_centroid_of_contour, but faster

        if not check_masked_image((point_x, point_y), mask):
            for row, col in generate_row_col(shape_of_rows):
                cell_count = row * shape_of_rows[row] + col

                in_polygon = cv2.pointPolygonTest(np.array(cell_contours[cell_count], dtype=np.float32), (point_x, point_y),
                                                  False)

                if in_polygon >= 0:
                    # TODO this looks extremely speed-up-able, are we just taking the mean of some numbers that will be the same every run?
                    R, G, B, _ = np.array(cv2.mean(diff_box[y:y + h, x:x + w])).astype(np.uint8)
                    gray_avg = 0.299 * R + 0.587 * G + 0.114 * B

                    posns[row][col].append(((point_x, point_y), gray_avg))

                    break
                    # if gray_avg < darkest_pixel_val:
                    #   darkest_pixel_val = gray_avg

    for row, col in generate_row_col(shape_of_rows):
        ten_darkest_centroids = sorted(posns[row][col], key=lambda posn: posn[1])[:10]
        if len(ten_darkest_centroids) > 0:
            sorted_contours.append(
                [time, frame_count, row, col, ten_darkest_centroids[0][0][0], ten_darkest_centroids[0][0][1]])

            # for c in ten_darkest_centroids:
            #    all_centr_in_frame.append([frame_count, row, col, c[0][0], c[0][1]])
    return sorted_contours

if __name__ == '__main__':
    frame_count = 0

    ux = np.zeros((frame_width, frame_height))
    uy = np.zeros((frame_width, frame_height))
    uxx = np.zeros((frame_width, frame_height))
    uyy = np.zeros((frame_width, frame_height))
    uxy = np.zeros((frame_width, frame_height))

    time = 0
    while cont:
        cont, curr_img = vidcap.read()

        masked_curr_img = cv2.bitwise_and(curr_img, curr_img, mask=mask)
        masked_curr_img = cv2.cvtColor(masked_curr_img, cv2.COLOR_BGR2GRAY)
        run_CV(masked_curr_img, frame_count, time, ux, uy, uxx, uyy, uxy)

        cv2.imshow("test", curr_img)

        if cv2.waitKey(1) == 1:
            break

        frame_count += 1
#    masked_curr_img = cv2.bitwise_and(
#        curr_img, curr_img, mask=mask)
