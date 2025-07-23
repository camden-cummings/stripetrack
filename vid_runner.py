import cv2
import os
import numpy as np
import pickle
import math
from speedy_str_sim import run_correlate_rearr_y
from structural_sim_from_scratch import setup, generate_weights
from calc_mode import calc_mode_img
import time
import cProfile, pstats, io
from pstats import SortKey

# TODO what is going on with this file - should it be deleted? is it used??

def convert_to_contours(cell_filename):
    if isinstance(cell_filename, str):
        with open(cell_filename, 'rb') as f:
            rois = pickle.load(f)

    #            if isinstance(rois, RoiPoly):
    #                rois_dup = []
    #                for roi in rois:
    #                    rois_dup.append(roi.lines)
    #                rois = rois_dup
    else:
        rois = cell_filename

    centers = []
    contours = []

    for roi in rois:
        contour = np.array(roi, dtype='int')
        contours.append(contour)

        cx, cy = find_centroid_of_contour(contour)

        for center in centers:
            if math.dist(center[0], (cx, cy)) == 0.0:
                break
        else:
            centers.append([[cx, cy], contour])

    centers.sort(key=lambda tup: tup[0][1])

    row = 0
    col = 0
    reorg_centers = []
    curr_row = []

    # sorting by row  ------------------------
    for i in range(0, len(centers)):
        if centers[i][0][1] - centers[i - 1][0][1] > 30:
            row += 1
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row.copy())
            curr_row.clear()

        curr_row.append(centers[i])

        if i == len(centers) - 1:
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row)

    num_rows = len(reorg_centers)
    # ------------------------
    # TODO if not possible try to group by vertical alignment

    cell_contours = [[] for i in range(len(centers))]
    cell_centers = [[] for j in range(num_rows)]

    shape_of_rows = []

    for row in range(num_rows):
        num_cols = len(reorg_centers[row])
        shape_of_rows.append(num_cols)
        for col in range(num_cols):
            y_count = row * num_cols + col

            cell_centers[row].append(reorg_centers[row][col][0])
            cell_contours[y_count] = reorg_centers[row][col][1]

    return cell_contours, cell_centers, shape_of_rows

def find_centroid_of_contour(contour):
    """Given a contour, finds centroid of it."""
    M = cv2.moments(contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def get_contour_mask(cell_contours, frame_width, frame_height):
    contour_mask = np.zeros((frame_height, frame_width, 3))

    for c in cell_contours:
        #print(c)
        contour_mask = cv2.drawContours(contour_mask, [c],
                                        -1, (255, 255, 255), thickness=cv2.FILLED)

    contour_mask = cv2.cvtColor(
        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    return contour_mask

def vid_runner(vidcap, mode_img, weights, data_range, frame_width, frame_height, cov_norm):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img = curr_img.astype(np.float32, copy=False, order='C')
    mode_img = mode_img.astype(np.float32, copy=False, order='C')

    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(frame_width, frame_height, 'C')
    ux, uy, uxx, uyy, uxy = setup(frame_height, frame_width, 'C')

    cell_contours, cell_centers, shape_of_rows = convert_to_contours("/home/chamomile/Thyme-lab/data/vids/social_and_many_well/to-be-processed/march/fc2_save_2025-03-18-142549-0000.cells")
    contour_mask = get_contour_mask(cell_contours, frame_width, frame_height)

    masked_mode_noblur_img = cv2.bitwise_and(
        mode_img, mode_img, mask=contour_mask)
    masked_mode_noblur_img = masked_mode_noblur_img.astype(np.float32, copy=False, order='C')

    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1
    np_weights = np.array(weights, dtype=np.float32, order='C')

    #run_correlate_rearr(curr_img, mode_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
    #                    size2, width, height, cov_norm, data_range)

    run_correlate_rearr_y(curr_img, masked_mode_noblur_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp,
                          uy, uyy_tmp, uyy, size1,
                          size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size)

    frame_count = 0
    tottime = 0

    prev_curr_img = np.zeros((frame_height, frame_width), dtype=np.float32, order='C')
    pr = cProfile.Profile()
    pr.enable()
    while cont:
        #pr = cProfile.Profile()
        #pr.enable()


        t1 = time.time()

        masked_curr_img = cv2.bitwise_and(curr_img, curr_img, mask=contour_mask)
        masked_curr_img = masked_curr_img.astype(np.float32, copy=False, order='C')

        #run_correlate_rearr(masked_curr_img, masked_mode_noblur_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
        #                    size2, width, height, cov_norm, data_range)

        diff_s = run_correlate_rearr_y(masked_curr_img, prev_curr_img, ux_tmp, ux, uxx_tmp, uxx, uxy_tmp, uxy, uy_tmp, uy, uyy_tmp, uyy, size1,
                              size2, frame_width, frame_height, cov_norm, data_range, np_weights, weight_size)
        uy = ux.copy()
        uyy = uxx.copy()

        cv2.imshow('diff', diff_s)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        cont, curr_img = vidcap.read()
        #cv2.imshow('curr', curr_img)

        prev_curr_img = masked_curr_img

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('currstore', curr_img_store)

        curr_img = curr_img_store.astype(np.float32, copy=False)
#        cv2.imshow('curr?', curr_img)

        frame_count += 1
        t0 = time.time()
        total = t0 - t1
        tottime += total

        #pr.disable()
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())
        #print(tottime / frame_count)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    #print(tottime / frame_count)


if __name__ == '__main__':
    filename = "/home/chamomile/Thyme-lab/data/vids/social_and_many_well/to-be-processed/march/fc2_save_2025-03-18-142549-0000.mp4"
    mode_noblur_path = filename[:-4] + "-mode.png"

    vidcap = cv2.VideoCapture(filename)

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    if not os.path.exists(mode_noblur_path):
        calc_mode_img(vidcap, frame_width, frame_height, mode_noblur_path, False)

    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    weights, cov_norm = generate_weights(ndim=2, sigma=1.5, truncate=3.5)
    weights = weights.tolist()

    vid_runner(vidcap, mode_noblur_img, weights, 255, frame_width, frame_height, cov_norm)
