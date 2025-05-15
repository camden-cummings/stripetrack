# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:30:01 2025

@author: ThymeLab
"""

import numba as nb
import numpy as np
import math
import cProfile, pstats, io
from pstats import SortKey
import cv2
from scipy.ndimage import correlate1d as correlate1d_scipy

#from skimage.metrics import structural_similarity
from structural_sim import structural_similarity
from structural_sim_from_scratch import setup, generate_weights, correlate1d, run_math
#from numba import int64, float64
from scipy import ndimage as ndi
from numbers import Integral
import copy

def structural_similarity_diff(
        im1,
        im2,
        data_range,
        weights,
        cov_norm):

    # ndimage filters need floating point data
    im1 = im1.astype(np.float64, copy=False)
    im2 = im2.astype(np.float64, copy=False)

    frame_width, frame_height = 1200, 1760
    ux = np.zeros((frame_width, frame_height))
    uy = np.zeros((frame_width, frame_height))
    uxx = np.zeros((frame_width, frame_height))
    uyy = np.zeros((frame_width, frame_height))
    uxy = np.zeros((frame_width, frame_height))
    #r = np.zeros((frame_width, frame_height))
    
    correlate1d(im1, weights, ux)
    correlate1d(im2, weights, uy)

    correlate1d(im1 * im1, weights, uxx)
    correlate1d(im2 * im2, weights, uyy)
    correlate1d(im1 * im2, weights, uxy)
    
    return run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy, )

def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f'crop_width has an invalid length: {len(crop_width)}\n'
                f'crop_width should be a sequence of N pairs, '
                f'a single pair, or a single integer'
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f'crop_width has an invalid length: {len(crop_width)}\n'
            f'crop_width should be a sequence of N pairs, '
            f'a single pair, or a single integer'
        )

    slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped

@nb.njit(parallel=True, fastmath=True)
def run_math_with_diff(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    K1 = 0.01
    K2 = 0.03

    ux_squared = ux * ux
    uy_squared = uy * uy
    ux_uy = ux * uy
    vx = cov_norm * (uxx - ux_squared)
    vy = cov_norm * (uyy - uy_squared)
    vxy = cov_norm * (uxy - ux_uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (
        2 * ux_uy + C1,
        2 * vxy + C2,
        ux_squared + uy_squared + C1,
        vx + vy + C2,
    )

    #ret[:][:] = (A1 * A2) / (B1 * B2)
    return  (((A1 * A2) / (B1 * B2))*255).astype("uint8")

"""
@nb.njit(parallel=True, fastmath=True)
def correlate1d(input, weights, output=None):
    height, width = (1200, 1760)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    """"""
    symmetric = 0
    if weight_size % 2 == 1:  # if the input weight array is even, it will be symmetric = 0, so we don't need to run this calculation
        if all(weights == weights[::-1]):
            symmetric = 1
        elif all(weights == -weights[::-1]):  # i believe this is ok but don't trust it 100%
            symmetric = -1
    """"""
    symmetric = 1

    for ii in nb.prange(height):
        np_row = input[ii]

        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]

        new_arr = np.concatenate((size1_arr, np_row, size2_arr))

        if symmetric > 0:
            for start in range(width):
                n = start+size1
                total_neighbour = weights[size1]*new_arr[n]
                for x in range(1, size1+1):
                    total_neighbour += (new_arr[n+x] + new_arr[n-x]) * weights[size1+x]

                output[ii][start] = total_neighbour

        elif symmetric < 0:
            pass
        else:
            for start in range(len(new_arr) - size1 - 1):
                output[ii][start] = new_arr[start + size1 + 1] * weights[-1]
                for i in range(start, start + size1 + 1):
                    output[ii][start] += new_arr[i] * weights[i - start]
"""

def run(diff):
    return (diff * 255).astype("uint8")

def tester(matr_to_check, correct_matrix):
    bool_arr = matr_to_check[:, :] == correct_matrix[:, :]
    val = len(bool_arr[bool_arr == False])
    print(val)
    if val > 0:
        arr = np.argwhere(bool_arr == False)

        for posn in arr:
            n = correct_matrix[posn[0], posn[1]]
            nd = matr_to_check[posn[0], posn[1]]
            if abs(n - nd) > 0.5:
                print(n, nd, abs(n - nd))

def vid_runner(vidcap, mode_img, weights, data_range):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img = curr_img.astype(np.float64, copy=False)
    mode_img = mode_img.astype(np.float64, copy=False)

    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    ndim = curr_img.ndim
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    ux, uy, uxx, uyy, uxy = setup(curr_img, mode_img, weights, 1760,1200)
    ux_tmp, uy_tmp, uxx_tmp, uyy_tmp, uxy_tmp = setup(curr_img, mode_img, weights, 1760, 1200)

    correlate1d(curr_img, weights, ux, 0)  # , curr_scipy)
    S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

    pr = cProfile.Profile()
    pr.enable()

    frame_count = 0
    while cont and frame_count < 100:
        axis = 0
        #print("curr")

        #curr_scipy = correlate1d_scipy(curr_img, weights, axis=axis, mode="reflect", cval=0.0)
        correlate1d(curr_img, weights, ux_tmp, 0)#, curr_scipy)
        correlate1d(ux_tmp, weights, ux, 1)#, curr_scipy)

        #tester(ux, curr_scipy)

        #cv2.imshow("corr", ux)
        #print("mode")
        #mode_scipy = correlate1d_scipy(mode_img, weights, axis=axis, mode="reflect", cval=0.0)
        correlate1d(mode_img, weights, uy_tmp, 0)#, mode_scipy)
        correlate1d(uy_tmp, weights, uy, 1)#, mode_scipy)

        #tester(uy, mode_scipy)

        #print("cc")
        #cc_scipy = correlate1d_scipy(curr_img*curr_img, weights, axis=axis, mode="reflect", cval=0.0)
        correlate1d(curr_img * curr_img, weights, uxx_tmp, 0)#, cc_scipy)
        correlate1d(uxx_tmp, weights, uxx, 1)#, mode_scipy)
        #tester(uxx, cc_scipy)

        #print("mm")
        #mm_scipy = correlate1d_scipy(mode_img*mode_img, weights, axis=axis, mode="reflect", cval=0.0)
        correlate1d(mode_img * mode_img, weights, uyy_tmp, 0)#, mm_scipy)
        correlate1d(uyy_tmp, weights, uyy, 1)#, mode_scipy)
        #tester(uyy, mm_scipy)

        #print("cm")
        #cm_scipy = correlate1d_scipy(curr_img*mode_img, weights, axis=axis, mode="reflect", cval=0.0)
        correlate1d(curr_img * mode_img, weights, uxy_tmp, 0)#, cm_scipy)
        correlate1d(uxy_tmp, weights, uxy, 1)#, mode_scipy)
        #tester(uxy, cm_scipy)

        S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)
        #corr_scipy = correlate1d_scipy(curr_img, weights, axis=-1, mode="reflect", cval=0.0)
        #bool_arr = np.round(ux[:, :]) == np.round(corr_scipy[:, :])
        #print(len(bool_arr[bool_arr == False]))

        #_, S_strsim = structural_similarity(curr_img, mode_img, full=True, data_range=data_range, gaussian_weights=True)

        diff = (S * 255).astype("uint8")
        cv2.imshow('diff', diff)
        #diff_strsim = (S_strsim * 255).astype("uint8")

        #cv2.imshow('diffstrsim', diff_strsim)

        #bool_arr = np.round(diff[:, :]) == np.round(diff_strsim[:, :])
        #val = len(bool_arr[bool_arr == False])
        #if val > 0:
        #    arr = np.argwhere(bool_arr == False)

        #   for posn in arr:
        #        print(posn, diff_strsim[posn[0], posn[1]], diff[posn[0], posn[1]])


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        cont, curr_img = vidcap.read()

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float64, copy=False)

        frame_count += 1

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == '__main__':
    filename = "/home/chamomile/Thyme-lab/data/shortened_vids/6dpf/_2024-02-27-105322-0000-short.avi"
    mode_noblur_path = filename[:-4] + "-mode.png"
    mode_noblur_img = cv2.cvtColor(cv2.imread(mode_noblur_path), cv2.COLOR_BGR2GRAY)

    vidcap = cv2.VideoCapture(filename)
    weights = generate_weights(2, sigma=1.5, truncate=3.5)[0].tolist()

    vid_runner(vidcap, mode_noblur_img, weights, 255)

"""
if __name__ == '__main__':
    frame_width, frame_height = 660, 992

    ux = np.zeros((frame_width, frame_height))
    uy = np.zeros((frame_width, frame_height))
    uxx = np.zeros((frame_width, frame_height))
    uyy = np.zeros((frame_width, frame_height))
    uxy = np.zeros((frame_width, frame_height))

    img1 = np.zeros((frame_width, frame_height))
    img2 = np.ones((frame_width, frame_height))
    weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
     0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]
    ndim = img1.ndim

    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    cov_norm = win_size ** ndim / (win_size ** ndim - 1)  # sample covariance

    data_range = 255

    correlate1d(img1, weights, ux)
    correlate1d(img2, weights, uy)

    correlate1d(img1 * img1, weights, uxx)
    correlate1d(img2 * img2, weights, uyy)
    correlate1d(img1 * img2, weights, uxy)

    run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy, )
    run_math_with_diff(cov_norm, data_range, ux, uy, uxx, uyy, uxy, )

    structural_similarity_diff(
            img1,
            img2,
            255,
            weights,
            cov_norm)

#    pr = cProfile.Profile()
#    pr.enable()

#    for i in range(100):
        structural_similarity_diff(
                img1,
                img2,
                data_range,
                weights,
                cov_norm)

        np.random.seed(i)
        img1 = np.random.randint(100, size=(frame_width, frame_height))
        img2 = np.random.randint(101, size=(frame_width, frame_height))

        correlate1d(img1, weights, ux)
        correlate1d(img2, weights, uy)

        correlate1d(img1 * img1, weights, uxx)
        correlate1d(img2 * img2, weights, uyy)
        correlate1d(img1 * img2, weights, uxy)

        diff = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy, )
        diff = run(diff)
        diff_w = run_math_with_diff(cov_norm, data_range, ux, uy, uxx, uyy, uxy, )

        #for x in range(frame_width):
        #    for y in range(frame_height):
        #        if diff[x][y] != diff_w[x][y]:
        #            print(diff[x][y], diff_w[x][y])
        bool_arr = diff[:, :] == diff_w[:, :]
        print(bool_arr[bool_arr == False])
        ##if all(all()):
         #   print("!")

    #    print(diff, diff_w)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

pr = cProfile.Profile()
pr.enable()

for i in range(50):
    structural_similarity(
            img1,
            img2,
            data_range=255)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
"""