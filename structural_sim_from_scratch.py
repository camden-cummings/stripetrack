import numba as nb
import numpy as np
import math
import cv2
import cProfile, pstats, io
from pstats import SortKey
import scipy.ndimage as ndi
#import tensorflow as tf

#from tensorflow.python.framework.ops import enable_eager_execution
#enable_eager_execution()


def structural_similarity(
        im1,
        im2,
        data_range):

    # ndimage filters need floating point data
    im1 = im1.astype(np.float64, copy=False)
    im2 = im2.astype(np.float64, copy=False)

    weights, cov_norm = generate_weights(im1.ndim)
    ux, uy, uxx, uyy, uxy = setup(im1, im2, weights, 1760, 1200)

    return run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

def generate_weights(ndim, sigma=1.5, truncate=3.5):
    radius = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * radius + 1
    NP = win_size**ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    weights = ndi._filters._gaussian_kernel1d(sigma, 0, radius)[::-1]
    return weights, cov_norm

def setup(frame_width, frame_height):
    ux = np.ascontiguousarray(np.zeros((frame_height, frame_width)))
    uy = np.ascontiguousarray(np.zeros((frame_height, frame_width)))
    uxx = np.ascontiguousarray(np.zeros((frame_height, frame_width)))
    uyy = np.ascontiguousarray(np.zeros((frame_height, frame_width)))
    uxy = np.ascontiguousarray(np.zeros((frame_height, frame_width)))

    return ux, uy, uxx, uyy, uxy

def vid_runner(vidcap, mode_img, weights, data_range):
    cont, curr_img = vidcap.read()
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    curr_img_store = np.zeros((1200, 1760))
    curr_img = curr_img.astype(np.float64, copy=False)
    mode_img = mode_img.astype(np.float64, copy=False)

    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    ndim = curr_img.ndim
    NP = win_size**ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    ux, uy, uxx, uyy, uxy = setup(curr_img, mode_img, weights)
    S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

    pr = cProfile.Profile()
    pr.enable()

    while cont:
        correlate1d(curr_img, weights, ux)
        correlate1d(mode_img, weights, uy)

        correlate1d(curr_img * curr_img, weights, uxx)
        correlate1d(mode_img * mode_img, weights, uyy)
        correlate1d(curr_img * mode_img, weights, uxy)

        S = run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy)

        diff = (S * 255).astype("uint8")
        cv2.imshow('diff', diff)
        thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]

        #scipy_contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #scipy_contours = scipy_contours[0] if len(scipy_contours) == 2 else scipy_contours[1]

        #if len(scipy_contours) > 0:
        #    cv2.drawContours(curr_img_store, scipy_contours, -1, (0,255,0), 1)
            #cv2.drawContours(curr_img, scipy_contours, -1, (255,0,0),1)
            #cv2.imshow('f', curr_img_store)
        
       # cv2.imshow('f', thresh)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        cont, curr_img = vidcap.read()

        curr_img_store = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img = curr_img_store.astype(np.float64, copy=False)


    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

#@nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
@nb.njit(parallel=True, fastmath=True)
def run_math(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
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

    return (A1 * A2) / (B1 * B2)

#@nb.njit(parallel=True, fastmath=True)
def correlate1d(input, weights, output=None, axis=0, correct_arr=None):
    height, width = (1200, 1760)
    #print(input.shape)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    """
    symmetric = 0
    if weight_size % 2 == 1:  # if the input weight array is even, it will be symmetric = 0, so we don't need to run this calculation
        if all(weights == weights[::-1]):
            symmetric = 1
        elif all(weights == -weights[::-1]):  # i believe this is ok but don't trust it 100%
            symmetric = -1
    """

    symmetric = 1

    if axis == 0:
        for jj in nb.prange(width):
            np_row = input[:, jj]

            size1_arr = np_row[0:size1][::-1]
            size2_arr = np_row[-size2:][::-1]
            new_arr = np.concatenate((size1_arr, np_row, size2_arr))
            if symmetric > 0:
                for start in range(height):
                    n = start+size1
                    total_neighbour = weights[size1]*new_arr[n]
                    for x in range(1, size1+1):
                        total_neighbour += (new_arr[n+x] + new_arr[n-x]) * weights[size1+x]
                    output[start][jj] = total_neighbour
    elif axis == 1:
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

                    #if correct_arr is not None and abs(correct_arr[ii][start] - output[ii][start]) > 0.2:
                    #    print(correct_arr[ii][start], output[ii][start])

            elif symmetric < 0:
                pass
            else:
                for start in range(len(new_arr) - size1 - 1):
                    output[ii][start] = new_arr[start + size1 + 1] * weights[-1]
                    for i in range(start, start + size1 + 1):
                        output[ii][start] += new_arr[i] * weights[i - start]

@nb.njit(parallel=True, fastmath=True)
def correlate1d_x(input, weights, output=None, correct_arr=None):
    height, width = (1200, 1760)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    symmetric = 1

    rearr = np.concatenate((input[0:size1][::-1], input, input[-size2:][::-1]))

    for start in nb.prange(height):  # could end early by checking that all vals in arr are the same in which case will be the value
        #print(start)
        end = start + size1 + size2 + 1
        np.dot(rearr[start:end].transpose(), np.array(weights, dtype=np.float64), out=output[start])
        #print(n)
        #output[start] = np.dot(rearr[start:end].transpose(), np.array(weights, dtype=np.float32), axes=1)
        #for ii in range(width):
        #    if correct_arr is not None and abs(correct_arr[start][ii] - output[start][ii]) > 0.2:
        #        print(correct_arr[start][ii], output[start][ii])
    """ 
    for jj in range(width):
        new_arr = rearr[:, jj]
        
        if symmetric > 0:
            for start in range(height):  # could end early by checking that all vals in arr are the same in which case will be the value
                n = start + size1

                arr = np.array([new_arr[i] for i in range(n-size1, n+size1+1)])
                output[start][jj] = np.tensordot(arr, weights, axes=1)
    """

#@nb.njit(parallel=True, fastmath=True)
def correlate1d_y(input, weights, output=None):
    #pr = cProfile.Profile()
    #pr.enable()

    height, width = (1200, 1760)
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    symmetric = 1

    rearr = np.concatenate((input[:, 0:size1][:,::-1], input, input[:, -size2:][:,::-1]), axis=1)

    for start in range(width):
        # print(start)
        end = start + size1 + size2 + 1

        np.dot(rearr[:, start:end], np.array(weights, dtype=np.float64), out=output[start])

    #pr.disable()
    #s = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())

"""
        if symmetric > 0:
            for start in range(width):
                n = start + size1

                arr = np.array([new_arr[i] for i in range(n-size1, n+size1+1)])
                output[ii][start]  = np.tensordot(arr, weights, axes=1)
"""
"""
    if correct_arr is not None:
        bool_arr = output[:, :] == correct_arr[:, :]
        val = len(bool_arr[bool_arr == False])
        print(val)
        if val > 0:
            arr = np.argwhere(bool_arr == False)

            for posn in arr:
                n = correct_arr[posn[0], posn[1]]
                nd = output[posn[0], posn[1]]
                if abs(n - nd) > 0.01:
                    print(n, nd, abs(n - nd))
    """
if __name__ == '__main__':
    fn = "/home/chamomile/Thyme-lab/data/vids/smart-dumb-run-fc2_save_2025-02-06-151144-0000.mp4"
    vidcap = cv2.VideoCapture(fn)
    _, mode_img = vidcap.read()
    mode_img = cv2.cvtColor(mode_img, cv2.COLOR_BGR2GRAY)

    weights = [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
               0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838]

    vid_runner(vidcap, mode_img, weights, 255)