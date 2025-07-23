import numba as nb
import numpy as np
import math
import scipy.ndimage as ndi

def generate_weights(ndim, sigma=1.5, truncate=3.5):
    radius = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * radius + 1
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance

    weights = ndi._filters._gaussian_kernel1d(sigma, 0, radius)[::-1]
    return weights, cov_norm


def setup(width, height, order):
    ux = np.zeros((height, width), dtype=np.float32, order=order)
    uy = np.zeros((height, width), dtype=np.float32, order=order)
    uxx = np.zeros((height, width), dtype=np.float32, order=order)
    uyy = np.zeros((height, width), dtype=np.float32, order=order)
    uxy = np.zeros((height, width), dtype=np.float32, order=order)

    return ux, uy, uxx, uyy, uxy


# @nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
"""
@nb.njit(parallel=True, fastmath=True)
def run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy):
    ux_squared = ux * ux
    uy_squared = uy * uy
    ux_uy = ux * uy
    vx = cov_norm * (uxx - ux_squared)
    vxy = cov_norm * (uxy - ux_uy)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1, A2, B1, B2 = (
        2 * ux_uy + C1,
        2 * vxy + C2,
        ux_squared + uy_squared + C1,
        vx + vy + C2,
    )

    return (A1 * A2) / (B1 * B2)
"""

@nb.njit(parallel=True, fastmath=True)
def run_math(cov_norm, data_range, ux, uy, uxx, vy, uxy):
    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)
    vx = np.multiply(cov_norm, (uxx-ux_squared))
    vxy = np.multiply(cov_norm, (uxy - ux_uy))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1 = np.multiply(ux_uy,2) + C1
    A2 = np.multiply(vxy,2) + C2
    B1 = ux_squared + uy_squared + C1
    B2 = vx + vy + C2

    return (A1 * A2) / (B1 * B2)

#@nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
@nb.njit(parallel=True, fastmath=True)
def run_math_(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    ux_squared = np.multiply(ux, ux)
    uy_squared = np.multiply(uy, uy)
    ux_uy = np.multiply(ux, uy)
    vx = np.multiply(cov_norm, (uxx - ux_squared))
    vxy = np.multiply(cov_norm, (uxy - ux_uy))
    vy = np.multiply(cov_norm, (uyy - uy_squared))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1 = 2 * ux_uy + C1
    A2 = 2 * vxy + C2
    B1 = ux_squared + uy_squared + C1
    B2 = vx + vy + C2

    return (A1 * A2) / (B1 * B2)


@nb.njit(fastmath=True)
def normalize_diff(diff, width, height):
    for x in range(height):
        for y in range(width):
            if diff[x][y] > 1:
                diff[x][y] = 1
            elif diff[x][y] < 0:
                diff[x][y] = 0

            diff[x][y] *= 255

    diff = diff.astype("uint8")
    return diff


@nb.njit(parallel=True, fastmath=True)
def correlate1d_x(input, weights, output, width, height):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[0:size1][::-1], input, input[-size2:][::-1]))
    for jj in nb.prange(width):
        np_row = input[:, jj]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))
        # new_arr = rearr[:, jj]

        for start in range(height):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[start][jj] = total_neighbour

@nb.njit(parallel=True, fastmath=True)
def correlate1d_y(input, weights, output, width, height):
    weight_size = len(weights)
    size1 = math.floor(weight_size / 2)
    size2 = weight_size - size1 - 1

    # rearr = np.concatenate((input[:, 0:size1][::-1], input, input[:, -size2:][::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
    for ii in nb.prange(height):
        np_row = input[ii]
        size1_arr = np_row[0:size1][::-1]
        size2_arr = np_row[-size2:][::-1]
        new_arr = np.concatenate((size1_arr, np_row, size2_arr))

        for start in range(width):
            n = start + size1
            total_neighbour = weights[size1] * new_arr[n]
            for x in range(1, size1 + 1):
                total_neighbour += (new_arr[n + x] + new_arr[n - x]) * weights[size1 + x]
            output[ii][start] = total_neighbour

@nb.njit(parallel=True, fastmath=True)
def correlate1d_x_r(rearr, weights, weight_size, output, width, height):    
    for start in nb.prange(height):
        end = start+weight_size
        new_arr = rearr[start:end].transpose()

        np.dot(new_arr, weights, output[start])


@nb.njit(parallel=True, fastmath=True)
def correlate1d_y_r(rearr, weights, weight_size, width, height, output):
    for start in nb.prange(width):
        end = start+weight_size
        np.dot(weights, rearr[start:end], output[start])
