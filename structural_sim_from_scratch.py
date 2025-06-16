import numba as nb
import numpy as np
import math
import scipy.ndimage as ndi

height, width = (1200, 1760)


def generate_weights(ndim, sigma=1.5, truncate=3.5):
    radius = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * radius + 1
    NP = win_size ** ndim
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


# @nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
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


#@nb.guvectorize([(float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])], '(),(),(m,n),(m,n),(m,n),(m,n),(m,n)->(m,n)', nopython=True, target='cuda')
@nb.njit(parallel=True, fastmath=True)
def run_math_(cov_norm, data_range, ux, uy, uxx, uyy, uxy):
    ux_squared = ux * ux
    uy_squared = uy * uy
    ux_uy = ux * uy
    vx = cov_norm * (uxx - ux_squared)
    vxy = cov_norm * (uxy - ux_uy)
    vy = cov_norm * (uyy - uy_squared)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    A1, A2, B1, B2 = (
        2 * ux_uy + C1,
        2 * vxy + C2,
        ux_squared + uy_squared + C1,
        vx + vy + C2,
    )

    return (A1 * A2) / (B1 * B2)


@nb.njit(fastmath=True)
def normalize_diff(diff):
    for x in range(height):
        for y in range(width):
            if diff[x][y] > 1:
                diff[x][y] = 1
            elif diff[x][y] < 0:
                diff[x][y] = 0

            diff[x][y] = 255 * diff[x][y]

    diff = diff.astype("uint8")
    return diff


@nb.njit(parallel=True, fastmath=True)
def correlate1d_x(input, weights, output):
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
def correlate1d_y(input, weights, output):
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