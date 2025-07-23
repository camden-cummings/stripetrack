import numba as nb
import numpy as np

@nb.njit(parallel=True)
def correlate1d_x(input, np_weights, output):
    rearr = np.concatenate((input[0:5][::-1], input, input[-5:][::-1]))
    rearr = rearr.transpose()

    for start in nb.prange(660): # height
        # could end early by checking that all vals in arr are the same in which case will be the value

        end = start + 11 # size1 (5) + size2 (5) + 1
        np.dot(rearr[:, start:end], np_weights, out=output[start])

@nb.njit(parallel=True)
def correlate1d_y(input, np_weights, output):
    rearr = np.concatenate((input[:, 0:5][:,::-1], input, input[:, -5:][:,::-1]), axis=1)
    rearr = rearr.transpose()

    for start in nb.prange(660): # width
        # print(start)
        end = start + 11 # size1 (5) + size2 (5) + 1
#        np.dot(rearr[:, start:end], np_weights, out=output[start])
        np.dot(rearr[start:end].transpose(), np_weights, out=output[start])
