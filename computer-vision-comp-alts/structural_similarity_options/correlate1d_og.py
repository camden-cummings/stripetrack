import math
import numba as nb
import numpy as np

@nb.njit(parallel=True, fastmath=True)
def correlate1d(input, weights, output=None, axis=0):
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
        rearr = np.concatenate((input[0:size1][::-1], input, input[-size2:][::-1]))
        for jj in nb.prange(width):
            #np_row = input[:, jj]

            #size1_arr = np_row[0:size1][::-1]
            #size2_arr = np_row[-size2:][::-1]
            #new_arr = np.concatenate((size1_arr, np_row, size2_arr))

            new_arr = rearr[:, jj]
            if symmetric > 0:
                for start in range(height):
                    n = start+size1

#                    test = new_arr[n-size1:n+size1+1]

                    #print(test, len(test[test == test[0]]))
                    #if np.all(test == test[0]):
                    #    output[start][jj] = test[0]
                    #else:
                    total_neighbour = weights[size1]*new_arr[n]
                    for x in range(1, size1+1):
                        total_neighbour += (new_arr[n+x] + new_arr[n-x]) * weights[size1+x]
                    output[start][jj] = total_neighbour
    elif axis == 1:
        #rearr = np.concatenate((input[:, 0:size1][::-1], input, input[:, -size2:][::-1]), axis=1) # something in the construction of this is wrong but I can't figure out what
        for ii in nb.prange(height):
            np_row = input[ii]
            size1_arr = np_row[0:size1][::-1]
            size2_arr = np_row[-size2:][::-1]
            new_arr = np.concatenate((size1_arr, np_row, size2_arr))
            #print("-", rearr[ii], new_arr)
            #new_arr = rearr[ii]

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