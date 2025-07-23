from fractions import Fraction
import math

def get_factors(num1, num2):
    arr = []
    for i in range(2, 9):
        print(i, num1 % i + num2 % i)

    return arr

def generate_fps(received_fps, desired_fps):
    """
    given the number of frames per second that are received, & the number of frames per second that are desired,
    returns list representing which frames should be taken to achieve desired fps while distributing the taken
    frames as evenly as possible through each second

    i.e.

    received_fps -> 10
    desired_fps -> 3

    frames taken per second -
    [0,1,2*,3,4,5*,6,7,8*,9]

    ret [2,2,2,1]
    :return: array
    """
    f = received_fps / desired_fps

    low = math.floor(f)
    high = math.ceil(f)

    arr = []
    for num_of_highs in range(1, received_fps // high + 1):
        if (received_fps - num_of_highs*high) % low == 0:
            num_of_lows = (received_fps - num_of_highs*high) // low
            for z in range(0, num_of_highs):
                arr.append(high)

            for z in range(0, num_of_lows):
                arr.append(low)
    """
    print(n_low, n_high)
    #lowest_common_factor = [i for i in math.lcm(n_low, n_high)]
    print(int(n_high / n_low))
    #f1 = get_factors(n_low, n_high)

    #print(f1)
    #print(n_low, n_high, )
    #print((n_low + n_high) % n_low)


    for i in range(n_low):
        arr.append(low)
    for i in range(n_high):
        arr.append(high)
    """
    #arr = []
    #print('sum', sum(arr), sum(arr)/len(arr))
    return arr

print(generate_fps(13, 3))