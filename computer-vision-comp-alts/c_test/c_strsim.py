from ctypes import *

so_file = "my_functions.so"
my_functions = CDLL(so_file)
