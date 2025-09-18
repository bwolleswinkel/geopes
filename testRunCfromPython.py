"""This is a script to run a C function from Python using ctypes, where we pass a numpy array to the C function."""

import ctypes
import numpy as np

# ====== C FUNCTION ======

"""
#include <stdio.h>
#include <stdlib.h>

// gcc -fPIC -shared -o sumAndAverageArray.so sumAndAverageArray.c

double* sumAndAverageArray(double *arr, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    double average = sum / size;
    double *result = malloc(2 * sizeof(double));
    result[0] = sum;
    result[1] = average;
    return result;
}
"""

# ====== C FUNCTION ======

# ------ C LIBRARY -----

lib = ctypes.CDLL('./sumAndAverageArray.so')
lib.sumAndAverageArray.restype = ctypes.POINTER(ctypes.c_double)
lib.sumAndAverageArray.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)

def np_array_to_c_double_array(np_array):
    np_array = np_array.astype(np.float64, copy=False)
    if not np_array.flags['C_CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def cfunc(a: np.ndarray) -> np.ndarray:
    res = lib.sumAndAverageArray(np_array_to_c_double_array(a), a.size)
    return np.array([res[i] for i in range(2)], dtype=np.float64)

# ------ TEST ------

a = np.array([1, 2.54, 3, 4, 5, 6, 7])

res = cfunc(a)

print(f"Result: {res} ({type(res)})")