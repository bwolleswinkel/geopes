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