#pragma once

#include "../matrix/matrix.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    Matrix** kernels;  // Array of kernel matrices
    Matrix* bias;
} ConvLayer;

typedef struct {
    Matrix* input;
    Matrix* output;
    int stride;
    int padding;
} ConvOutput;

// Convolution operations
ConvLayer* conv_layer_create(int in_channels, int out_channels, int kernel_size);
void conv_layer_free(ConvLayer* layer);
Matrix* conv2d(Matrix* input, ConvLayer* layer, int stride, int padding);
Matrix* max_pool2d(Matrix* input, int kernel_size, int stride);
Matrix* avg_pool2d(Matrix* input, int kernel_size, int stride);
