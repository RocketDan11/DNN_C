#include "conv.h"
#include <stdlib.h>
#include <math.h>

ConvLayer* conv_layer_create(int in_channels, int out_channels, int kernel_size) {
    ConvLayer* layer = malloc(sizeof(ConvLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    
    // Initialize kernels
    layer->kernels = malloc(out_channels * sizeof(Matrix*));
    for (int i = 0; i < out_channels; i++) {
        layer->kernels[i] = matrix_create(kernel_size, kernel_size);
        // Initialize weights using He initialization
        for (int r = 0; r < kernel_size; r++) {
            for (int c = 0; c < kernel_size; c++) {
                double scale = sqrt(2.0 / (in_channels * kernel_size * kernel_size));
                layer->kernels[i]->entries[r][c] = random_normal() * scale;
            }
        }
    }
    
    // Initialize bias
    layer->bias = matrix_create(1, out_channels);
    matrix_fill(layer->bias, 0.0);
    
    return layer;
}

Matrix* conv2d(Matrix* input, ConvLayer* layer, int stride, int padding) {
    int input_height = input->rows;
    int input_width = input->cols;
    int kernel_size = layer->kernel_size;
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;
    
    Matrix* output = matrix_create(output_height, output_width);
    
    // Implement convolution operation
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            double sum = 0.0;
            for (int k = 0; k < layer->out_channels; k++) {
                for (int di = 0; di < kernel_size; di++) {
                    for (int dj = 0; dj < kernel_size; dj++) {
                        int in_i = i * stride + di - padding;
                        int in_j = j * stride + dj - padding;
                        if (in_i >= 0 && in_i < input_height && 
                            in_j >= 0 && in_j < input_width) {
                            sum += input->entries[in_i][in_j] * 
                                  layer->kernels[k]->entries[di][dj];
                        }
                    }
                }
                sum += layer->bias->entries[0][k];
            }
            output->entries[i][j] = sum;
        }
    }
    
    return output;
}

Matrix* max_pool2d(Matrix* input, int kernel_size, int stride) {
    int input_height = input->rows;
    int input_width = input->cols;
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    
    Matrix* output = matrix_create(output_height, output_width);
    
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            double max_val = -INFINITY;
            for (int di = 0; di < kernel_size; di++) {
                for (int dj = 0; dj < kernel_size; dj++) {
                    int in_i = i * stride + di;
                    int in_j = j * stride + dj;
                    if (input->entries[in_i][in_j] > max_val) {
                        max_val = input->entries[in_i][in_j];
                    }
                }
            }
            output->entries[i][j] = max_val;
        }
    }
    
    return output;
}

Matrix* avg_pool2d(Matrix* input, int kernel_size, int stride) {
    int input_height = input->rows;
    int input_width = input->cols;
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    
    Matrix* output = matrix_create(output_height, output_width);
    
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            double sum = 0.0;
            for (int di = 0; di < kernel_size; di++) {
                for (int dj = 0; dj < kernel_size; dj++) {
                    int in_i = i * stride + di;
                    int in_j = j * stride + dj;
                    sum += input->entries[in_i][in_j];
                }
            }
            output->entries[i][j] = sum / (kernel_size * kernel_size);
        }
    }
    
    return output;
}

