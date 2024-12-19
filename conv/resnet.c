#include "resnet.h"
#include "../neural/activations.h"
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

ResNet* resnet_create(int num_blocks, int num_classes) {
    ResNet* net = malloc(sizeof(ResNet));
    
    // Initial convolution layer
    net->initial_conv = conv_layer_create(3, 64, 3);  // 3 input channels (RGB)
    
    // Create ResNet blocks
    ResBlock* current = NULL;
    for (int i = 0; i < num_blocks; i++) {
        ResBlock* block = malloc(sizeof(ResBlock));
        block->conv1 = conv_layer_create(64, 64, 3);
        block->conv2 = conv_layer_create(64, 64, 3);
        block->shortcut = NULL;
        block->next = NULL;
        
        if (net->blocks == NULL) {
            net->blocks = block;
        } else {
            current->next = block;
        }
        current = block;
    }
    
    // Final fully connected layer
    net->fc_weights = matrix_create(64, num_classes);
    net->fc_bias = matrix_create(1, num_classes);
    net->num_classes = num_classes;
    
    return net;
}

Matrix* resnet_forward(ResNet* net, Matrix* input) {
    // Initial convolution
    Matrix* x = conv2d(input, net->initial_conv, 1, 1);
    x = relu(x);
    
    // Process through ResNet blocks
    ResBlock* block = net->blocks;
    while (block != NULL) {
        Matrix* identity = matrix_copy(x);
        
        // First conv layer
        Matrix* out = conv2d(x, block->conv1, 1, 1);
        out = relu(out);
        
        // Second conv layer
        out = conv2d(out, block->conv2, 1, 1);
        
        // Add identity connection
        matrix_add_inplace(out, identity);
        out = relu(out);
        
        matrix_free(x);
        matrix_free(identity);
        x = out;
        block = block->next;
    }
    
    // Global average pooling
    x = avg_pool2d(x, x->rows, 1);
    
    // Fully connected layer
    Matrix* output = dot(x, net->fc_weights);
    matrix_add_inplace(output, net->fc_bias);
    
    matrix_free(x);
    return output;
}