#pragma once
#include "conv.h"

typedef struct ResBlock {
    ConvLayer* conv1;
    ConvLayer* conv2;
    Matrix* shortcut;
    struct ResBlock* next;
} ResBlock;

typedef struct {
    ConvLayer* initial_conv;
    ResBlock* blocks;
    Matrix* fc_weights;
    Matrix* fc_bias;
    int num_classes;
} ResNet;

ResNet* resnet_create(int num_blocks, int num_classes);
void resnet_free(ResNet* net);
Matrix* resnet_forward(ResNet* net, Matrix* input);
void resnet_train(ResNet* net, Matrix* input, int label);
