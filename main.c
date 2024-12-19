#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"
#include "conv/resnet.h"
#include "conv/conv.h"


#define NUM_EPOCHS 10
#define BATCH_SIZE 1000
#define TRAIN_IMGS 10000
#define TEST_IMGS 1000
#define LEARNING_RATE 0.01


int main() {
	srand(time(NULL));

	// // Create ResNet with 5 blocks
    // ResNet* net = resnet_create(5, 10);  // 10 classes for CIFAR-10
    
    // // Load and preprocess CIFAR-10 images
    // int number_imgs = TRAIN_IMGS;
    // Img** imgs = cifar_to_imgs("./data/cifar-10-batches-bin/data_batch_1.bin", number_imgs);
    
    // // Training loop
    // for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    //     printf("Training epoch %d/%d...\n", epoch + 1, NUM_EPOCHS);
        
    //     for (int i = 0; i < number_imgs; i += BATCH_SIZE) {
    //         int batch_end = i + BATCH_SIZE;
    //         if (batch_end > number_imgs) batch_end = number_imgs;
            
    //         for (int j = i; j < batch_end; j++) {
    //             Matrix* prediction = resnet_forward(net, imgs[j]->img_data);
    //             // Implement backpropagation here
    //             matrix_free(prediction);
    //         }
    //     }
    // }
	/***************************************** */
	//TRAINING
	// int number_imgs = TRAIN_IMGS;
	// printf("Debug: About to load CIFAR images...\n");
	// Img** imgs = cifar_to_imgs("./data/cifar-10-batches-bin/data_batch_1.bin", number_imgs);
	
	// if (imgs == NULL) {
	// 	printf("Error: Failed to load CIFAR images\n");
	// 	return 1;
	// }
	// printf("Debug: Successfully loaded %d images\n", number_imgs);
	
	// printf("Debug: Creating neural network...\n");
	// NeuralNetwork* net = network_create(3072, 512, 10, LEARNING_RATE);
	// if (net == NULL) {
	// 	printf("Error: Failed to create neural network\n");
	// 	imgs_free(imgs, number_imgs);
	// 	return 1;
	// }
	// printf("Debug: Neural network created successfully\n");
	
	// // Train for 10 epochs
	// for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
	// 	printf("Training epoch %d/10...\n", epoch + 1);
	// 	printf("Debug: Starting batch training for epoch %d\n", epoch + 1);
		
	// 	// Add more detailed validation prints
	// 	printf("Debug: Checking first image data...\n");
	// 	if (imgs[0] == NULL) {
	// 		printf("Error: First image is NULL\n");
	// 		network_free(net);
	// 		imgs_free(imgs, number_imgs);
	// 		return 1;
	// 	}
	// 	printf("Debug: First image matrix dimensions: %dx%d\n", 
	// 		   imgs[0]->img_data->rows, 
	// 		   imgs[0]->img_data->cols);
		
	// 	printf("Debug: First image label: %d\n", imgs[0]->label);
		
	// 	printf("Debug: About to start training on first image...\n");
	// 	network_train_batch_imgs(net, imgs, BATCH_SIZE);
	// 	printf("Debug: Completed batch training for epoch %d\n", epoch + 1);
	// }
	
	// network_save(net, "cifar10_net");
	// imgs_free(imgs, number_imgs);
	// network_free(net);
/*****************************************************/



/**************************************************** */
	//PREDICTING
	int number_imgs = TEST_IMGS;
	Img** imgs = cifar_to_imgs("./data/cifar-10-batches-bin/test_batch.bin", number_imgs);
	NeuralNetwork* net = network_load("cifar10_net");
	double score = network_predict_imgs(net, imgs, TEST_IMGS);
	printf("Score: %1.5f\n", score);

	imgs_free(imgs, number_imgs);
	network_free(net);


/************************************************/ 
	return 0;
}