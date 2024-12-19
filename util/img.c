#include "img.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CIFAR_IMG_SIZE 32
#define CIFAR_CHANNELS 3
#define INPUT_SIZE (CIFAR_IMG_SIZE * CIFAR_IMG_SIZE * CIFAR_CHANNELS)

#define MAXCHAR 10000

Img** csv_to_imgs(char* file_string, int number_of_imgs) {
	FILE *fp;
	Img** imgs = malloc(number_of_imgs * sizeof(Img*));
	char row[MAXCHAR];
	fp = fopen(file_string, "r");

	// Read the first line 
	fgets(row, MAXCHAR, fp);
	int i = 0;
	while (feof(fp) != 1 && i < number_of_imgs) {
		imgs[i] = malloc(sizeof(Img));

		int j = 0;
		fgets(row, MAXCHAR, fp);
		char* token = strtok(row, ",");
		imgs[i]->img_data = matrix_create(28, 28);
		while (token != NULL) {
			if (j == 0) {
				imgs[i]->label = atoi(token);
			} else {
				imgs[i]->img_data->entries[(j-1) / 28][(j-1) % 28] = atoi(token) / 256.0;
			}
			token = strtok(NULL, ",");
			j++;
		}
		i++;
	}
	fclose(fp);
	return imgs;
}

void img_print(Img* img) {
	matrix_print(img->img_data);
	printf("Img Label: %d\n", img->label);
}

void img_free(Img* img) {
	matrix_free(img->img_data);
	free(img);
	img = NULL;
}

void imgs_free(Img** imgs, int n) {
	for (int i = 0; i < n; i++) {
		img_free(imgs[i]);
	}
	free(imgs);
	imgs = NULL;
}

// Helper function to read CIFAR-10 binary format
Img** cifar_to_imgs(const char* filename, int num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    Img** imgs = malloc(num_images * sizeof(Img*));
    
    for (int i = 0; i < num_images; i++) {
        imgs[i] = malloc(sizeof(Img));
        
        // Read label (1 byte)
        unsigned char label;
        size_t label_read = fread(&label, sizeof(unsigned char), 1, file);
        if (label_read != 1) {
            printf("Error: Failed to read label for image %d\n", i);
            fclose(file);
            return NULL;
        }
        
        // Add validation and debug output for label
        if (label > 9) {
            printf("Warning: Unexpected label value %d at image %d\n", (int)label, i);
            printf("First few bytes of image data: ");
            unsigned char peek[4];
            fread(peek, 1, 4, file);
            for(int k = 0; k < 4; k++) {
                printf("%d ", peek[k]);
            }
            printf("\n");
            fseek(file, -4, SEEK_CUR); // Reset file position
        }
        
        imgs[i]->label = (int)label;
        
        // Read image data (32x32x3 bytes)
        unsigned char buffer[INPUT_SIZE];
        size_t data_read = fread(buffer, sizeof(unsigned char), INPUT_SIZE, file);
        if (data_read != INPUT_SIZE) {
            printf("Error: Failed to read image data for image %d (read %zu bytes)\n", i, data_read);
            fclose(file);
            return NULL;
        }
        
        imgs[i]->img_data = matrix_create(INPUT_SIZE, 1);
        
        // Add validation for first few pixel values
        if (i == 0) {
            printf("Debug: First few pixel values: ");
            for(int k = 0; k < 5; k++) {
                printf("%d ", buffer[k]);
            }
            printf("\n");
        }
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            imgs[i]->img_data->entries[j][0] = buffer[j] / 255.0f;
        }
    }
    
    fclose(file);
    return imgs;
}