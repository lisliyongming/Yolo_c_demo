/*
 * read_file.c
 *
 *  Created on: 2018��3��14��
 *      Author: ������
 */
#include "./include/read_file.h"
#include <iostream>
void load_convolutional_weights(FILE *fp,layer* l){
	int nweight;
	int callback_num = 0;
	nweight = l->n*l->c*l->size*l->size;
	callback_num = fread(l->weights, sizeof(float), nweight, fp);
	//std::cout << *(l->weights) << std::endl;
	if(callback_num == nweight){
		if(l->batch_normalize == 1){
			fread(l->rolling_mean, sizeof(float),l->n,fp);
			//std::cout << "mean " << *(l->rolling_mean) << std::endl;
			fread(l->rolling_variance, sizeof(float),l->n,fp);
			//std::cout << "variance " <<*(l->rolling_variance) << std::endl;
			fread(l->scales, sizeof(float),l->n,fp);
			//std::cout << "scales " << *(l->scales) << std::endl;
		}
		fread(l->biases, sizeof(float),l->n,fp);
		//std::cout << "biases " << *(l->biases) << std::endl;
	}
	else{
		printf("file size is wrong!\n");
		exit(0);
	}

}
