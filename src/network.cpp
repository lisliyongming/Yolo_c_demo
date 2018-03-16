/*
 * network.c
 *
 *  Created on: 2018��3��13��
 *      Author: ������
 */
#include "./include/network.h"
#include "./include/read_file.h"

network cfg_network(int n) {
	network net;
	net.n = n;
	net.layers = (struct layer*) calloc(net.n, sizeof(layer));
	net.current_c = 3;
	net.current_h = 352;
	net.current_w = 640;
	net.workspace = (float*)calloc(1,sizeof(float));
	FILE *fp = fopen("/home/hans/workspace_eclipse/yolo_framework/src/parameter.bin","rb");
	if(fp == NULL){
		printf("Open file failed!");
		fclose(fp);
		exit(0);
	}
	net.layers[0] = cfg_convolutional_layer(&net, 16, 1, 3, 1, 1, 1); //conv1
	load_convolutional_weights(fp, &net.layers[0]);

	net.layers[1] = cfg_maxpool_layer(&net, 2, 2, 0); //max1

	net.layers[2] = cfg_convolutional_layer(&net, 32, 1, 3, 1, 1, 1); //conv2
	load_convolutional_weights(fp, &net.layers[2]);

	net.layers[3] = cfg_maxpool_layer(&net, 2, 2, 0); //max2

	net.layers[4] = cfg_convolutional_layer(&net, 64, 1, 3, 1, 1, 1); //conv3
	load_convolutional_weights(fp, &net.layers[4]);
	net.layers[5] = cfg_maxpool_layer(&net, 2, 2, 0); //max3

	net.layers[6] = cfg_convolutional_layer(&net, 128, 1, 3, 1, 1, 1); //conv4
	load_convolutional_weights(fp, &net.layers[6]);
	net.layers[7] = cfg_maxpool_layer(&net, 2, 2, 0); //max4

	net.layers[8] = cfg_convolutional_layer(&net, 256, 1, 3, 1, 1, 1); //conv5
	load_convolutional_weights(fp, &net.layers[8]);
	net.layers[9] = cfg_maxpool_layer(&net, 2, 2, 0); //max5

	net.layers[10] = cfg_convolutional_layer(&net, 512, 1, 3, 1, 1, 1); //conv6
	load_convolutional_weights(fp, &net.layers[10]);

	net.layers[11] = cfg_convolutional_layer(&net, 512, 1, 3, 1, 1, 1); //conv7
	load_convolutional_weights(fp, &net.layers[11]);

	net.layers[12] = cfg_convolutional_layer(&net, 512, 1, 3, 1, 1, 1); //conv8
	load_convolutional_weights(fp, &net.layers[12]);

	net.layers[13] = cfg_convolutional_layer(&net, 40, 1, 1, 0, 0, 0); //conv9
	load_convolutional_weights(fp, &net.layers[13]);
	fclose(fp);
	return net;
}

