/*
 * parameters.h
 *
 *  Created on: 2018��3��13��
 *      Author: ������
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include<stdlib.h>

typedef struct network{
	int n;         //网络中层的数量
	int current_c;
	int current_h;
	int current_w;
	float *workspace;	//存贮计算的中间值
	float *workspace_input;
	struct layer *layers;
}network;

typedef struct layer{
	int n;
	int c;
	int h;
	int w;
	int inputs;
	int out_h;
	int out_w;
	int outputs;       //特征图的输出像素点数
	int stride;
	int size;
	int pad;
	int batch_normalize;
	float *output;
	float *x;
	float *weights;
	float *rolling_mean;
	float *rolling_variance;
	float *biases;
	float *scales;
	void (*forward) (struct layer,struct network *);
	int actfunc_flag;
}layer;

#endif /* PARAMETERS_H_ */
