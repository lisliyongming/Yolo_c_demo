/*
 * convolutional.h
 *
 *  Created on: 2018��3��13��
 *      Author: ������
 */

#ifndef CONVOLUTIONAL_H_
#define CONVOLUTIONAL_H_

#include "parameters.h"
#include "math.h"
#include "read_file.h"

typedef layer convolutional_layer;

layer cfg_convolutional_layer(network *,int,int,int,int,int,int);

void forward_conv_layer (convolutional_layer,network*);

void fill_cpu(int, float, float*, int);

void activate_array(float *x, const int n);

void add_bias(float *output, float *biases, int batch, int n, int size);

void scale_bias(float *output, float *scales, int batch, int n, int size);

void normalize_cpu(float *x, float *mean, float *variance, int batch,
		int filters, int spatial);

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);

void forward_batchnorm_layer(convolutional_layer l);

void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
		int ldb, float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A,
		int lda, float *B, int ldb, float BETA, float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda,
		float *B, int ldb, float BETA, float *C, int ldc);

void im2col_cpu(float* data_im, int channels, int height, int width, int ksize,
		int stride, int pad, float* data_col);

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad);
inline float relu_activate(float x);
#endif /* CONVOLUTIONAL_H_ */
