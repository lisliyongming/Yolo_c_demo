/*
 * convolutional.c
 *
 *  Created on: 2018��3��13��
 *      Author: ������
 */
#include "./include/convolutional.h"
#include<stdio.h>

convolutional_layer cfg_convolutional_layer(network *net,int output_channel,int stride,int size,int pad,int bn,int actfunc_flag){
	convolutional_layer l = {0};
	l.n = output_channel;
	l.c = net->current_c;
	l.w = net->current_w;
	l.h = net->current_h;
	l.stride = stride;
	l.size = size;
	l.pad = pad;
	l.out_h = (net->current_h + 2*l.pad - l.size)/l.stride + 1;
	l.out_w = (net->current_w + 2*l.pad - l.size)/l.stride + 1;
	l.outputs = l.n*l.out_w*l.out_h;
	l.weights = (float*)calloc(l.n*l.c*l.size*l.size,sizeof(float));
	l.output = (float*)calloc(l.n*l.out_w*l.out_h,sizeof(float));
	l.batch_normalize = bn;
	//net->workspace = (float *)calloc(l.c*l.size*l.size*l.out_w*l.out_h,sizeof(float));
	if(net->workspace == NULL)
	{
		printf("calloc failed \n");
		exit(0);
	}
	if(l.batch_normalize){
		l.rolling_mean = (float*)calloc(l.n,sizeof(float));
		l.rolling_variance = (float*)calloc(l.n,sizeof(float));
		l.scales = (float*)calloc(l.n,sizeof(float));
	}
	l.biases = (float*)calloc(l.n,sizeof(float));
	l.forward = forward_conv_layer;
	l.actfunc_flag = actfunc_flag;
	net->current_c = l.n;
	net->current_h = l.out_h;
	net->current_w = l.out_w;
	return l;
}

void forward_conv_layer(convolutional_layer l, network *net) {
	int out_h = l.out_h;
	int out_w = l.out_w;
	fill_cpu(l.outputs * 1, 0, l.output, 1);
	int m = l.n;                // 该层卷积核个数
	int k = l.size * l.size * l.c;  // 该层每个卷积核的参数元素个数
	int n = out_h * out_w;        // 该层每个特征图的尺寸（元素个数）

	float *a = l.weights; // 所有卷积核（也即权重），元素个数为l.n*l.c*l.size*l.size，按行存储，共有l*n行，l.c*l.size*l.size列
	net->workspace = (float *)realloc(net->workspace,l.c*l.size*l.size*l.out_w*l.out_h*sizeof(float));
	float *b = net->workspace;   // 对输入图像进行重排之后的图像数据
	float *c = l.output; // 存储一张输入图片（多通道）所有的输出特征图（输入图片是多通道的，输出图片也是多通道的，有多少个卷积核就有多少个通道，每个卷积核得到一张特征图即为一个通道）
	im2col_cpu(net->workspace_input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
	gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
	if (l.batch_normalize) {
		forward_batchnorm_layer(l);
	} else {
		add_bias(l.output, l.biases, 1, l.n, out_h * out_w);
	}
	if(l.actfunc_flag == 1){
		activate_array(l.output, m * n * 1);
	}else
	{
	}

}

 void fill_cpu(int N, float ALPHA, float *X, int INCX) {
	int i;
	for (i = 0; i < N; ++i)
		X[i * INCX] = ALPHA;
}

void im2col_cpu(float* data_im, int channels, int height, int width, int ksize,
		int stride, int pad, float* data_col) {
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width,
						channels, im_row, im_col, c_im, pad);
				if(col_index > 1000){
					//printf("data_col[%d]=%f",col_index,data_col[col_index]);
				}
			}
		}
	}
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda,
		float *B, int ldb, float BETA, float *C, int ldc) {
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A,
		int lda, float *B, int ldb, float BETA, float *C, int ldc) {
	int i, j;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * ldc + j] *= BETA;
		}
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);//fixme:去掉TA和TB
}

void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
		int ldb, float *C, int ldc) {
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[i * lda + k];	//m是输出的通道数，k是卷积核乘上输入通道，n是输出的特征图的尺寸
			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];//lda是k的个数，ldb是n的个数，ldc是n的个数
				//printf("C[%d][%d]=%f,",k,i * ldc + j,C[i * ldc + j]);
			}
		}
	}
}

void forward_batchnorm_layer(convolutional_layer l) {
	normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, 1,
				l.n, l.out_h * l.out_w);
	scale_bias(l.output, l.scales, 1, l.n, l.out_h * l.out_w);
	add_bias(l.output, l.biases, 1, l.n, l.out_h * l.out_w);
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY) {
	int i;
	for (i = 0; i < N; ++i)
		Y[i * INCY] = X[i * INCX];
}

void normalize_cpu(float *x, float *mean, float *variance, int batch,
		int filters, int spatial) {
	int b, f, i;
	for (b = 0; b < batch; ++b) {
		for (f = 0; f < filters; ++f) {
			for (i = 0; i < spatial; ++i) {
				int index = b * filters * spatial + f * spatial + i;
				x[index] = (x[index] - mean[f])
						/ (sqrt(variance[f]) + .000001f);
					//printf("x[%d][%d][%d]=%f,",b,f,index,x[index]);
			}
		}
	}
}

void scale_bias(float *output, float *scales, int batch, int n, int size) {
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b * n + i) * size + j] *= scales[i];
			}
		}
	}
}

void add_bias(float *output, float *biases, int batch, int n, int size) {
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b * n + i) * size + j] += biases[i];
			}
		}
	}
}

void activate_array(float *x, const int n) {
	int i;
	for (i = 0; i < n; ++i) {
		x[i] = relu_activate(x[i]);
	}
}

 inline float relu_activate(float x)
{
	return x*(x>0);
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

