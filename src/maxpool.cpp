/*
 * maxpool.c
 *
 *  Created on: 2018��3��13��
 *      Author: ������
 */
#include "./include/maxpool.h"
layer cfg_maxpool_layer(network* net, int stride, int size, int pad) {
	maxpool_layer l = { 0 };
	l.n = net->current_c;
	l.c = net->current_c;
	l.h = net->current_h;
	l.w = net->current_w;
	l.pad = pad;
	l.stride = stride;
	l.size = size;
	l.out_h = (l.h + 2 * l.pad) / stride;
	l.out_w = (l.w + 2 * l.pad) / stride;
	l.output = (float*)calloc(l.n * l.out_h * l.out_w, sizeof(float));
	l.forward = forward_maxpool_layer;
	net->current_c = l.n;
	net->current_h = l.out_h;
	net->current_w = l.out_w;
	return l;
}

void forward_maxpool_layer(const maxpool_layer l, network *net) {
	int i, j, k, m, n;
	int w_offset = -l.pad;
	int h_offset = -l.pad;
	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	for (k = 0; k < c; ++k) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				int out_index = j + w * (i + h * k);
				float max = -FLT_MAX;
				//int max_i = -1;
				for (n = 0; n < l.size; ++n) {
					for (m = 0; m < l.size; ++m) {
						int cur_h = h_offset + i * l.stride + n;
						int cur_w = w_offset + j * l.stride + m;
						int index = cur_w + l.w * (cur_h + l.h * k);
						int valid = (cur_h >= 0 && cur_h < l.h && cur_w >= 0
								&& cur_w < l.w);
						float val = (valid != 0) ? net->workspace_input[index] : -FLT_MAX;
						max = (val > max) ? val : max;
					}
				}
				l.output[out_index] = max;
			}
		}
	}
}

