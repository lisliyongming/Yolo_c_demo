#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "detection.h"
#include <math.h>
//#define DEBUG_TEST
#define SIDE_FEATUREMAP_W 20
#define SIDE_FEATUREMAP_H 11
#define NUM_BOUNDINGBOX 5
#define MEMORY_IAMGE_HDMIIN 3686400 //1280*720*4
#define MEMORY_IAMGE_HDMIIN_STRIDE 16777216 //2048*2048*4
#define Y_POSITION 220 // 20*11*1
#define W_POSITION 440 // 20*11*2
#define H_POSITION 660 // 20*11*3
#define PRO_OBJ_POSITION 880 // 20*11*4 
#define PEDSTRIAN_POSITION 1100 // 20*11*5
#define CAR_POSITION 1320 // 20*11*6
#define CYCLIST_POSITION 1540 // 20*11*7
#define ONE_BOUNDINGBOX_RANGE 1760 // 20*11*8
#define PRO_THRESHOLD 0.2
#define OVERLAP_THRESHOLD 0.2
#define ONE_SILE_NUM 88 //11*(4+1+3)
#define ONE_PIECE_NUM 440 //11*(4+1+3)*5
#define MUTIPLE_VALUE1 1760 //SIDE_FEATUREMAP_W*SIDE_FEATUREMAP_H*(4+1+3)
#define MUTIPLE_VALUE2 1100  //5*SIDE_FEATUREMAP_W*SIDE_FEATUREMAP_H
#define MUTIPLE_VALUE3 880  //4*SIDE_FEATUREMAP_W*SIDE_FEATUREMAP_H
#define MUTIPLE_VALUE4 220  //SIDE_FEATUREMAP_W*SIDE_FEATUREMAP_H
#define MUTIPLE_VALUE5 440  //2*SIDE_FEATUREMAP_W*SIDE_FEATUREMAP_H
#define MUTIPLE_VALUE6 660  //3*SIDE_FEATUREMAP*SIDE_FEATUREMAP
#define ANCHOR_COORDINATE_NUM 10
float biases[ANCHOR_COORDINATE_NUM] = {0.738768,0.874946,2.42204,2.65704,4.30971,7.04493,10.246,4.59428,12.6868,11.8741};
static inline float get_max(float a, float b, float c){
	float m = 0;
	float n = 0;
	float max = 0;
	m = a > b ? a : b;
	n = b > c ? b : c;
	max = m > n ? m : n;
	return max;
}

static inline int lap(int x1_min_in,int x1_max_in,int x2_min_in,int x2_max_in){
	if(x1_min_in < x2_min_in){
		if(x1_max_in < x2_min_in){
			return 0;
		}else{
			if(x1_max_in > x2_min_in){
				if(x1_max_in < x2_max_in){
					return x1_max_in - x2_min_in;
				}else{
					return x2_max_in - x2_min_in;
				}
			}else{
				return 0;
			}
		}
	}else{
		if(x1_min_in < x2_max_in){
			if(x1_max_in < x2_max_in)
				return x1_max_in-x1_min_in;
			else
				return x2_max_in-x1_min_in;
		}else{
			return 0;
		}
	}
}
uint32_t detection(int* bboxs, float *result_float) {
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t k = 0;
	uint32_t n = 0;
	uint32_t cla = 0;
	float max = 0.6;
	int class_idx,obj_idx,box_index,max_idx;
	uint32_t position_current_temp1 = 0;
	float pro_obj[1100];//[5*20*11]
	int idx_class[1100];//[5*20*11]
	float sum_exp = 0;
	int x_min,x_max,y_min,y_max;
	float x,y,w,h;
	int overlap_x = 0;
	int overlap_y = 0;
	float overlap = 0;
	uint32_t valid_box_num = 0;
	//float* result_float = new float[8800];
	//get x value from accelerator pattern to darknet pattern
//	for (i = 0; i < NUM_BOUNDINGBOX; ++i){
//		for (j = 0; j < SIDE_FEATUREMAP_H; ++j){
//			for (k = 0; k < SIDE_FEATUREMAP_W; ++k){
//				result_float[ONE_BOUNDINGBOX_RANGE*i + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + Y_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + W_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*2 + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + H_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*3 + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + PRO_OBJ_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*4 + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + PEDSTRIAN_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*5 + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + CAR_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*6 + j];
//				result_float[ONE_BOUNDINGBOX_RANGE*i + CYCLIST_POSITION + j*SIDE_FEATUREMAP_W + k] = result_float_fromacc[ONE_SILE_NUM*i + ONE_PIECE_NUM*k + SIDE_FEATUREMAP_H*7 + j];
//			}
//		}
//	}
	//compute the exp value of x,y,pro and exp (pedestrian car cyclist)
	for(i=0;i<NUM_BOUNDINGBOX;i++){
		for(j=0;j<SIDE_FEATUREMAP_H;j++){
			for(k=0;k<SIDE_FEATUREMAP_W;k++){
				position_current_temp1 = i*ONE_BOUNDINGBOX_RANGE+j*SIDE_FEATUREMAP_W+k;
				result_float[position_current_temp1] = 1.0/(1.0 + exp(-result_float[position_current_temp1]));
				result_float[position_current_temp1+Y_POSITION] = 1.0/(1.0 + exp(-result_float[position_current_temp1+Y_POSITION]));
				result_float[position_current_temp1+PRO_OBJ_POSITION] = 1.0/(1.0 + exp(-result_float[position_current_temp1+PRO_OBJ_POSITION]));

				max = get_max(result_float[position_current_temp1+PEDSTRIAN_POSITION],result_float[position_current_temp1+CAR_POSITION],result_float[position_current_temp1+CYCLIST_POSITION]);
				result_float[position_current_temp1+PEDSTRIAN_POSITION] = exp(result_float[position_current_temp1+PEDSTRIAN_POSITION] - max);
				result_float[position_current_temp1+CAR_POSITION] = exp(result_float[position_current_temp1+CAR_POSITION] - max);
				result_float[position_current_temp1+CYCLIST_POSITION] = exp(result_float[position_current_temp1+CYCLIST_POSITION] - max);
				sum_exp = result_float[position_current_temp1+PEDSTRIAN_POSITION] + result_float[position_current_temp1+CAR_POSITION] + result_float[position_current_temp1+CYCLIST_POSITION];
				result_float[position_current_temp1+PEDSTRIAN_POSITION] = result_float[position_current_temp1+PEDSTRIAN_POSITION]/sum_exp;
				result_float[position_current_temp1+CAR_POSITION] = result_float[position_current_temp1+CAR_POSITION]/sum_exp;
				result_float[position_current_temp1+CYCLIST_POSITION] = result_float[position_current_temp1+CYCLIST_POSITION]/sum_exp;
			}
		}
	}

	//get max value of pro and the ID of max object probability
	for(i=0;i<SIDE_FEATUREMAP_H;i++){
		for(j=0;j<SIDE_FEATUREMAP_W;j++){
			for(n=0;n<NUM_BOUNDINGBOX;n++){
				max = 0;
				max_idx = 0;
				class_idx = n*MUTIPLE_VALUE1 + MUTIPLE_VALUE2 + i*SIDE_FEATUREMAP_W + j;
				obj_idx = n*MUTIPLE_VALUE1 + MUTIPLE_VALUE3 + i*SIDE_FEATUREMAP_W + j;
				for(cla=0;cla<3;cla++){
					k=class_idx + cla*MUTIPLE_VALUE4;
					if (result_float[k] > max){
						max = result_float[k];
						max_idx = cla;
					}
					idx_class[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j] = max_idx;
					pro_obj[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j] = max*result_float[obj_idx];
				}
			}
		}
	}

	//compute the valid coordinate
	for(i=0;i<SIDE_FEATUREMAP_H;i++){
		for(j=0;j<SIDE_FEATUREMAP_W;j++){
			for(n=0;n<NUM_BOUNDINGBOX;n++){
				if (pro_obj[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j] > PRO_THRESHOLD){
#ifdef DEBUG_TEST
					printf("pro=%f\n",pro_obj[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j]);
#endif
					//box_index = n*SIDE_FEATUREMAP*SIDE_FEATUREMAP*(4+1+3) + 0*SIDE_FEATUREMAP*SIDE_FEATUREMAP + i*SIDE_FEATUREMAP+j;
					box_index = n*MUTIPLE_VALUE1 + i*SIDE_FEATUREMAP_W+j;
					//x,y stand for the central coordinate, w,h stand for the width and height of the boundingbox
					//x = (j+result_float[box_index+0*SIDE_FEATUREMAP*SIDE_FEATUREMAP])/SIDE_FEATUREMAP;
					x = (j+result_float[box_index])/SIDE_FEATUREMAP_W;
					y = (i+result_float[box_index+MUTIPLE_VALUE4])/SIDE_FEATUREMAP_H;
					w = exp(result_float[box_index+MUTIPLE_VALUE5])*biases[n*2]/SIDE_FEATUREMAP_W;
					h = exp(result_float[box_index+MUTIPLE_VALUE6])*biases[n*2+1]/SIDE_FEATUREMAP_H;

					//get two corner coordinates of the boundingbox
					x_min = (x - w/2)*1280;
					y_min = (y - h/2)*720;
					x_max = (x + w/2)*1280;
					y_max = (y + h/2)*720;

					//avoid overflow and underflow
					if (x_min<0){
						x_min = 0;
					}
					if (y_min<0){
						y_min = 0;
					}
					if (x_max>= 1280 - 1){
						x_max = 1280 - 4;
					}
					if (y_max>= 720 - 1){
						y_max = 720 - 4;
					}

					//save the computed results in a new two dimensional array or point
					//bboxs[valid_box_num][0] = idx_class[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j];//the identified target
					//bboxs[valid_box_num][1] = x_min;//coordinate x of left top corner
					//bboxs[valid_box_num][2] = y_min;//coordinate y of left top corner
					//bboxs[valid_box_num][3] = x_max;//coordinate x of right bottom corner
					//bboxs[valid_box_num][4] = y_max;//coordinate y of right bottom corner
					//bboxs[valid_box_num][5] = (int)(pro_obj[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j]*100);//the confidence level of object in a boundingbox
					//bboxs[valid_box_num][6] = 0;//For non-maximum suppression mark flag
					bboxs[valid_box_num*7] = idx_class[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j];//the identified target
					bboxs[valid_box_num*7+1] = x_min;//coordinate x of left top corner
					bboxs[valid_box_num*7+2] = y_min;//coordinate y of left top corner
					bboxs[valid_box_num*7+3] = x_max;//coordinate x of right bottom corner
					bboxs[valid_box_num*7+4] = y_max;//coordinate y of right bottom corner
					bboxs[valid_box_num*7+5] = (int)(pro_obj[n*MUTIPLE_VALUE4 + i*SIDE_FEATUREMAP_W+j]*100);//the confidence level of object in a boundingbox
					bboxs[valid_box_num*7+6] = 0;//For non-maximum suppression mark flag
					valid_box_num = valid_box_num + 1;
				}
			}
		}
	}

	//non maximum suppression
	for(i = 0; i < valid_box_num; ++i){
		for(j = i+1; j < valid_box_num; ++j){
			overlap_x = lap(bboxs[i*7+1],bboxs[i*7+3],bboxs[j*7+1],bboxs[j*7+3]);
			overlap_y = lap(bboxs[i*7+2],bboxs[i*7+4],bboxs[j*7+2],bboxs[j*7+4]);
			overlap = (overlap_x*overlap_y)*1.0/((bboxs[i*7+1]-bboxs[i*7+3])*(bboxs[i*7+2]-bboxs[i*7+4])+(bboxs[j*7+1]-bboxs[j*7+3])*(bboxs[j*7+2]-bboxs[j*7+4])-(overlap_x*overlap_y));
			if(overlap > OVERLAP_THRESHOLD){
				if(bboxs[i*7+5] > bboxs[j*7+5]){
					bboxs[i*7+6] = 1;
				}else{
					bboxs[j*7+6] = 1;
				}
			}
		}
	}
	//delete []result_float;
	return valid_box_num;
}
