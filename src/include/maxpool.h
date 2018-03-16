/*
 * maxpool.h
 *
 *  Created on: 2018年3月13日
 *      Author: 李星宇
 */

#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include "parameters.h"
#include "float.h"
typedef layer maxpool_layer;

layer cfg_maxpool_layer(network* net, int stride, int size, int pad);
void forward_maxpool_layer(const maxpool_layer l, network *net);

#endif /* MAXPOOL_H_ */
