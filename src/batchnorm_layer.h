#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

/** 构造BN层函数    **/
layer make_batchnorm_layer(int batch, int w, int h, int c);

/** BN层前向传播函数*/
void forward_batchnorm_layer(layer l, network net);

/** BN层后向传播函数*/
void backward_batchnorm_layer(layer l, network net);

#ifdef GPU
void forward_batchnorm_layer_gpu(layer l, network net);
void backward_batchnorm_layer_gpu(layer l, network net);
void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);
#endif

#endif
