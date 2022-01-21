#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer dropout_layer;

/** 构建dropout层 **/
dropout_layer make_dropout_layer(int batch, int inputs, float probability);

/** dropout层前向传播函数 **/
void forward_dropout_layer(dropout_layer l, network net);

/** dropout层反向传播函数 **/
void backward_dropout_layer(dropout_layer l, network net);


void resize_dropout_layer(dropout_layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer l, network net);
void backward_dropout_layer_gpu(dropout_layer l, network net);

#endif
#endif
