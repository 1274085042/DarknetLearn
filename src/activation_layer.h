#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

/** 构造激活层函数    **/
layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

/** 激活层前向传播函数*/
void forward_activation_layer(layer l, network net);

/** 激活层后向传播函数 **/
void backward_activation_layer(layer l, network net);

#ifdef GPU
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);
#endif

#endif

