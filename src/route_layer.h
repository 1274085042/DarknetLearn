#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer route_layer;

/** 构建route层 **/
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size);

/** route层前向传播 **/
void forward_route_layer(const route_layer l, network net);

/** route层后向传播 **/
void backward_route_layer(const route_layer l, network net);

void resize_route_layer(route_layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net);
void backward_route_layer_gpu(const route_layer l, network net);
#endif

#endif
