#include "rnn_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//YB: 这个函数的作用就是rnn向前的过程中，调整参数的初始位置

/*
** 每前向传播一步后，都要调整下一时刻输出向量以及错误的起始位置
** 输入     l       指向当前RNN层的指针
**          steps   移动的步数
*/
static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}

/*
** 构建RNN层， RNN层本质上是三个全连接层构成的
** darknet中的RNN是vanilla RNN，具体结构可以参考 https://pjreddie.com/darknet/rnns-in-darknet/
**
** 输入：  batch               输入中字符的个数，这个不是最终的batch值，因为RNN需要设置step值，即batch=batch/step
          inputs              RNN层每个输入字符的元素个数
          hidden              RNN隐含层的元素个数
          outputs             RNN层输出元素个数
          steps                RNN网络的步长
          activation          激活方式
          batch_normalize     是否需要BN操作
          log                 通过该值确定激活函数，如果值为1，则为Logistic， 如果值为2， 则为LOGGY， 否则为参数中确定的激活函数
** 返回：  RNN层l
 */
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps; // 上面已经说明，RNN需要设置步长，所以要将batch分成step份
    layer l = {0};
    l.batch = batch;
    l.type = RNN;
    l.steps = steps;
    l.inputs = inputs;

    l.state = calloc(batch*outputs, sizeof(float));
    l.prev_state = calloc(batch*outputs, sizeof(float));

    /*
    ** vanilla RNN由三个隐含层组成
    ** input_layer        隐含层中的第一层，与输入层相连
    ** self_layer         隐含层中的第二层
    ** output_layer       隐含层中的第三层，与输出层相连
    ** 这三个隐含层每一层都是一个全连接层，直接调用make_connected_layer创建
    */

    l.input_layer = malloc(sizeof(layer)); //YB: 隐藏层1
    fprintf(stderr, "\t\t");
    *(l.input_layer) = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
    l.input_layer->batch = batch;

    l.self_layer = malloc(sizeof(layer)); //YB: 隐藏层2
    fprintf(stderr, "\t\t");
    *(l.self_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.self_layer->batch = batch;

    l.output_layer = malloc(sizeof(layer)); //YB: 隐藏层3
    fprintf(stderr, "\t\t");
    *(l.output_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.output_layer->batch = batch;

    l.outputs = outputs;
    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;


    /*
    ** RNN层的前向传播，后向传播以及更新方法
    ** 这些方法的基础是全连接层定义的前向传播，后向传播和更新方法
    */

    l.forward = forward_rnn_layer;
    l.backward = backward_rnn_layer;
    l.update = update_rnn_layer;
#ifdef GPU
    l.forward_gpu = forward_rnn_layer_gpu;
    l.backward_gpu = backward_rnn_layer_gpu;
    l.update_gpu = update_rnn_layer_gpu;
    l.state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.output_gpu = l.output_layer->output_gpu;
    l.delta_gpu = l.output_layer->delta_gpu;
#ifdef CUDNN
    cudnnSetTensor4dDescriptor(l.input_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.input_layer->out_c, l.input_layer->out_h, l.input_layer->out_w); 
    cudnnSetTensor4dDescriptor(l.self_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.self_layer->out_c, l.self_layer->out_h, l.self_layer->out_w); 
    cudnnSetTensor4dDescriptor(l.output_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.output_layer->out_c, l.output_layer->out_h, l.output_layer->out_w); 
#endif
#endif

    return l;
}

//YB: 更新参数信息，调用三次全连接层的更新函数
/*
** 更新权值，因为每一层都是全连接层，所以直接嗲用update_connected_layer即可
** 输入     l                       当前RNN层
**          batch                   该层中一个batch所包含的字符数
**          learning_rate           学习率
**          momentum                动量因子
**          decay                   权值衰减系数
*/

void update_rnn_layer(layer l, update_args a)
{
    update_connected_layer(*(l.input_layer),  a);
    update_connected_layer(*(l.self_layer),   a);
    update_connected_layer(*(l.output_layer), a);
}

/*
** RNN层前向传播
** 输入    l      当前的RNN层
**        net     当前网络
**
** RNN层前向传播与其他网络不同，RNN中的全连接层的当前状态与上一个时间的状态有关，所以要在每次传播后记录上一个时刻的状态
*/

void forward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    /* 开始训练前要将三个全连接层的错误值设置为0 */

    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_layer.delta, 1);
	    /* 如果网络处于训练状态，要将state设置为0，因为初始状态的上一时刻状态是不存在的，我们只能假设它存在，并把它赋值为0 */
    if(net.train) fill_cpu(l.outputs * l.batch, 0, l.state, 1); //YB：如果网络是训练状态的话，将state设置为0

    /*
    ** 以下是RNN层前向传播的主要过程，该层总共要传播steps次，每次的输入batch个字符。
    ** Vanilla RNN具体结构参考 https://pjreddie.com/darknet/rnns-in-darknet/，这里我只简单的说明一下。
    ** Vanilla RNN的RNN层虽然包含三个全连接层，但是只有中间一层（也就是self_layer)与传统的RNN的隐含层一致。
    **
    ** 第一层input_layer可以理解为embedding层，它将input编码为一个hidden维的向量，以darknet的字符预测问题为例，网络的input是英文字母
    ** 采用one-hot编码，是一个256维的向量，embedding后成为了hidden维的向量。
    **
    ** 第二层self_layer与普通RNN的隐含层功能相同，它接收输入层和上一时刻的状态作为输入。
    **
    ** 第三层output_layer，接收self_layer的输出为输入，需要注意的是这一层的输出并不是最终结果，还需要做进一步处理，还是以darknet
    ** 的字符预测为例，第三层的输出要进一步转化为一个256维的向量，然后进行归一化，找到概率最大的字符作为预测结果
    */

    for (i = 0; i < l.steps; ++i) {
        s.input = net.input;
        forward_connected_layer(input_layer, s);

        s.input = l.state; 
        forward_connected_layer(self_layer, s);

        float *old_state = l.state; //YB：存储当前状态 // 将当前状态存入上一时刻状态
        if(net.train) l.state += l.outputs*l.batch; // 如果网络处于训练状态，注意的是上一时刻的状态包含一个batch
        if(l.shortcut){ //YB：是否添加shortcut连接 // 如何设置当前状态，由shortcut的值决定
            copy_cpu(l.outputs * l.batch, old_state, 1, l.state, 1);
        }else{
            fill_cpu(l.outputs * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.outputs * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_connected_layer(output_layer, s);
		
		//YB: 结束后，三个层都向前移动一步
		/* 一次传播结束，将三个层同时向前推移一步 */
        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

/*
** 误差后向传播
** 输入     l       当前RNN层
**          net     当前网络
*/

void backward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    /* 误差传播从网络中最后一步开始 */

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.outputs*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i) {
        copy_cpu(l.outputs * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        /* 计算output_layer层的误差 */

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_connected_layer(output_layer, s);

        l.state -= l.outputs*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.outputs * l.batch, input_layer.output - l.outputs*l.batch, 1, l.state, 1);
           axpy_cpu(l.outputs * l.batch, 1, self_layer.output - l.outputs*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.outputs * l.batch, 0, l.state, 1);
           }
         */

        /* 计算self_layer层的误差 */

        s.input = l.state;
        s.delta = self_layer.delta - l.outputs*l.batch;
        if (i == 0) s.delta = 0;
        backward_connected_layer(self_layer, s);

        copy_cpu(l.outputs*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.outputs*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.outputs*l.batch, 1);
        s.input = net.input + i*l.inputs*l.batch;
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
		        /* 计算input_layer层的误差 */
        backward_connected_layer(input_layer, s);
		
        //YB: 误差传回后，后退一步
                /* 误差传播一步之后，需要重新调整各个连接层， 向后移动一步 */
        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

#ifdef GPU

void pull_rnn_layer(layer l)
{
    pull_connected_layer(*(l.input_layer));
    pull_connected_layer(*(l.self_layer));
    pull_connected_layer(*(l.output_layer));
}

void push_rnn_layer(layer l)
{
    push_connected_layer(*(l.input_layer));
    push_connected_layer(*(l.self_layer));
    push_connected_layer(*(l.output_layer));
}

void update_rnn_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.input_layer),  a);
    update_connected_layer_gpu(*(l.self_layer),   a);
    update_connected_layer_gpu(*(l.output_layer), a);
}

void forward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);

    if(net.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(input_layer, s);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(self_layer, s);

        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(output_layer, s);

        net.input_gpu += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    float *last_input = input_layer.output_gpu;
    float *last_self = self_layer.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = self_layer.delta_gpu;
        backward_connected_layer_gpu(output_layer, s);

        if(i != 0) {
            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        }else {
            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        }

        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
        if (i == 0) s.delta_gpu = 0;
        backward_connected_layer_gpu(self_layer, s);

        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
        else s.delta_gpu = 0;
        backward_connected_layer_gpu(input_layer, s);

        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
}
#endif
