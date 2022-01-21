#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 
** 构建全连接层
** 输入：batch             该层输入中一个batch所含有的图片张数，等于net.batch
**       inputs            全连接层每张输入图片的元素个数
**       outputs           全连接层输出元素个数（由网络配置文件指定，如果未指定，默认值为1,在parse_connected()中赋值）
**       activation        激活函数类型
**       batch_normalize   是否进行BN
** 返回： 全连接层 l
*/
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs; // 全连接层一张输入图片的元素个数
    l.outputs = outputs; // 全连接层对应一张输入图片的输出元素个数
    l.batch=batch; // 一个batch中的图片张数
    l.batch_normalize = batch_normalize; // 是否进行BN
    l.h = 1;  // 全连接层输入图片高为1,宽也为1
    l.w = 1;
    l.c = inputs;  // 全连接层的输入通道数等于单张输入图片的元素个数
    l.out_h = 1;  // 全连接层的输出图高为1,宽也为1
    l.out_w = 1;
    l.out_c = outputs; // 全连接层输出图片的通道数等于一张输入图片对应的输出元素个数

    l.output = calloc(batch*outputs, sizeof(float)); // 全连接层所有输出（包含整个batch的）
    l.delta = calloc(batch*outputs, sizeof(float));  // 全连接层的误差项（包含整个batch的）

	// 由下面forward_connected_layer()函数中调用的gemm()可以看出，l.weight_updates应该理解为outputs行，inputs列
    l.weight_updates = calloc(inputs*outputs, sizeof(float));  // 全连接层权重系数更新值个数等于一张输入图片元素个数与其对应输出元素个数之积
    l.bias_updates = calloc(outputs, sizeof(float)); // 全连接层偏置更新值个数就等于一张输入图片的输出元素个数

	// 由下面forward_connected_layer()函数中调用的gemm()可以看出，l.weight应该理解为outputs行，inputs列
    l.weights = calloc(outputs*inputs, sizeof(float)); // 全连接层权重系数个数等于一张输入图片元素个数与其对应输出元素个数之积
    l.biases = calloc(outputs, sizeof(float)); // 全连接层偏置个数就等于一张输入图片的输出元素个数

	// 全连接层前向、反向、更新函数
    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    // 初始化权重：使用He初始化方法。缩放因子*-1到1之间的均匀分布，缩放因子等于sqrt(2./inputs) 
    // 注意：与卷积层make_convolutional_layer()中初始化值不同，这里是均匀分布，而卷积层中是正态分布。
    // 思考：这里或许应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了（事实上在detector.c的train_detector()函数中，
    // 紧接着parse_network_cfg()函数之后，就添加了if(weightfile)语句判断是否导入权重系数文件，如果导入了权重系数文件，也许这里初始化的值也会覆盖掉，
    // 总之这里的权重初始化的处理方式还是值得思考的，也许更好的方式是应该设置专门的函数进行权重的初始化，同时偏置也是）

    // float scale = 1./sqrt(inputs);
    
    //权重初始化
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    // 初始化所有偏置值为0
    for(i = 0; i < outputs; ++i)
    {
        l.biases[i] = 0;
    }

    if(adam)
    {   //采用Adam时
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize)
    {   //采用BN时
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

/** 全连接层更新函数 **/
void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

	//更新偏置,对应公式(FC－4),这里学习率除以batch,应该等效于batch个梯度的平均值
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);

	//计算下次梯度需要的偏置的动量,对应公式（update-2）
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize)
    {
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }
    
	//计算权重衰减,对应公式（update-4）
    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    
	//更新权重,对应公式(FC-5)
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
	//计算下次梯度需要的权重的动量,对应公式（update-2）
	scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

/* 
** 全连接层前向传播函数
** 输入： l     当前全连接层
**       net   整个网络
** 流程： 全连接层的前向传播相对简单，首先初始化输出l.output全为0, 在进行相关参数赋值之后，直接调用gemm_nt()完成Wx操作，
**       而后根据判断是否需要BN，如果需要，则进行BN操作，完了之后为每一个输出元素添加偏置得到Wx+b，最后使用激活函数处理
**       每一个输出元素，得到f(Wx+b)
*/
void forward_connected_layer(layer l, network net)
{
    // 初始化全连接层的所有输出（包含所有batch）为0值
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	// m：全连接层接收的一个batch的图片张数
    // k：全连接层单张输入图片元素个数
    // n：全连接层对应单张输入图片的输出元素个数
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    // a：全连接层的输入数据，维度为l.batch*l.inputs（包含整个batch的输入），可视作l.batch行，l.inputs列，每行就是一张输入图片
    // b：全连接层的所有权重，维度为l.outputs*l.inputs(见make_connected_layer())
    // c：全连接层的所有输出（包含所有batch），维度为l.batch*l.outputs（包含整个batch的输出）
    float *a = net.input;   //input 每张图片按行存储
    float *b = l.weights;   //weights 每个filter按行存储
    float *c = l.output;

    // 根据维度匹配规则，显然需要对b进行转置，故而调用gemm_nt()函数，最终计算得到的c的维度为l.batch*l.outputs,
    // 全连接层的的输出很好计算，直接矩阵相乘就可以了。
    // 所谓全连接，就是全连接层的输出与输入的每一个元素都有关联（当然是同一张图片内的，
    // 最终得到的c有l.batch行,l.outputs列，每行就是一张输入图片对应的输出）
    // m：a的行，值为l.batch，含义为全连接层接收的一个batch的图片张数
    // n：b'的列数，值为l.outputs，含义为全连接层对应单张输入图片的输出元素个数
    // k：a的列数，值为l.inputs，含义为全连接层单张输入图片元素个数
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if(l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    } 
    else 
    {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
	// 前向传播最后一步：前面得到每一个输出元素的加权输入Wx+b,这一步利用激活函数处理l.output中的每一个输出元素，
    // 最终得到全连接层的输出f(Wx+b)
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

/* 
** 全连接层反向传播函数
** 输入： l    当前全连接层
**       net   整个网络
** 流程：先完成之前未完成的计算：计算当前层的误差项l.delta，而后调用axpy_cpu()函数计算当前全连接层的偏置更新值，
**      然后判断是否进行BN，如果进行，则完成BN操作，再接着计算当前层权重更新值，最后计算上一层的误差项。
**      相比于卷积神经网络，全连接层很多的计算变得更为直接，不需要调用诸如im2col_cpu()或者col2im_cpu()函数对数据重排，
**      直接矩阵相乘就可以搞定。
*/
void backward_connected_layer(layer l, network net)
{
    // gradient_array()函数完成激活函数对加权输入的导数，并乘以之前得到的l.delta，得到当前层最终的l.delta（误差函数对加权输入的导数）
	// 对应于公式(P2)
    /*
    *  l.delta 为损失函数对未经激活的输出求导
    * 这个导数可以是损失函数对BN后的输出求导
    * 也可以是损失函数对加权后的输出求导
    */
	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize)
    {
        /*
        * @brief:得到的l.delta为损失函数对BN后的输出求导
        */
        backward_batchnorm_layer(l, net);
    } 
    else 
    {
        // 计算当前全连接层的偏置梯度值
        // 对应公式(FC-2)
        // 误差函数对偏置的导数实际就等于以上刚求完的敏感度值，因为有多张图片，需要将多张图片的效果叠加
        // 不同于卷积层每个卷积核才有一个偏置参数，全连接层是每个输出元素就对应有一个偏置参数，共有l.outputs个，每次循环将求完一张图片所有输出的偏置梯度值。
        // 最终会把每一张图的偏置更新值叠加，因此，最终l.bias_updates中每一个元素的值是batch中所有图片对应输出元素偏置更新值的叠加。
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }
    
    // 计算当前全连接层的权重更新值,对应公式(FC-3)
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
	// a：当前全连接层误差项，维度为l.batch*l.outputs
    // b：当前全连接层所有输入，维度为l.batch*l.inputs
    // c：当前全连接层权重更新值，维度为l.outputs*l.inputs（权重个数）
    // 由行列匹配规则可知，需要将a转置，故而调用gemm_tn()函数，转置a实际上是想把batch中所有图片的影响叠加。
    // 全连接层的权重更新值的计算也相对简单，简单的矩阵乘法即可完成：
    // 当前全连接层的误差项乘以当前层的输入即可得到当前全连接层的权重更新值，
    // （当前层的误差项是误差函数对于加权输入的导数，所以再乘以对应输入值即可得到权重更新值）
    // m：a'的行，值为l.outputs，含义为每张图片输出的元素个数
    // n：b的列数，值为l.inputs，含义为每张输入图片的元素个数
    // k：a’的列数，值为l.batch，含义为一个batch中含有的图片张数
    // 最终得到的c维度为l.outputs*l.inputs，对应所有权重的更新值
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    // 由当前全连接层计算上一层的误差项（完成绝大部分计算：当前全连接层误差项乘以当前层还未更新的权重）
    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;
	
    // 一定注意此时的c等于net.delta，已经在network.c中的backward_network()函数中赋值为上一层的delta
    // a：当前全连接层误差项，维度为l.batch*l.outputs
    // b：当前层权重（连接当前层与上一层），维度为l.outputs*l.inputs
    // c：上一层误差项（包含整个batch），维度为l.batch*l.inputs
    // 由行列匹配规则可知，不需要转置。由全连接层误差项计算上一层的误差项也很简单，直接利用矩阵相乘，将当前层l.delta与当前层权重相乘就可以了，
    // 只需要注意要不要转置。不同于卷积层，需要对权重或者输入重排。
    // m：a的行，值为l.batch，含义为一个batch中含有的图片张数
    // n：b的列数，值为l.inputs，含义为每张输入图片的元素个数
    // k：a的列数，值为l.outputs，含义为每张图片输出的元素个数
    // 最终得到的c维度为l.bacth*l.inputs（包含所有batch）
    //对应公式(P1)
    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}

void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){ //Adam更新只用在GPU版本
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
