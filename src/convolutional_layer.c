#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}
/*  计算卷积层输出的特征图高度函数
**  根据输入图像的高度(h)，两边补0的个数(pad)，卷积核尺寸(size)以及步幅(stride)计算输出的特征图的高度
**  输入：l    卷积层，包含该卷积层的所有参数，实际这里没有必要输入整个l，因为只需要到其中的四个参数而已
**  输出：int类型，输出图像的高度
**  说明：此函数只是在最初构建网络时调用一次，之后就不调用了。
**       但l比较大，按值传递复制过程比较冗长，可改善一下。如只输入用到的四个参数，或者传入l的指针，
**      这样不需要返回值，直接在函数内部为l.out_h赋值
*/
int convolutional_out_height(convolutional_layer l)
{
    // pad是每边补0的个数，因此乘以2
    // 当stride=1，pad=size/2（整数除法，会往下取整）时，输出高度就等于输入高度（same策略）；
    // 当stride=1,pad=0时，为valid策略
    // 当stride不等于1时，输出高度恒小于输入高度（尺寸一定会缩小）
    // 计算公式推导：设输出高度为x，总图像高度为h+2*pad个像素，输出高度为x，则共有x-1次卷积核移位，
    // 共占有(x-1)*stride+size个像素，可能还剩余res个像素，且res一定小于stride（否则还可以再移位一次），
    // 因此有(x-1)*stride+size+res=h+2*pad，->x=(h+2*pad-size)/stride+1-res/stride，因为res<stride，
    // 对于整数除法来说，值为0,于是得到最终的输出高度为x=(h+2*pad-size)/stride+1
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

/*  计算卷积层输出的特征图宽度函数
**  根据输入图像的宽度(w)，两边补0的个数(pad)，卷积核尺寸(size)以及步幅(stride)计算输出的特征图的宽度
**  与上一个函数convolutional_out_height()类似，不再赘述
*/
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

/*
**  构建卷积层
**  输入：batch    每个batch含有的图片数
**      h               图片高度（行数）
**      w               图片宽度（列数）
        c               输入图片通道数
        n               卷积核个数
        size            卷积核尺寸
        stride          步幅
        padding         四周补0长度
        activation      激活函数类别
        batch_normalize  是否进行BN 
        binary          是否对权重进行二值化
        xnor            是否对权重以及输入进行二值化
        adam            是否使用adam
*/
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL; // 层属性：卷积层

    l.groups = groups;
    l.h = h; // 输入图像高度
    l.w = w;  // 输入图像宽度
    l.c = c; // 输入图像通道
    l.n = n; // 卷积核个数（即滤波器个数）
    l.binary = binary; // 是否对权重进行二值化
    l.xnor = xnor; // 是否对权重以及输入进行二值化??
    l.batch = batch; // 每个batch含有的图片数
    l.stride = stride; // 步幅
    l.size = size; // 卷积核尺寸
    l.pad = padding; // 四周补0长度
    l.batch_normalize = batch_normalize; // 是否进行BN

	// 该卷积层总的权重元素（卷积核元素）个数=输入图像通道数*卷积核个数*卷积核尺寸
	// 因为一个卷积核要作用在输入图片的所有通道上，所以实际含有的卷积核参数个数需要乘以输入图片的通道数
    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

	// bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

	// 该卷积层总的权重元素个数（权重元素个数等于输入数据的通道数*卷积核个数*卷积核的二维尺寸
	// 注意因为每一个卷积核是同时作用于输入数据 的多个通道上，
	// 因此实际上卷积核是三维的，包括两个维度的平面尺寸，以及输入数据通道数这个维度。
	// 每个通道上的卷积核参数都是独立的训练参数
    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    
	// 初始化权重：使用He初始化方法。缩放因子*标准正态分布随机数，缩放因子等于sqrt(2./(size*size*c)) 
	float scale = sqrt(2./(size*size*c/l.groups));

	for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();

	// 根据该层输入图像的尺寸、卷积核尺寸以及步幅计算输出特征图的宽度和高度
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h; // 输出图像高度
    l.out_w = out_w; // 输出图像宽度
    l.out_c = n; // 输出图像通道（等于卷积核个数，有多少个卷积核，最终就得到多少张特征图，每张图是一个通道）
    l.outputs = l.out_h * l.out_w * l.out_c; // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    l.inputs = l.w * l.h * l.c; // mini-batch中每张输入图片的像素元素个数
	// 关于上面两个参数的说明：
	// 一个mini-batch中有多张图片，每张图片可能有多个通道（彩色图有三通道），l.inputs是每张输入图片所有通道的总元素个数，
	// 而每张输入图片会有n个卷积核对其进行卷积操作，因此一张输入图片会输出n张特征图，这n张特征图的总元素个数就为l.outputs


    // l.output为该层所有的输出（包括mini-batch所有输入图片的输出）
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    // 卷积层三种指针函数，对应三种计算：前向，反向，更新
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b)
    {
        for(i = 0; i < n; ++i)
        {
            for(j = 0; j < size; ++j)
            {
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b)
    { 
        for(i = 0; i < n; ++i)、
        {
            for(j = 0; j < size; ++j)
            {
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

/*
** 计算每个卷积核的偏置更新值，所谓偏置更新值，就是bias = bias - alpha * bias_update中的bias_update
** 输入： bias_updates     当前层所有偏置的更新值，维度为l.n（即当前层卷积核的个数）
**       delta            当前层的误差项（即l.delta）
**       batch            一个batch含有的图片张数（即l.batch）
**       n                当前层卷积核个数（即l.h）
**       k                当前层输入特征图尺寸（即l.out_w*l.out_h）
** 原理：当前层的误差项l.delta是误差函数对加权输入的导数，也就是偏置更新值，只是其中每l.out_w*l.out_h个元素都对应同一个
**      偏置，因此需要将其加起来，得到的和就是误差函数对当前层各偏置的导数（l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
**      可理解成共有l.batch行，每行有l.n*l.out_h*l.out_w列，而这一大行又可以理解成有l.n，l.out_h*l.out_w列，这每一小行就
**      对应同一个卷积核也即同一个偏置）
**      //对应公式(Conv-2)
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
	// 遍历batch中每张输入图片
    // 注意，最后的偏置更新值是所有输入图片的总和（多张图片无非就是重复一张图片的操作，求和即可）。
    // 总之：一个卷积核对应一个偏置更新值，该偏置更新值等于batch中所有输入图片累积的偏置更新值，
    // 而每张图片也需要进行偏置更新值求和（因为每个卷积核在每张图片多个位置做了卷积运算，这都对偏置更新值有贡献）以得到每张图片的总偏置更新值。
    //对应公式(Conv-2)
	for(b = 0; b < batch; ++b)
    {
		// 求和得一张输入图片的总偏置更新值
        for(i = 0; i < n; ++i)
        {   //l.n = n; 卷积核个数（即滤波器个数）
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

/**卷积层前向传播函数**/
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

	// l.outputs = l.out_h * l.out_w * l.out_c在make各网络层函数中赋值（比如make_convolutional_layer()），
    // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    // 初始化输出l.output全为0.0；输入l.outputs*l.batch为输出的总元素个数，其中l.outputs为batch
    // 中一个输入对应的输出的所有元素的个数，l.batch为一个batch输入包含的图片张数；0表示初始化所有输出为0；
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    // 是否进行二值化操作
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups; // 该层卷积核个数
    int k = l.size*l.size*l.c/l.groups; // 该层每个卷积核的参数元素个数
    int n = l.out_w*l.out_h; // 该层每个特征图的尺寸（元素个数）
    // 该循环即为卷积计算核心代码：所有卷积核对batch中每张图片进行卷积运算
    // 可以参考：https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    // 进行辅助理解（主要是辅助理解，实际执行并不一样）。
    // 每次循环处理一张输入图片（所有卷积核对batch中一张图片做卷积运算）
    for(i = 0; i < l.batch; ++i){
		// 将多通道二维图像net.input变成按一定存储规则排列的数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂）
        // 注意net.input包含batch中所有图片的数据，但是每次循环只处理一张（本循环最后一句对net.input进行了移位），因此在im2col_cpu仅会对其中一张图片
        // 进行重排，l.c为每张图片的通道数，l.h为每张图片的高度，l.w为每张图片的宽度，l.size为卷积核尺寸，l.stride为步幅
        // 得到的b为一张图片重排后的结果，也是按行存储的一维数组（共有l.c*l.size*l.size行，l.out_w*l.out_h列），
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
	    // GEneral Matrix to Matrix Multiplication
        // 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作
        // 0,0表示不对输入a,b进行转置，
        // m是输入a,c的行数，具体含义为每个卷积核的个数，
        // n是输入b,c的列数，具体含义为每个输出特征图的元素个数(out_h*out_w)，
        // k是输入a的列数也是b的行数，具体含义为卷积核元素个数乘以输入图像的通道数（l.size*l.size*l.c），
        // a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数，
        // a为所有卷积核集合,元素个数为l.n*l.c*l.size*l.size，按行存储，共有l*n行，l.c*l.size*l.size列，
        // 即a中每行代表一个可以作用在3通道上的卷积核，
        // b为一张输入图像经过im2col_cpu重排后的图像数据（共有l.c*l.size*l.size行，l.out_w*l.out_h列），
        // c为gemm()计算得到的值，包含一张输入图片得到的所有输出特征图（每个卷积核得到一张特征图），c中一行代表一张特征图，
        // 各特征图铺排开成一行后，再将所有特征图并成一大行，存储在c中，因此c可视作有l.n行，l.out_h*l.out_w列。
        // 详细查看该函数注释（比较复杂）
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    // 如需要BN 
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

/*
** 卷积神经网络反向传播函数
** 主要流程：1） 调用gradient_array()计算当前层l所有输出元素关于加权输入的导数值（也即激活函数关于输入的导数值），
**             并乘上上一次调用backward_convolutional_layer()还没计算完的l.delta，得到当前层最终的误差项；
**          2） 如果网络进行了BN，则；
**          3） 如果网络没有进行BN，则直接调用 backward_bias()计算当前层所有卷积核的偏置更新值；
**          4） 依次调用im2col_cpu()，gemm_nt()函数计算当前层权重系数更新值；
**          5） 如果上一层的delta已经动态分配了内存，则依次调用gemm_tn(), col2im_cpu()计算上一层的误差项（并未完成所有计算，还差一个步骤）；
** 强调：每次调用本函数会计算完成当前层的误差项计算，同时计算当前层的偏置、权重更新值，除此之外，还会计算上一层的误差项，但是要注意的是，
**      并没有完全计算完，还差一步：乘上激活函数对加权输入的导数值。这一步在下一次调用本函数时完成。
*/
void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups; // 卷积核个数
    // 每一个卷积核元素个数（包括l.c（l.c为该层网络接受的输入图片的通道数）个通道上的卷积核元素个数总数，比如卷积核尺寸为3*3,
    // 输入图片有3个通道，因为要同时作用于输入的3个通道上，所以实际上这个卷积核是立体的，共有3*3*3=27个元素，这些元素都是要训练的参数）
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h; // 每张输出特征图的元素个数：out_w，out_h是输出特征图的宽高

    // 计算当前层激活函数对加权输入的导数值并乘以l.delta相应元素，从而完成当前层误差项的计算，得到当前层的误差项l.delta。
    // l.output存储了该层网络的所有输出：该层网络接受一个batch的输入图片，其中每张图片经卷积处理后得到的特征图尺寸为：l.out_w,l.out_h，
    // 该层卷积网络共有l.n个卷积核，因此一张输入图片共输出l.n张宽高为l.out_w,l.out_h的特征图（l.outputs为一张图所有输出特征图的总元素个数），
    // 所以所有输入图片也即l.output中的总元素个数为：l.n*l.out_w*l.out_h*l.batch；
    // l.delta是一个一维数组，长度为l.batch * l.outputs（其中l.outputs = l.out_h * l.out_w * l.out_c），在make_convolutional_layer()动态分配内存；
    // 再强调一次：gradient_array()不单单是完成激活函数对输入的求导运算，还完成计算当前层误差项的最后一步：l.delta中每个元素乘以激活函数对输入的导数
    // （注意gradient_arry中使用的是*=运算符）。
    // 每次调用backward_convolutional_laye时，都会完成当前层误差项的计算，同时会计算上一层的误差项，但对于上一层，其误差项并没有完全计算完成，还差一步，
    // 需要等到下一次调用backward_convolutional_layer()时来完成，诚如col2im_cpu()中注释一样。
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        // 计算偏置的更新值： 对应公式(Conv-2)
        // 每个卷积核都有一个偏置，偏置的更新值也即误差函数对偏置的导数，这个导数的计算很简单，实际所有的导数已经求完了，都存储在l.delta中，
        // 接下来只需把l.delta中对应同一个卷积核的项加起来就可以（卷积核在图像上逐行逐列跨步移动做卷积，每个位置处都有一个输出，共有l.out_w*l.out_h个，
        // 这些输出都与同一个偏置关联，因此将l.delta中对应同一个卷积核的项加起来即得误差函数对这个偏置的导数）
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }
    // 遍历batch中的每张图片，对于l.delta来说，每张图片是分开存的，因此其维度会达到：l.batch*l.n*l.out_w*l.out_h，
    // 对于l.weights,l.weight_updates以及上面提到的l.bias,l.bias_updates，是将所有图片对应元素叠加起来
    // （循环的过程就是叠加的过程，注意gemm()这系列函数含有叠加效果，不是覆盖输入C的值，而是叠加到之前的C上），
    // 因此l.weights与l.weight_updates维度为l.n*l.size*l.size，l.bias与l.bias_updates的维度为l.h，都与l.batch无关

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
	    // net.workspace的元素个数为所有层中最大的l.workspace_size（在make_convolutional_layer()计算得到workspace_size的大小，在parse_network_cfg()中动态分配内存，此值对应未使用gpu时的情况）,
        // net.workspace充当一个临时工作空间的作用，存储临时所需要的计算参数，比如每层单张图片重排后的结果（这些参数马上就会参与卷积运算），一旦用完，就会被马上更新（因此该变量的值的更新频率比较大）
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;
			
			// 进入本函数之前，在backward_network()函数中，已经将net.input赋值为prev.output，也即若当前层为第l层，net.input此时已经是第l-1层的输出
            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

			// 下面两步：im2col_cpu()与gemm()是为了计算当前层的权重更新值（其实也就是误差函数对当前层权重的导数）
			// 将多通道二维图像net.input变成按一定存储规则排列的数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂），
			// im2col_cput每次仅处理net.input（包含整个batch）中的一张输入图片（对于第一层，则就是读入的图片，对于之后的层，这些图片都是上一层的输出，通道数等于上一层卷积核个数）。
			// 最终重排的b为l.c * l.size * l.size行，l.out_h * l.out_w列。
			// 你会发现在前向forward_convolutional_layer()函数中，也为每层的输入进行了重排，但是很遗憾的是，并没有一个l.workspace把每一层的重排结果保存下来，而是统一存储到net.workspace中，
			// 并被不断擦除更新，那为什么不保存呢？保存下来不是省掉一大笔额外重复计算开销？原因有两个：1）net.workspace中只存储了一张输入图片的重排结果，所以重排下张图片时，马上就会被擦除，
			// 当然你可能会想，那为什么不用一个l.worspaces将每层所有输入图片的结果保存呢？这引出第二个原因；2）计算成本是降低了，但存储空间需求急剧增加，想想每一层都有l.batch张图，且每张都是多通道的，
			// 重排后其元素个数还会增多，这个存储量太大。如果一个batch有128张图，输入图片尺寸为400*400，3通道，网络有16层（假设每层输入输出尺寸及通道数都一样），那么单单为了存储这些重排结果，
			// 就需要128*400*400*3*16*4/1024/1024/1024 = 3.66G，所以为了权衡，只能重复计算。
            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }
			
			// 下面计算当前层的权重更新值，对应公式(Conv-3)
			// 所谓权重更新值就是weight = weight - alpha * weight_update中的weight_update，
			// 权重更新值等于当前层误差项中每个元素乘以相应的像素值，因为一个权重跟当前层多个输出有关联（权值共享，即卷积核在图像中跨步移动做卷积，每个位置卷积得到的值
			// 都与该权值相关），所以对每一个权重更新值来说，需要在l.delta中找出所有与之相关的误差项，乘以相应像素值，再求和，具体实现的方式依靠im2col_cpu()与gemm_nt()完成。
			// （backward_convolutional_layer整个函数的代码非常重要，仅靠文字没有公式与图表辅助说明可能很难说清，所以这部分更为清晰详细的说明，请参考课件PPT！）
			// GEneral Matrix to Matrix Multiplication
			// 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作；
			// 0表示不对输入a进行转置，1表示对输入b进行转置；
			// m是输入a,c的行数，具体含义为卷积核的个数(l.n)；
			// n是输入b,c的列数，具体含义为每个卷积核元素个数乘以输入图像的通道数(l.size*l.size*l.c)；
			// k是输入a的列数也是b的行数，具体含义为每个输出特征图的元素个数（l.out_w*l.out_h）；
			// a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数；
			// a为l.delta的一大行。l.delta为本层所有输出元素（包含整个batch中每张图片的所有输出特征图）关于加权输入的导数（即激活函数的导数值）集合,
			// 元素个数为l.batch * l.out_h * l.out_w * l.out_c（l.out_c = l.n），按行存储，共有l.batch行，l.out_c * l.out_h * l.out_w列，
			// 即l.delta中每行包含一张图的所有输出图，故这么一大行，又可以视作有l.out_c（l.out_c=l.n）小行，l.out_h*l*out_w小列，而一次循环就是处理l.delta的一大行，
			// 故可以将a视作l.out_c行，l.out_h*l*out_w列的矩阵；
			// b为单张输入图像经过im2col_cpu重排后的图像数据；
			// c为输出，按行存储，可视作有l.n行，l.c*l.size*l.size列（l.c是输入图像的通道数，l.n是卷积核个数），
			// 即c就是所谓的误差项（输出关于加权输入的导数），或者误差项（推荐：https://www.zybuluo.com/hanbingtao/note/485480）（一个核有l.c*l.size*l.size个权重，共有l.n个核）。
			// 由上可知：
			// a: (l.out_c) * (l.out_h*l*out_w)
			// b: (l.c * l.size * l.size) * (l.out_h * l.out_w)
			// c: (l.n) * (l.c*l.size*l.size)（注意：l.n = l.out_c）
			// 故要进行a * b + c计算，必须对b进行转置（否则行列不匹配），因故调用gemm_nt()函数
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

			// 接下来，用当前层的误差项l.delta以及权重l.weights（还未更新）来获取上一层网络的误差项，BP算法的主要流程就是依靠这种层与层之间误差项反向递推传播关系来实现。
			// 在network.c的backward_network()中，会从最后一层网络往前遍循环历至第一层，而每次开始遍历某一层网络之前，都会更新net.input为这一层网络前一层的输出，即prev.output,
			// 同时更新net.delta为prev.delta，因此，这里的net.delta是当前层前一层的误差项。
			// 再强调一次：下面得到的上一层的误差项并不完整，完整的误差项是损失函数对上一层的加权输入的导数，
			// 而这里得到的误差项是损失函数对上一层输出值的导数，还差乘以一个输出值也即激活函数对加权输入的导数。
            if (net.delta) {
				// 当前层还未更新的权重
                a = l.weights + j*l.nweights/l.groups;
				// 每次循环仅处理一张输入图，注意移位（l.delta的维度为l.batch * l.out_c * l.out_w * l.out_h）（注意l.n = l.out_c，另外提一下，对整个网络来说，每一层的l.batch其实都是一样的）
                    b = l.delta + (i*l.groups + j)*m*k;
			    // net.workspace和上面一样，还是一张输入图片的重排，不同的是，此处我们只需要这个容器，而里面存储的值我们并不需要，在后面的处理过程中，
                // 会将其中存储的值一一覆盖掉（尺寸维持不变，还是(l.c * l.size * l.size) * (l.out_h * l.out_w）
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }
				// 相比上一个gemm，此处的a对应上一个的c,b对应上一个的a，c对应上一个的b，即此处a,b,c的行列分别为：
				// a: (l.n) * (l.c*l.size*l.size)，表示当前层所有权重系数
				// b: (l.out_c) * (l.out_h*l*out_w)（注意：l.n = l.out_c），表示当前层的误差项
				// c: (l.c * l.size * l.size) * (l.out_h * l.out_w)，表示上一层的误差项（其元素个数等于上一层网络单张输入图片的所有输出元素个数），
				// 此时要完成a * b + c计算，必须对a进行转置（否则行列不匹配），因故调用gemm_tn()函数。
				// 此操作含义是用：用当前层还未更新的权重值对误差项做卷积，得到包含上一层所有误差项信息的矩阵，但这不是上一层最终的误差项，
				// 因为此时的c，也即net.workspace的尺寸为(l.c * l.size * l.size) * (l.out_h * l.out_w)，明显不是上一层的输出尺寸l.c*l.w*l.h，
				// 接下来还需要调用col2im_cpu()函数将其恢复至l.c*l.w*l.h（可视为l.c行，l.w*l.h列），这才是上一层的误差项（实际还差一个环节，
				// 这个环节需要等到下一次调用backward_convolutional_layer()才完成：将net.delta中每个元素乘以激活函数对加权输入的导数值）。
				// 完成gemm这一步，如col2im_cpu()中注释，是考虑了多个卷积核导致的一对多关系（上一层的一个输出元素会流入到下一层多个输出元素中），
				// 接下来调用col2im_cpu()则是考虑卷积核重叠（步长较小）导致的一对多关系。
                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

				// 对c也即net.workspace进行重排，得到的结果存储在net.delta中，每次循环只会处理一张输入图片，因此，此处只会得到一张输入图产生的敏感图（注意net.delta的移位）,
				// 整个循环结束后，net.delta的总尺寸为l.batch * l.h * l.w * l.c，这就是上一层网络整个batch的误差项，可视为有l.batch行，l.h*l.w*l.c列，
				// 每行存储了一张输入图片所有输出特征图的误差项
				// col2im_cpu()函数中会调用col2im_add_pixel()函数，该函数中使用了+=运算符，也即该函数要求输入的net.delta的初始值为0,而在gradient_array()中注释到l.delta的元素是不为0（也不能为0）的，
				// 看上去是矛盾的，实则不然，gradient_array()使用的l.delta是当前层的误差项，而在col2im_cpu()使用的net.delta是上一层的误差项，正如gradient_array()中所注释的，
				// 当前层l.delta之所以不为0,是因为从后面层反向传播过来的，对于上一层，显然还没有反向传播到那，因此net.delta的初始值都是为0的（注意，每一层在构建时，就为其delta动态分配了内存，
				// 且在前向传播时，为每一层的delta都赋值为0,可以参考network.c中forward_network()函数）
                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

/** 卷积层更新函数     **/
void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
	//更新偏置,对应公式(Conv-4),这里学习率除以batch,应该等效于batch个梯度的平均值
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
	//计算下次梯度需要的偏置的动量,对应公式（update-2）
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
	//计算权重衰减,对应公式（update-4）
    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
	//更新权重,对应公式(Conv-5)
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
	//计算下次梯度需要的权重的动量,对应公式（update-2）
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

