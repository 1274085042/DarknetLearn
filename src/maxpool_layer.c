#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

/*
** 以image格式获取最大池化层输出l.output：将l.output简单打包以image格式返回（并没有改变什么值）
** 输入： l     最大池化层
** 返回： image数据类型
** 说明：本函数将数据简单的打包了一下，float_to_image()函数中，创建了一个image类型数据out，指定了每张图片的
**      宽、高、通道数为最大池化层输出图的宽、高、通道数，然后将out.data置为l.output
*/
image get_maxpool_image(maxpool_layer l)
{
    // 获取最大池化层输出图片的高度，宽度，通道数

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c; // 对于最大池化层，l.c == l.out_c，也即输入通道数等于输出通道数
    return float_to_image(w,h,c,l.output);
}

/*
** 以image格式获取最大池化层输出敏感度图l.delta，与上面get_maxpool_image差不多
*/
image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

/*
** 构建最大池化层
** 输入： batch     该层输入中一个batch所含有的图片张数，等于net.batch
**       h,w,c     该层输入图片的高度（行），宽度（列）与通道数
**       size      池化核尺寸
**       stride    步幅
**       padding   四周补0长度
** 返回： 最大池化层l
** 说明：最大池化层与卷积层有较多的变量可以类比卷积层参数，比如池化核，池化核尺寸，步幅，补0长度等等
*/
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;  // 网络类型为最大池化层
    l.batch = batch; // 一个batch中含有的图片张数（等于net.batch）
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
	// 由最大池化层的输入图像尺寸以及跨度计算输出图像尺寸 
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c; // 最大池化层输出图像的通道数等于输入图像的通道数 
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
	// calloc()函数有两个参数,分别为元素的数目和每个元素的大小
	// calloc()会将所分配的内存空间中的每一位都初始化为零
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

/** 最大池化层前向传播函数 **/
void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        f or(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max; //池化后输出值
                    l.indexes[out_index] = max_i; //池化后最大值索引
                }
            }
        }
    }
}

/** 最大池化层后向传播函数 **/
void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
		// 下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，
		// 而其他神经元的误差项的值都是0
        net.delta[index] += l.delta[i];
    }
}

