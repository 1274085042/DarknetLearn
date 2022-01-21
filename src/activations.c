#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** 获得激活函数对应的字符串描述 **/
char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

/*  
**  根据输入的激活函数名称，返回定义的枚举类型的激活函数类别
**  输入：*s    C风格字符数组，即激活函数名称，比如relu,logistic等
**  返回：ACTIVATION   激活函数类别；枚举类型
**  说明：该函数通过匹配字符数组，返回激活函数类别；
**       如果输入的激活函数名称未能识别，则使用RELU
*/
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

/* 
** 根据不同的激活函数类型，调用不同的激活函数处理输入元素x
** 输入： x    待处理的元素（单个）
**      a    激活函数类型
*/
float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

/**  
** 用激活函数处理输入x中的每一个元素
** 输入：x    待处理的数组；一般为网络层每个神经元的加权输入Wx+b，在本函数中也是输出
**      n    x中含有多少个元素
**      a    激活函数类型
** 说明：该函数会逐个处理x中的元素；该函数一般用于每一层网络的前向传播函数中；
**      该函数的输出即为每一层网络的输出
*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);   //  根据不同的激活函数类型，调用不同的激活函数处理
    }
}

/*  
** 根据不同的激活函数求取对输入的梯度
** 输入： x    激活函数接收的输入值
**      a    激活函数类型; 见activations.h中枚举类型ACTIVATION的定义
** 输出： 激活函数关于输入x的导数值
*/
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

/*  
** 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta（误差项）
** 输入： x    当前层的所有输出（维度为l.batch * l.out_c * l.out_w * l.out_h）
**       n    l.output的维度，即为l.batch * l.out_c * l.out_w * l.out_h（包含整个batch的）
**       ACTIVATION    激活函数类型
**       delta     当前层误差（与当前输入x维度一样）
** 说明1： 该函数不但计算了激活函数对于加权输入的导数，还将该导数乘以了之前完成计算的误差项delta（对应元素相乘），因此调用该函数之后，将得到该层最终的误差项
** 说明2： 这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式
**       比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
** 说明3： 关于l.delta的初值，比如卷积层中的backward_convolutional_layer()函数，并没有对l.delta赋初值，
**       只是用calloc为其动态分配了内存。但是整个网络会以COST或者REGION为最后一层，这些层中会对l.delta赋初值，
**       又由于l.delta是由后向前逐层传播。因此，当反向执行到某一层时，l.delta的值将都不会为0.
*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        delta[i] *= gradient(x[i], a);
    }
} 
