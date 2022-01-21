#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

/* 获得定义的枚举类型的激活函数类别 */
ACTIVATION get_activation(char *s);

/* 获得激活函数对应的字符串描述 **/
char *get_activation_string(ACTIVATION a);

/* 根据不同的激活函数类型，调用不同的激活函数处理输入元素x */
float activate(float x, ACTIVATION a);

/* 根据不同的激活函数求取对输入的梯度 */
float gradient(float x, ACTIVATION a);

/* 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta（误差项）*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);

/* 用激活函数处理输入x中的每一个元素 */
void activate_array(float *x, const int n, const ACTIVATION a);

#ifdef GPU
void activate_array_gpu(float *x, int n, ACTIVATION a);
void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
#endif

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
/*
** 内联函数可以加快调用的速度，但是调用次数多的话，会使可执行文件变大，这样会降低速度。
** static 修饰的内联函数，一般情况下不会产生函数本身的代码，而是全部被嵌入在被调用的地方。
** 如果不加static，则表示该函数有可能会被其他编译单元所调用，所以一定会产生函数本身的代码。
** gcc的static inline相对于static函数来说只是在调用时建议编译器进行内联展开； 
** gcc不会特意为static inline函数生成独立的汇编码，除非出现了必须生成不可的情况（如通过函数指针调用和递归调用）； 
** gcc的static inline函数仅能作用于文件范围内。
*/
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

// 返回线性激活函数（就是f(x)=x）值
static inline float linear_activate(float x){return x;}

// 返回logistic (sigmoid) 函数值
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}

// 返回ReLU非线性激活函数值
static inline float relu_activate(float x){return x*(x>0);}

// 返回指数线性单元（Exponential Linear Unit, ELU）值
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}

// 返回leaky ReLU非线性激活函数值
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}

// 返回tanh非线性激活函数值
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
// 返回线性激活函数（就是f(x)=x）关于输入x的导数值
static inline float linear_gradient(float x){return 1;}

// 返回logistic (sigmoid) 函数关于输入x的导数值
// ** 说明： 这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式
//  比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}

// 返回ReLU非线性激活函数关于输入x的导数值
static inline float relu_gradient(float x){return (x>0);}

// 返回指数线性单元（Exponential Linear Unit, ELU）非线性激活函数关于输入x的导数值
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}

static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}

// 返回leaky ReLU非线性激活函数关于输入x的导数值
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}

// 返回tanh非线性激活函数关于输入x的导数值
static inline float tanh_gradient(float x){return 1-x*x;}


static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

