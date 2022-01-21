#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

/*
* 构建yolo层
*/
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1); //coco:3*(5 + 80) = 255
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;

	// calloc()函数有两个参数,分别为元素的数目和每个元素的大小
	// calloc()会将所分配的内存空间中的每一位都初始化为零
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else
    {
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1); // 默认每张图片可检测90个目标; 可改为l.truths = l.max_boxes*(4 + 1); 
    l.delta = calloc(batch*l.outputs, sizeof(float)); //注意：l.delta和l.output分配的内存空间大小相同
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i)
    {
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/*
** 计算yolo矩形框坐标值
*/

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

/*
** 计算yolo矩形框坐标参数的误差项
*/
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

/*
** 计算yolo分类参数的误差项
*/
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index])
    {
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n)
    {
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

/* 计算某个矩形框参数在l.output中的索引 */
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

/*
**  yolo层前向传播函数
*/
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
   // 将net.input中的元素全部拷贝至l.output中
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
	// 遍历l.batch中的每张图片（l.output含有l.batch训练图片对应的输出）
    for (b = 0; b < l.batch; ++b)
    {
		// 注意l.n含义是每个cell grid（网格）中预测的矩形框个数
        for(n = 0; n < l.n; ++n)
        {
			//下面的entry_index()计算某个矩形框中的坐标参数在l.output中的索引
            int index = entry_index(l, b, n*l.w*l.h, 0);

		   //注意第二个参数是2*l.w*l.h，也就是从index+l.output处开始，对之后2*l.w*l.h个元素进行logistic激活函数处理，
           //只对tx,ty作激活(sigma)处理,不对tw,th作激活处理。原因可查看论文中的计算公式
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);

			//下面的entry_index()计算某个矩形框的类别参数在l.output中的索引
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif
	// 误差项清零
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

	// 如果不是训练过程，则返回不再执行下面的语句
	// 前向推理即检测过程也会调用这个函数，这时就不需要执行下面训练时才会用到的语句
    if(!net.train) return;
    float avg_iou = 0; // 平均IoU（Intersection over Union）
    float recall = 0;  // 召回率
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0; // 该batch内检测的目标数
    int class_count = 0;
    *(l.cost) = 0;

    for (b = 0; b < l.batch; ++b) 
    {
        for (j = 0; j < l.h; ++j) 
        {
            for (i = 0; i < l.w; ++i) 
            {
                for (n = 0; n < l.n; ++n) 
                {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);

					// 计算yolo矩形框坐标值
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;

					// 循环l.max_boxes次，每张图片固定处理l.max_boxes个矩形框
                    for(t = 0; t < l.max_boxes; ++t)
                    {
					    // 通过移位来获取每一个真实矩形框的信息，net.truth存储了网络吞入的所有图片的真实矩形框信息（一次吞入一个batch的训练图片），
                        // net.truth作为这一个大数组的首地址，l.truths参数是每一张图片含有的真实值参数个数（可参考layer.h中的truths参数中的注释），
                        // b是batch中已经处理完图片的图片的张数，
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
						// 计算预测框和真实框的IoU
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) 
                        {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    
					//下面的entry_index()计算某个矩形框的目标性参数在l.output中的索引
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];

					//计算某个矩形框的目标置信度的error值
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) 
                    {
						//对于重叠大于预定义阈值（默认值0.5）的其他先验框，不会产生任何代价。 
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) 
                    {
                        l.delta[obj_index] = 1 - l.output[obj_index];
						//float_to_box()中没有读取矩形框中包含的目标类别编号的信息，就在此处获取。
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
		// 通过移位来获取每一个矩形框的信息
        for(t = 0; t < l.max_boxes; ++t)
        {

            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
			
			// 这个if语句是用来判断一下是否有读到真实矩形框值
            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n)
            {
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
				// 获取真实标签矩形定位坐标后，与模型检测出的矩形框求IoU
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou)
                {
                    best_iou = iou; // 找出最大的IoU值
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0)
            {
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
				// 计算每个预测矩形框位置的误差项
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

				// 获取当前遍历矩形框含有目标的置信度信息c在l.output中的索引值
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
				// 叠加每个预测矩形框的目标置信度c
                avg_obj += l.output[obj_index];
				// 计算每个预测矩形框的目标置信度c的误差项
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
				//计算yolo分类参数的误差项
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1; //统计召回率@0.5
                if(iou > .75) recall75 += 1; //统计召回率@0.75
                avg_iou += iou;  
            }
        }
    }
	//计算最终的误差，其实是对各误差项平方后求和
	//只是用在打印信息
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

/*
**  yolo层后向传播函数
*/
void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i)
    {
        for(n = 0; n < l.n; ++n)
        {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh)
            {
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n)
        {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j)
            {
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

