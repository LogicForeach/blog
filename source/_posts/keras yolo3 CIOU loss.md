---
title: keras yolo3 使用 CIOU Loss
date: {{ date }}
categories:
 - 深度学习
tags:
 - YOLO
 - CIOU
 - Loss
toc: true
mathjax: true
---

### 描述

本文设计了 keras 实现的 CIOU 函数，并分析  keras YOLO3 源码，在原有代码的基础上进行修改，使用 CIOU Loss 替换原来的位置大小回归。

<!--more-->

## 原 keras yolo3 loss 分析

参考链接：https://blog.csdn.net/lzs781/article/details/105086179

### 关键函数分析

#### yolo_head

```python
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

```

该函数用于从最终输出的特征图里提取预测框信息

**参数 :**

+ **feats** : 特征图，通道数为 `5+类别数目`
+ **anchors :** 特征图中所含锚框，结构为 `[[w1,h1],[w2,h2],...]`
+ **num_classes** : 类别数目
+ **input_shape** :  原图尺寸信息，`(高,宽)`
+ **calc_loss** : 是否用于计算 loss 值 

**返回 :**

+ 如果 `calc_loss == True` ，则返回 `grid, feats, box_xy, box_wh`

+ 否则返回 `box_xy, box_wh, box_confidence, box_class_probs`

+ 其中`grid, feats, box_xy, box_wh, box_confidence, box_class_probs` 分别是网格坐标信息、原始特征图信息、预测框中心点坐标比例（相对于原图）、预测框大小比例（相对于锚框）、置信度、类别信息。

+ 形状的形状信息为：

    ```
    grid.shape=(特征图高,特征图宽,1,2)
    feats.shape=(批数,特征图高,特征图宽,锚框数,5+类别数)
    box_xy.shape=(批数,特征图高,特征图宽,锚框数,2)
    box_wh.shape=(批数,特征图高,特征图宽,锚框数,2)
    box_confidence.shape=(批数,特征图高,特征图宽,锚框数,1)
    box_class_probs.shape=(批数,特征图高,特征图宽,锚框数,类别数)
    ```

#### yolo_loss

```python
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss

```

**参数：**

+ **args** ：包含`(yolo_outputs,y_true)` ，`yolo_outputs` 指 YOLO3 模型输出的 y1，y2，y3，这里的输出是含有批维度的，且其第一维度为批维度。`y_true` 是经过`preprocess_true_boxes`函数预处理的真实框信息：

    + `yolo_outputs` 是三元素列表，其中元素分别为`[m批13*13特征图张量,m批26*26特征图张量,m批52*52特征图张量]`，每张特征图的深度都为`图内锚框数*(5+类别数)`，所以列表内每个元素的 `shape=(批数,特征图宽,特征图高,图内锚框数*(5+类别数))`
    + `y_true`  是三元素列表，列表内是 np 数组，每个 np 数组对于不同尺寸的特征图，它的形状为 `shape=(批数,特征图宽,特征图高,图内锚框数,5+类别数)`，每个特征图的尺寸为 `13*13、26*26、52*52`

    > 关于 `yolo_outputs`  和  `y_true`  的形状分析可参考前几篇博文

+ **anchors** : 锚框二维数组，结构如`[[w1,h1],[w2,h2]..]`

+ **num_classes** ：整型，类别数

+ **ignore_thresh** ：浮点型，IOU 小于这个值的将被忽略。

**返回：**

+ 一维向量，loss值。

## 改造 yolo_loss

### 关键变量分析

#### `y_true`

+ 类型为三元素列表，每个元素是一个张量，分别表示`[m批13*13特征图张量,m批26*26特征图张量,m批52*52特征图张量]`
+ 形状 `shape=(批数,特征图宽,特征图高,图内锚框数,5+类别数)`
+ 数值为位置大小信息（x，y，w，h）+ 置信度信息 confidence + 类别的独热码，其中位置大小信息是相对于原图的比例数据

#### `pred_box` 与  `raw_pred[0:4]`

`pred_box` 是由 `[pred_xy, pred_wh]` 连接而成

+ 类型为张量
+ 形状  `shape=(批数,特征图高,特征图宽,锚框数,4)`
+ 数值为位置大小信息（x，y，w，h）, 数值为相对于原图的比例

`raw_pred[0:4]` 是特征图输出的切片

+ 类型为张量
+ 形状`shape=(批数,特征图高,特征图宽,锚框数,4)`
+ 数值为未经归一化处理的位置大小信息（x，y，w，h）, 只是网络输出的数据

#### `raw_true_xy` 与 `raw_true_wh`

+ 类型均为张量
+ 形状均为`shape=(批数,特征图高,特征图宽,锚框数,2)`
+ 数值是将 `y_true` 中的 位置大小信息逆运算，使它意义与 `raw_pred[0:4]` 一致

#### 小结

+ `y_true` 与 `pred_box`  在大小位置信息（x，y，w，h）上意义一致
+ `raw_true_xy`  、`raw_true_wh` 与 `raw_pred[0:4]` 意义一致

### CIOU LOSS

参考链接：https://blog.csdn.net/lzs781/article/details/105515150
$$
CIOU(B,B^{gt})=DIOU(B,B^{gt})-\alpha\upsilon\\L_{reg}(B,B^{gt})=1-CIOU(B,B^{gt})
$$

其中
$$
\alpha=\frac{\upsilon}{(1-IOU)+\upsilon}\\  \upsilon = \frac{4}{\pi ^2}\left(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h} \right)^2
\\DIOU(B,B^{gt}) = IOU(B,B^{gt}) - \frac{\rho^{2}(b,b^{gt})}{c^{2}}
$$

所以实现 CIOU loss 的核心是实现 IOU 运算，这里选择将 `y_true` 与 `pred_box`  作为参数构造 CIOU 运算函数

### 代码实现

**创建 ciou.py**

```python
from keras import backend as K
import numpy as np
import tensorflow as tf

def ciou(true_boxes,pred_box):
    '''
    true_boxes: shape=(批数,特征图高,特征图宽,锚框数,4) (x,y,w,h)
    pred_box: shape=(批数,特征图高,特征图宽,锚框数,4) (x,y,w,h)
    return ciou shape=(批数,特征图高,特征图宽,锚框数,1) 
    '''
    b1_xy = true_boxes[..., :2]
    b1_wh = true_boxes[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = pred_box[..., :2]
    b2_wh = pred_box[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    iou = intersect_area / (b1_area + b2_area - intersect_area)

    outer_mins = K.minimum(b1_mins, b2_mins)
    outer_maxes=K.maximum(b1_maxes, b2_maxes)
    outer_diagonal_line = K.square(outer_maxes[...,0]-outer_mins[...,0])+K.square(outer_maxes[...,1]-outer_mins[...,1])


    center_dis=K.square(b1_xy[...,0]-b2_xy[...,0])+K.square(b1_xy[...,1]-b2_xy[...,1])

    # TODO: use keras backend instead of tf.
    v = (4.0/(np.pi)**2) * tf.math.square((
            tf.math.atan((b1_wh[...,0]/b1_wh[...,1])) -
            tf.math.atan((b2_wh[..., 0] / b2_wh[..., 1])) ))
    alpha = tf.maximum(v / (1-iou+v),0) # (1-iou+v) 在完全重合时等于 0 , 0/0=-nan(ind)

    

    ciou = iou - (center_dis / outer_diagonal_line + alpha*v)
    ciou=K.expand_dims(iou, -1)
    return ciou
```

**修改 model.py，在 yolo_loss 处：**

主要是使用

```python
reg_loss=object_mask*(1-ciou(
    true_boxes=y_true[l][..., :4],
    pred_box=pred_box
))

```

代替原来的位置大小回归。

我的 yolo_loss 是这样：

```python
from yolo3.ciou import ciou

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh,box_class_probs = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        # raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
       
       
        # raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        # raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
       
       
        # box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        # wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])

        # 使用 icou 作为回归损失函数
        reg_loss=object_mask*(1-ciou(
            true_boxes=y_true[l][..., :4],
            pred_box=pred_box
        ))

        # confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        #     (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            0.1*(1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
           
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, box_class_probs, from_logits=False)
        
        # xy_loss = K.sum(xy_loss) / mf
        # wh_loss = K.sum(wh_loss) / mf
        # confidence_loss = K.sum(confidence_loss) / mf
        # class_loss = K.sum(class_loss) / mf
        # loss += xy_loss + wh_loss + confidence_loss + class_loss

        reg_loss=K.sum(reg_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += reg_loss + confidence_loss + class_loss
        
        
        # if print_loss:
        #     loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
```
