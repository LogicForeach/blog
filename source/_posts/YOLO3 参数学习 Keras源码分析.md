---
categories:
 - YOLO3 源码分析
tags:
 - YOLO3
 - Keras
toc: true
mathjax: true
---

## 参数学习

### 损失函数

Keras 源码中的损失函数：

$$
loss(object)=-\sum_{i=0}^{K\times K}\sum_{j=0}^M I_{ij}^{obj} \cdot (2-w_i\times h_i) \cdot [\hat x_ilog(x_i)+(1-\hat x_i)log(1-x_i)]-\\ 
\sum_{i=0}^{K\times K}\sum_{j=0}^M I_{ij}^{obj} \cdot (2-w_i\times h_i) \cdot [\hat y_ilog(y_i)+(1-\hat y_i)log(1-y_i)]+\\ 
0.5 \cdot \sum_{i=0}^{K\times K}\sum_{j=0}^M I_{ij}^{obj} \cdot (2-w_i\times h_i) \cdot [(w_i-\hat w_i)^2+(h_i-\hat h_i)^2]-\\
\sum_{i=0}^{K\times K}\sum_{j=0}^M I_{ij}^{obj} \cdot [\hat C_ilog(C_i)+(1-\hat C_i)log(1-C_i)]-\\
\sum_{i=0}^{K\times K}\sum_{j=0}^M I_{ij}^{noobj} \cdot [\hat C_ilog(C_i)+(1-\hat C_i)log(1-C_i)]-\\
\sum_{i=0}^{K\times K}\sum_{j=0}^M \sum_{c \in classes} I_{ij}^{obj} \cdot [\hat p_i(c)log(p_i(c))+(1-\hat p_i(c))log(1-(p_i(c))]\\
$$
$K \times K$ 是网格数目，$M$ 是每个网格锚框数目，$I_{ij}^{obj}$ 表示 `i` 号 网格中 `j` 号锚框是否负责物体，所谓负责物体就是指是否有物体的中心落到这个锚框，如果有物体就落入则值，没物体落入则值为0。$w_i,h_i,x_i,y_i$ 表示网络预测的盒子长宽和中心位置，戴帽子的表示的是真实的长宽和中心位置。$C_i$ 是网络的预测的置信度，带帽是真实置信度。$p_i(c)$ 表示类别为 `c` 的概率，带帽表示真实概率。

<!-- more -->

### 代码实现

#### 基础函数

##### `yolo_head`

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

参数 :

+ **feats** : 特征图，通道数为 `5+类别数目`
+ **anchors :** 特征图中所含锚框，结构为 `[[w1,h1],[w2,h2],...]`
+ **num_classes** : 类别数目
+ **input_shape** :  原图尺寸信息，`(高,宽)`
+ **calc_loss** : 是否用于计算 loss 值 

返回 :

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


执行过程 :

```python
"""Convert final layer features to bounding box parameters."""
num_anchors = len(anchors)
# Reshape to batch, height, width, num_anchors, box_params.
anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
```

+ 获取锚框数目 `num_achors` 
+ 将 `anchors` 转化为 tf 张量 `anchors_tensor`，并将形状改变为`shape=1, 1, 1, num_anchors, 2` 

```python
grid_shape = K.shape(feats)[1:3] # height, width
grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
[1, grid_shape[1], 1, 1])
grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
[grid_shape[0], 1, 1, 1])
grid = K.concatenate([grid_x, grid_y])
grid = K.cast(grid, K.dtype(feats))
```

+ 获取特征图尺寸 `grid_shape`

+ 通过 `K.arange(0, stop=grid_shape[0])` 生成一个长度为特征图高度的向量，元素值是`0,1,2,...,grid_shape[0]-1` ; 用`a` 代指上述向量，利用 `K.reshape(a, [-1, 1, 1, 1])` 将上述向量变成 4 维张量，张量 `shape=(特征图高,1,1,1)` ；用 `b` 代指上述张量，利用 `K.tile(b, [1, grid_shape[1], 1, 1])` 将上述张量，变成 `shape=(特征图高,特征图宽,1,1)` 的 4 维向量，将结果记为 `grid_y` 

    `grid_y`  表示每个格子的纵坐标，比如`grid_y[5,9,0,0]==5` 意思是：特征图中横坐标是 9 、纵坐标是 5 的像素点的纵坐标是 5，**这里特征图的一个像素点被称为一个网格**。

+ 相似的手段求 `grid_x`

    `grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])` ，注意求 `grid_x` 和求 `grid_y` 不一样的地方在于 `-1` 的位置。

    `grid_y[5,9,0,0]==9` 意思是：特征图中横坐标是 9、纵坐标是 5 的网格的横坐标坐标是 5。

+ 通过 `K.concatenate` 连接 `grid_x` 和 `grid_y`  并记为`grid`，注意`K.concatenate` 默认是沿着最后一个轴连接。将 `grid` 转化为浮点型。

    `grid` 的性质是`(特征图高,特征图宽,1,2)` , 它表示每个格子的横纵坐标，比如：`grid[5,9,1]==(9,5)` 即横坐标是 9、纵坐标是 5 的格子（像素点）坐标是`(9,5)`

```python
feats = K.reshape(
 	feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

# Adjust preditions to each spatial grid point and anchor size.
box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
box_confidence = K.sigmoid(feats[..., 4:5])
box_class_probs = K.sigmoid(feats[..., 5:])
```

+ 参数 `feats` 变化维度为 `(批数,特征图高,特征图宽,锚框数,5+类别数)` 这样的目的是和标注信息形状一致
+ 计算预测框中心点比例 `box_xy`，计算方法是：将特征图 0 通道和 1 通道的信息经过`K.sigmoid()` 压缩至 0 到 1 后，加上网格坐标信息，将上述张量和分别除以网格的高和宽，最终得到一个相对于网格大小的坐标比例信息，因为网格大小由原图大小缩放而来，故可**认为相对于网格大小的坐标比例信息，就是相对于原图的坐标比例信息**。
+ 同理通过将特征图的 2 通道和 4 通道的数据输送给`K.exp()`，再乘上锚框张量 `anchors_tensor` , 除以网格高和宽，得到相对于锚框的大小比例信息 `box_wh`
+ 将 4 通道数据通过 `K.sigmoid()` 计算得置信度信息 `box_confidence`
+ 将剩下的通道的数据通过 `K.sigmoid()` 计算得到类别概率

```python
if calc_loss == True:
    return grid, feats, box_xy, box_wh
return box_xy, box_wh, box_confidence, box_class_probs
```

+ 根据 `calc_loss` 参数返回相应的变量

##### `box_iou`

```python
def box_iou(b1, b2):
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
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

    return iou
```

实现 IOU 运算

#### 实现函数

##### `yolo_loss`

关于损失函数都被定义在`\yolo3\model.py` 中：

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

参数：

+ **args** ：包含`(yolo_outputs,y_true)` ，`yolo_outputs` 指 YOLO3 模型输出的 y1，y2，y3，这里的输出是含有批维度的，且其第一维度为批维度。`y_true` 是经过`preprocess_true_boxes`函数预处理的真实框信息：

    + `yolo_outputs` 是三元素列表，其中元素分别为`[m批13*13特征图张量,m批26*26特征图张量,m批52*52特征图张量]`，每张特征图的深度都为`图内锚框数*(5+类别数)`，所以列表内每个元素的 `shape=(批数,特征图宽,特征图高,图内锚框数*(5+类别数))`
    + `y_true`  是三元素列表，列表内是 np 数组，每个 np 数组对于不同尺寸的特征图，它的形状为 `shape=(批数,特征图宽,特征图高,图内锚框数,5+类别数)`，每个特征图的尺寸为 `13*13、26*26、52*52`

    > 关于 `yolo_outputs`  和  `y_true`  的形状分析可参考前几篇博文

+ **anchors** : 锚框二维数组，结构如`[[w1,h1],[w2,h2]..]`

+ **num_classes** ：整型，类别数

+ **ignore_thresh** ：浮点型，IOU 小于这个值的将被忽略。

返回：

+ 一维向量，loss值。

执行过程

```python
 num_layers = len(anchors)//3 # default setting
 yolo_outputs = args[:num_layers]
 y_true = args[num_layers:]
 anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
 input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
 grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
 loss = 0
 m = K.shape(yolo_outputs[0])[0] # batch size, tensor
 mf = K.cast(m, K.dtype(yolo_outputs[0]))
```

+ 获取输出特征图数目 `num_layers` ，YOLO3 输出 3 张特征图，每张特征图内有 3 个锚框，而tiny-yolo3 则输出两张，故可以根据 `len(anchors)` 来计算输出特征图数目，**以下假设输出特征图数目为 3**
+ 通过 `num_layers`  对参数 `args` 进行分割，得到 `yolo_outputs` 和 `y_true`
+ 定义锚框掩码 `anchor_mask` , 锚框掩码用于给每个输出特征图分配锚框。
+ 第一个输出特征图的尺寸为 `K.shape(yolo_outputs[0])[1:3]`，由第一个输出特征图的尺寸\*32，可知原图尺寸 `input_shape`
+ 由每个输出特征图的尺寸 `K.shape(yolo_outputs[0])[1:3]` 可求每个特征图内的网格信息 `grid_shapes` 。
+ 求批大小`m` ，并将批大小`m` 转化成 tf.Tensor 类型，记为 `mf`

```
for l in range(num_layers):
```

+ 之后是对每张输出特征图的操作，以下假设操作第 `l` 号特征图

```python
object_mask = y_true[l][..., 4:5]
true_class_probs = y_true[l][..., 5:]

grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
	anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
pred_box = K.concatenate([pred_xy, pred_wh])
```

+ 从 `y_true` 里获得物体掩码 `object_mask` 和类别概率 `true_class_probs`
+ 为了由 `l` 号特征图的信息提取得到预测框相关信息，调用 `yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)`，并将其返回值记为 `grid, raw_pred, pred_xy, pred_wh`，分别表示：网格坐标、原始特征图信息、预测框的中心点比例信息（相对于原图的比例）、预测框的大小比例信息（相对于锚框的比例）
+ 计算预测框信息 `pred_box` ，它的值就是合并预测的位置信息和预测的大小信息 `pred_box = K.concatenate([pred_xy, pred_wh])`

```python
# Darknet raw box to calculate loss.
raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]
```

+ 对真实盒子信息进行处理，按逆运算，从 `l` 号特征图的真是信息 `y_true[l]` 中求出真实框的中心点比例信息 `raw_true_xy` 和真实框大小比例信息  `raw_true_wh` 
+ 利用 `K.switch` ，通过 `object_mask` 对 `raw_true_wh`  进行修正，`raw_true_wh`  中含有物体的网格包含真实框的大小比例信息，不包含物体的网格大小为 0
+ 定义修正比例 `box_loss_scale` 这个比例被赋值为 `2-w*h` ，最终要乘到坐标和大小的误差项中，意味着 loss 函数对小物体的误差比大物体的误差更敏感。

```python
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
```

+ 定义 `ignore_mask` ，它的形状与 `pred_box` 只有最后一个维度不同，`ignore_mask` 的最后一维为 1，通过 tf 的静态图控制语句，动态定义一个掩码张量，用每批预测框张量与 `l` 层特征图中的每批真实框张量进行 IOU 运算，如果一个预测框存在一个与 IOU 大于参数`ignore_thresh` 的真是框，就在这个 `ignore_mask`  中把这个预测框对应位置的元素置值为 1 ，将无效预测框过滤掉。

```python
# K.binary_crossentropy is helpful to avoid exp overflow.
xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
(1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
```

+ 按照损失函数计算  `xy_loss` 、`wh_loss`、`confidence_loss` 、`class_loss`

```python
xy_loss = K.sum(xy_loss) / mf
wh_loss = K.sum(wh_loss) / mf
confidence_loss = K.sum(confidence_loss) / mf
class_loss = K.sum(class_loss) / mf
loss += xy_loss + wh_loss + confidence_loss + class_loss
```

+ 将 `l` 号特征图的各 loss 项求和并除批大小，最终化为标量，随后将各标量加和到 `loss`

```python
return loss
```

+ 处理好所有特征图后，返回 `loss` ，`loss.shape=(1,)`

## 训练模型

#### 基础函数

##### `get_classes`


```python
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
```

+ 从 classes_path 中按行读取类别，构建类别列表并返回

##### `get_anchors`


```python
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
```

+ 从 anchors_path 按行读取锚框信息，构建锚框列表并返回

##### `create_model()`


```python
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model
```

参数：

+ `input_shape` : 二维列表或张量，输入图片的尺寸，高在前，宽在后
+ `anchors` : 锚框信息列表，结构类似 `[[w1,h1]...]`
+ `num_classes` ：整型，类别数目
+ `load_pretrained` ：是否加载预训练的权重，默认是 `True`
+ `freeze_body`：设置冻结那些层
+ `weights_path`：预训练权重的存储路径

分段讲解：


```python
K.clear_session() # get a new session
image_input = Input(shape=(None, None, 3))
h, w = input_shape
num_anchors = len(anchors)
```

+ 创建 Input 类型张量作为静态图的输入结点，从 `input_shape` 中分离高 `h` 和宽 `w`，获取锚框数目 `num_anchors`


```python
y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
```

+ 构建 `y_true` ，`y_true` 是一个拥有三个 Input 张量的列表，三个 Input 张量的尺寸分别为 `原图长宽/32,原图窗口/16,原图长宽/8`，每个 Input 张量的通道数都是 `类别数+5`


```python
model_body = yolo_body(image_input, num_anchors//3, num_classes)
print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
```

+ 构建 yolo 网络，记为`model_body`，通过 `yolo_body`
+ `yolo_body` 是一个定义在 `model.py` 的函数，里面定义了 输入结点里的数据在静态图中的流转过程。


```python
if load_pretrained:
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        # Freeze darknet53 body or freeze all but 3 output layers.
        num = (185, len(model_body.layers)-3)[freeze_body-1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
```

+ 根据参数 `load_pretrained`  判断是否需要加载预训练权重
+ 如果需要加载预训练权重，则从预训练权重路径加载权重，并忽略不匹配的层
+ 计算总共需要冻结的层 `num` ，`load_pretrained=1` 则冻结 darknet53，如果`load_pretrained=2` 就除了倒数三层（三个用于输出的层）其余层都冻结。
+ 把 0 到 `num` 层全冻结。


```python
model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
    [*model_body.output, *y_true])
model = Model([model_body.input, *y_true], model_loss)

return model
```

+ 使用 `keras.layers.Lambda` 自定义一个层，这个层使用的是 `yolo_loss` 函数，层的输出形状是 `(1,)`，名称为 yolo_loss，输入的参数是 `{'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}`

    > 使用 `Lambda`  可以将一个函数转化为层对象，使之可以被添加进模型之中，要求函数的第一个参数是输的入张量，其余参数可以通过 `arguments` 以字典的形式传入

+ 将 `[*model_body.output, *y_true]` 作为输入的张量，输入到 `Lambda` 定义的层的，将其返回的张量定义为 `model_loss` ，该过程另静态图中衔接了新的结构，训练时从输入结点输入的数据将会先通过 `model_body` 静态图，再通过 `Lambda` 定义的静态图，最后输送到输出结点
+ 使用 `keras.models.Model` 将 `[model_body.input, *y_true]` 作为输入结点，将 `model_loss` 作为输出结点，构建静态图，并返回模型。

> `create_model()` 创建的并非是 YOLO3 网络本身，而是 YOLO3 网络，加上 loss 函数层，在定义 loss 函数时，只需要将最后一层的输出作为最终loss值即可，也就是说，loss 函数的运算部分作为模型的最后层被 `create_model()`  所创建。

#### 实现函数


```python
def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.
```

分段讲解：


```python
annotation_path = 'train.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
```

+ 配置相关文件路径，`annotation_path` 是由 voc_annotation.py 生成的 YOLO3 格式的批注文件的路径, `log_dir` 用于保存 TensorBoard 和 checkpoint;`classes_path` 是 YOLO3 类别文件路径; `anchors_path` 是 YOLO3 锚框(预设框)文件路径


```python
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw

is_tiny_version = len(anchors)==6 # default setting
```

+ 定义相关变量，`class_names` 是由 `get_classes(classes_path)` 生成的类别名数组; `num_classes` 是类别数据中的类别的数目; `anchors` 是由 `get_anchors(anchors_path)` 生成的锚框坐标大小信息数组; `input_shape` 定义输入图像的尺寸，高在前宽在后。`is_tiny_version` 通过判断锚框数目断定是否是 tiny yolo。


```python
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
```

+ 根据 `is_tiny_version` 变量判断使用的网络版本,不同版本通过不同函数创建网络,如 yolo3 网络通过 `create_model()` 创建.
+ 使用 `create_model()` 创建模型的时候，通过参数设计冻结两层，并设置好预训练权重路径 


```python
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
```

+ 设置 TensorBoard 回调，用于保存训练过程中的信息，进行可视化，实例名记为 `logging`
+ 设置 ModelCheckpoint 回调函数，用于保存训练过程的权重，参数解读：保存权重的路径为 `log_dir` ，文件名格式为`ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5` ，监视的目标是在测试集的loss值 `val_loss` ，只保存权重不保存模型结构，只保存结果好与已有结果的权重，每 3 轮保存一次。记实例名为 `checkpoint`
+ 设置 ReduceLROnPlateau 回调，当 loss 不再下降的时候，按照一定因数下调学习率，继续训练。根据参数解读：监视的 loss 为训练集上的 loss 值 `val_loss` ,每次调整学习率的因数为 0.1，根据公式 `new_lr = lr * factor` 可知，每次将学习率缩小十倍，如果 `val_loss` 连续 3 轮没有下降，则降低学习率，日志模式为 1。记实例名为 `reduce_lr`。
+ 设置 EarlyStopping 回调，当 loss 值不再下降时，提前停止训练，参数解读：监视目标为 `val_loss` ，最小下降值为 0，如果连续 10 论没有下降则早停，日志模型为 1。记实例名为 `early_stopping`


```python
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
```

+ 给定测试集划分比例为 0.1 ，总载入样本的 `val_split` 部分将会作为测试集，不参与训练。 
+ 将 `annotation_path` 中的信息读入到 `lines` 列表中
+ 对 `lines` 按照 `val_split` 进行划分，将测试数目记为 `num_val`, 训练集数目记为 `num_train`


```python
# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 32
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
```

+ 进行第一阶段训练,该阶段会冻结主干网络的权重(已经已经预加载了一些),通过调整部分权重使模型快速的 loss 快速下降到一个可接受的值。

+ 编译模型，使用 `Adam` 优化器，名为 `yolo_loss`  的输出经过函数 `lambda y_true, y_pred: y_pred` 得到 loss 值

    > 这里的 loss 函数把模型最后一层的输出作为 loss 值，原因是 `create_model()` 把 loss 函数的运算部分定义在了模型的最后一层

+ 批大小设置为 32

+ 训练数据，通过 `data_generator_wrapper()` 生成器生成训练数据，用 `data_generator_wrapper` 生成验证数据，使用 `logging` 和 `checkpoint` 作为回调对象

+ 将训练好的权重保存起来


```python
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')

    batch_size = 32 # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'trained_weights_final.h5')
```

+ 第二阶段训练，这个阶段将会解冻所有层，在第一阶段训练的基础上进行微调。
+ 用 `for` 解冻所有层
+ 编译模型，与第一阶段相同
+ 批大小为 32
+ 训练模型，数据载入与第一阶段相同，使用 `logging,checkpoint,reduce_lr,early_stopping` 作为回调
+ 保存最终模型