---
categories:
 - YOLO3 源码分析
tags:
 - YOLO3
 - Keras
toc: true
mathjax: true
---



## 使用 YOLO3

### 配置 yolo.py

修改 `yolo.py` 的 `_defaults`，主要是把自己训练好的权重文件路径、锚框文件路径、类别文件路径配置上。

```python
 	_defaults = {
        "model_path": 'trained_weights_final.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'voc_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

```

### 运行 yolo_video.py

`python yolo_video.py --image` 对单张图片进行测试，`python yolo_video.py --video` 对视频进行测试。

<!-- more -->

### 图片检测源码分析

#### 基础函数

##### `letterbox_image`

位于 `\utils.py`

```python
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image
```

使用填充调整图像大小，保持纵横比不变 

参数：

+ image： `Image` 对象
+ size：目标大小。

返回：

+ 返回已经调整过大小的新的 `Image` 对象

执行过程：

+ 获取原图的宽和高，`iw, ih`
+ 获取目标的宽和高，`w, h`
+ 取 `w/iw, h/ih` 中最小的比例作为缩放比例 `scale` 
+ 按照缩放比例计算新的宽和高，`nw = iw*scale, nh=ih*scale`
+ 将 `Image` 对象按照新的宽高进行调整，仍命名为 `image`
+ 创建一个尺寸为 `size` 颜色为灰色的  `Image` 对象，命名为 `new_image`
+ 将 `image` 贴到 `new_image`  中间
+ 返回 `new_image`

##### `YOLO.__init__(self, **kwargs)`

位于 `\yolo.py`

```
 def __init__(self, **kwargs):
     self.__dict__.update(self._defaults) # set up default values
     self.__dict__.update(kwargs) # and update with user overrides
     self.class_names = self._get_class()
     self.anchors = self._get_anchors()
     self.sess = K.get_session()
     self.boxes, self.scores, self.classes = self.generate()
```

YOLO 类的初始化，主要完成以下任务：

+ 通过 `self._get_anchors()` 获得类别名
+ 通过 `self._get_anchors()` 获取锚框数组，并记为 `self.anchors`
+ 通过 `K.get_session()` 获得 session，记为 `self.sess`
+ 通过 `self.generate()` 得到 `self.boxes, self.scores, self.classes`，这一步并不会得到具体的值，只是在 YOLO 网络模型的运算图后衔接 `yolo_eval` 定义的运算图。

##### `YOLO._get_class()`

```python
def _get_class(self):
    classes_path = os.path.expanduser(self.classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
```

+ 通过 `self.classes_path` 的路径，读取类别文件，获得类别名数组

##### `YOLO._get_anchors()`

```python
def _get_anchors(self):
    anchors_path = os.path.expanduser(self.anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
```

+ 通过 `self.anchors_path)` 路径，读取锚框文件，获得锚框数组

##### `YOLO.generate()`

```python
def generate(self):
    model_path = os.path.expanduser(self.model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    # Load model, or construct model and load weights.
    num_anchors = len(self.anchors)
    num_classes = len(self.class_names)
    is_tiny_version = num_anchors==6 # default setting
    try:
        self.yolo_model = load_model(model_path, compile=False)
    except:
        self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
            if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
    else:
        assert self.yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'

    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(self.class_names), 1., 1.)
                    for x in range(len(self.class_names))]
    self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    self.colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            self.colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    self.input_image_shape = K.placeholder(shape=(2, ))
    if self.gpu_num>=2:
        self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
    boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
            len(self.class_names), self.input_image_shape,
            score_threshold=self.score, iou_threshold=self.iou)
    return boxes, scores, classes
```

`generate()` 函数主要干了三件事，加载模型、生成类别颜色框，完善模型的运算图。

执行过程：

```python
model_path = os.path.expanduser(self.model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

# Load model, or construct model and load weights.
num_anchors = len(self.anchors)
num_classes = len(self.class_names)
is_tiny_version = num_anchors==6 # default setting
try:
	self.yolo_model = load_model(model_path, compile=False)
except:
    self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
    	if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
    self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
else:
    assert self.yolo_model.layers[-1].output_shape[-1] == \
    	num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
    	'Mismatch between model and given anchor and class sizes'
```

+ 通过锚框数目判断是否为 tiny 版本
+ 尝试直接加载模型，如果 .h5 文件本身带有模型结构的话。
+ 如果 .h5 文件本身不带有模型结构，就先根据是否是 tiny 版本，创造对应的模型，然后再加载权重
+ 最终 `self.yolo_model` 储存着 yolo3 的模型

```python
# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(self.class_names), 1., 1.)
                for x in range(len(self.class_names))]
self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
self.colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        self.colors))
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.
```

+ 随机生成不同类别的预测框的颜色

```python
# Generate output tensor targets for filtered bounding boxes.
self.input_image_shape = K.placeholder(shape=(2, ))
if self.gpu_num>=2:
    self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
        len(self.class_names), self.input_image_shape,
        score_threshold=self.score, iou_threshold=self.iou)
return boxes, scores, classes
```

+ 判断 GPU 数目，如果 GPU 数目大于 2 则将模型升级为多 GPU 模型
+ 在 `self.yolo_model.output` 衔接 `yolo_eval` 定义的运算图， 通过 `yolo_eval`  可过滤大部分无效的预测框。

##### `yolo_eval()`

```python
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
```

参数：

+ yolo_outputs：`yolo_body` 模型的输出
+ anchors：锚框数组
+ num_classes：类别数目
+ image_shape： 图片尺寸
+ max_boxes：一张图片中，最多出现的预测框数目，默认 20
+ score_threshold：预测框分数阈值，默认 0.6，预测框分数指预测框置信度与类别可能性的积。
+ iou_threshold：IOU 阈值，默认 0.5

返回：

+ boxex_ ：预测框列表，每个预测框用绝对四边坐标表示。
+ scores_：预测框分数列表，上述预测框对应于的分数。
+ classes_：类别列表，上述预测框对应的类别独热码。

执行过程：

```python
num_layers = len(yolo_outputs)
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
input_shape = K.shape(yolo_outputs[0])[1:3] * 32
boxes = []
box_scores = []
for l in range(num_layers):
    _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
        anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
    boxes.append(_boxes)
    box_scores.append(_box_scores)
boxes = K.concatenate(boxes, axis=0)
box_scores = K.concatenate(box_scores, axis=0)
```

+ 先获取 `yolo_outputs` 中有多少特征图的输出，记为 `num_layers`
+ `anchor_mask` 为锚框掩码，为每一个特征图分配锚框列表中的锚框
+ `input_shape` 是指 `yolo_body` 的输入尺寸，用第一个特征图的尺寸乘 32 即可
+ 创建预测框列表 `boxes` 、预测框分数列表 `box_scores`
+ 循环遍历每个输出的特征图，操作是假设建立在第 `l` 号特征图上
+ 通过 `yolo_boxes_and_scores` 函数，`l` 号特征图中的预测框信息，和预测框分数信息。
+ 将分离出来的盒子信息追加到预测框列表 `boxes` ，预测框分数追加到预测框分数列表 `box_scores`
+ 按第一维度（不同特征图）连接 `boxes` 和 `box_scores` 中的张量，也就是说将多特征图的结果组合成在一起。

```python
mask = box_scores >= score_threshold
max_boxes_tensor = K.constant(max_boxes, dtype='int32')
boxes_ = []
scores_ = []
classes_ = []
for c in range(num_classes):
    # TODO: use keras backend instead of tf.
    class_boxes = tf.boolean_mask(boxes, mask[:, c])
    class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
    nms_index = tf.image.non_max_suppression(
        class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
    class_boxes = K.gather(class_boxes, nms_index)
    class_box_scores = K.gather(class_box_scores, nms_index)
    classes = K.ones_like(class_box_scores, 'int32') * c
    boxes_.append(class_boxes)
    scores_.append(class_box_scores)
    classes_.append(classes)
boxes_ = K.concatenate(boxes_, axis=0)
scores_ = K.concatenate(scores_, axis=0)
classes_ = K.concatenate(classes_, axis=0)

return boxes_, scores_, classes_
```

+ 设置预测框分数掩码 `mask` 要求预测框分数大于等于预测框分数阈值  `score_threshold`	
+ 设置代表最大预测框数目的张量，`max_boxes_tensor`
+ 建立最终返回的预测框列表 `boxes_` ，最终返回的预测框分数列表 `scores_`，最终返回的类别列表 `classes_`
+ 遍历每一个类别，以下操作第 `c` 号类别
+ 使用 `tf.boolean_mask` 从 `boxes`  中提取预测框分数大于阈值且预测类别为 `c` 号类的预测框，将结果记为 `class_boxes`
+ 使用 `tf.boolean_mask` 从 `c`  号类的预测框分数列表 `box_scores[:, c]` 中，提取预测框分数大于阈值且预测类别为 `c` 号类的分数值，记为 `class_box_scores` ，显然 `class_boxes` 与 `class_box_scores`  具有对应关系。
+ 使用 TensorFlow 的 nms 方法 `tf.image.non_max_suppression` ，获取 nms 后剩余的预测框的索引 `nms_index` 。
+ 使用 `K.gather` 按照索引再提取从 `class_boxes` 与  `class_box_scores` 中提取预测框与预测框分数，得到新的 `class_boxes` 与  `class_box_scores` 
+ 通过 `K.ones_like(class_box_scores, 'int32') * c` 得到新的类别独热码
+ `class_boxes` 、`class_box_scores` 、`classes` 分别追加到 `boxes_` 、`scores_`、`classes_` 中
+ 最后将 `boxes_` 、`scores_`、`classes_`  中每个类别的结果，通过 `K.concatenate` 合并在一起。
+ 返回 `boxes_` 、`scores_`、`classes_` 

##### `yolo_boxes_and_scores` 

```python
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores
```

参数：

+ feats：特征图
+ anchors：锚框数目
+ num_classes：类别数目
+ input_shape：网络输入尺寸
+ image_shape：实际输入的图片尺寸

返回：

+ 预测框数组 `boxes` 与预测框分数数组 `box_scores`

执行过程：

+ 先使用 `yolo_head` 将特诊图的输出划分为 `box_xy, box_wh, box_confidence, box_class_probs` ，`yolo_head` 在前面的博文介绍过
+ 再使用 `yolo_correct_boxes` 函数，将 `box_xy, box_wh` 依据网络的输入尺寸和实际图片的尺寸进行修正，并将它们合并在一起。把输出结果记为 `boxes` ，`boxes` 内储存着每个预测框在图片上的绝对四边坐标。
+ 对 `boxes` 的 shape 进行更改，使之变成 `[预测框数目，4]` 
+ `box_scores` 预测框分数，它值等于置信度 \* 属于某个的类别可能性
+ 对 `box_scores` 的 shape 进行更改，使之变成 `[预测框数目，类别数目]` 
+ 返回 `boxes`  和 `box_scores` 

##### `yolo_correct_boxes` 

```python
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes
```

参数：

+ box_xy：预测框中心点坐标
+ box_wh：预测框的宽高信息
+ input_shape：模型的输入尺寸
+ image_shape：实际图片尺寸

返回：

+ boxes：`shape=[...,4]` 4 分别代表 `y_min,x_min,y_max,x_max` 是在原图上的绝对坐标

执行过程：

```python
box_yx = box_xy[..., ::-1]
box_hw = box_wh[..., ::-1]
```

+ 转置最后两个维度

```python
input_shape = K.cast(input_shape, K.dtype(box_yx))
image_shape = K.cast(image_shape, K.dtype(box_yx))
new_shape = K.round(image_shape * K.min(input_shape/image_shape))
offset = (input_shape-new_shape)/2./input_shape
scale = input_shape/new_shape
box_yx = (box_yx - offset) * scale
box_hw *= scale
```

+ 获得输入尺寸张量 `input_shape`，图片尺寸张量 `image_shape`
+ 通过 ` K.min(input_shape/image_shape)` 得到 `input_shape` 与 `image_shape` 之间的最小比例
+ `image_shape` 去乘这个最小比例，结果是将 `image_shape`  所表示的尺寸进行缩放，这个尺寸恰好能放到 `input_shape` 尺寸的图片内。
+ 使用 `K.round` 对上述尺寸进行四舍五入，得到整数的尺寸值，将结果保存到 `new_shape` 中
+ 计算图像偏移比例  `offset`  等于（输入尺寸-新尺寸 / 2）/ 输入尺寸
+ 计算缩放比例 `scale` 等于输入尺寸 / 新尺寸
+ 根据偏移比例和缩放比例，计算新的 `box_yx` 和 `box_hw`

```python
box_mins = box_yx - (box_hw / 2.)
box_maxes = box_yx + (box_hw / 2.)
boxes =  K.concatenate([
	box_mins[..., 0:1],  # y_min
    box_mins[..., 1:2],  # x_min
    box_maxes[..., 0:1],  # y_max
    box_maxes[..., 1:2]  # x_max
])

boxes *= K.concatenate([image_shape, image_shape])
return boxes
```

+ 计算预测框的四边坐标在图中的比例 `y_min,x_min,y_max,x_max`
+ 将四边坐标比例合并在 `boxes` 变量内
+ 四边坐标比例乘输入图片尺寸，得到四边坐标在输入的图片中的真实尺寸，结果保存在 `boxes` 变量内
+ 返回 `boxes` 

#### 实现函数

##### `detect_img(yolo)`

位于 `\yolo_video.py`

```python
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
```

参数和返回值：

+ 参数 `yolo` 是一个 YOLO 实例，YOLO 类被定义在 `\yolo.py` 中

执行过程：

+ 通过 `input()` 内置函数读取图片路径
+ 使用 `PIL` 的 `Image` 从这个路径上读取图片到变量，`image`
+ 调用 YOLO 对象的 `detect_image` 方法，该方法返回一个 `Image` 对象 ，记为 `r_image`
+ 展示 `r_image`，并关闭 `session`

##### `YOLO.detect_image(self, image)`

位于 `\yolo.py`

```python
def detect_image(self, image):
    start = timer()

    if self.model_image_size != (None, None):
        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = self.sess.run(
        [self.boxes, self.scores, self.classes],
        feed_dict={
            self.yolo_model.input: image_data,
            self.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = self.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    end = timer()
    print(end - start)
    return image
```

参数和返回值：

+ 参数是一个 `Image`  对象，返回一个 `Image` 对象。

执行过程：

```
start = timer()
```

+ `timer()` 定义于 `from timeit import default_timer as timer` 用户获取当前时间，变量 `start` 将配合变量 `end` 实现度量单张图片处理所用时长。

```python
if self.model_image_size != (None, None):
    assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
    assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
    boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
else:
    new_image_size = (image.width - (image.width % 32),
    image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
image_data = np.array(boxed_image, dtype='float32')
```

+ 将从参数传入的 `Image` 对象重新调整大小，调整到 `self.model_image_size` 规定的大小，并转换成 np 数组，记为 `image_data` 。

```
print(image_data.shape)
image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
```

+ 对 `image_data` 进行处理，首先使它的值缩放到 0 到 1 ，其次对它增加批维度

```python
out_boxes, out_scores, out_classes = self.sess.run(
	[self.boxes, self.scores, self.classes],
    feed_dict={
        self.yolo_model.input: image_data,
        self.input_image_shape: [image.size[1], image.size[0]],
        K.learning_phase(): 0
	})
```

+ 在类初始化阶段，已经调用  `self.generate()` 得到 `self.boxes, self.scores, self.classes` 这其实是在原有模型的计算图后，衔接了新的运算图。通过 `self.sess.run()` 可以得到，当给一下结点规定指定值时：

    ```
    self.yolo_model.input: image_data,
    self.input_image_shape: [image.size[1], image.size[0]],
    K.learning_phase(): 0
    ```

    计算图中 `[self.boxes, self.scores, self.classes]` 变量的值，将返回值保存为`out_boxes, out_scores, out_classes`

```python
font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness = (image.size[0] + image.size[1]) // 300

for i, c in reversed(list(enumerate(out_classes))):
    predicted_class = self.class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=self.colors[c])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=self.colors[c])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
```

+ 根据 `out_boxes, out_scores, out_classes` 在原图上绘制预测框，绘制完预测框后的图像，保存在 `image` 中

```python
end = timer()
print(end - start)
return image
```

+ 记录结束时间，并打印总耗时
+ 返回 `image` 变量

### 视频检测源码分析

#### 实现函数

##### `detect_video`

位于 `\yolo.py`

```python
def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
```

+ 原理就是用 cv2 截视频里的每一帧，然后调用 `image = yolo.detect_image(image)` 对每一帧进行识别和输出。

## 性能评价

### TP TN FP FN 的定义

从样本角度分类，可分为两类：

+ 正例（Positives）
+ 负例（Negatives）

从分类器的结果正确与否，可以为两类：

+ 分类正确（True ）
+ 分类错误（False ）

那么结合上述两种划分，可以得到四类：

+ 分类正确的正例（True Positives，**TP**）
+ 分类正确的负例（True  Negatives，**TN**）
+ 分类错误的正例（False Positives，**FP**）
+ 分类错误的负例（False Negatives，**FN**）

### 查准率（Precision）

查准率（Precision）表示在所有分类为正例的情况下（包括真正例和假反例），分类正确的正例占了多少比例。
$$
Precision=\frac{TP}{TP+FN}
$$


### 召回率（Recall，又称为 TPR）

召回率（Recall, 又称为 TPR）表示在所有正例中（包括真正例和假正例），分类正确的正例占了多少比例。
$$
Recall=\frac{TP}{TP+FP}
$$

### 交并比（Intersection over Union，IOU）

IOU (intersection over union) 为检测结果（预测框）与真实框 Ground Truth 的交集面积比上它们的并集面积。

设 A , B 为表示平面的集合，S 为求面积的函数，则：
$$
IOU(A,B)=\frac{S(A\cap B)}{S(A\cup B)}
$$


### 目标检测任务中的 TP FP

将预测框以分数为依据，按照某一顺序进行排序。若预测框的分数大于某一阈值（Score threshold），则被视为预测为正例（Positives）。

同时，设立一个 IOU 阈值（IOU threshold），如果一个被视为正例的预测框与真实框的 IOU 大于该 IOU 阈值，则表示这是一个正确的预测，即 TP；小于该阈值，则说明这是一个错误的预测，记为 FP。

如果对于同一个真实框有多个预测框满足 IOU 大于阈值，此时**只将 IOU 最大的作为 TP**，其余作为FP。

在 yolo3 网络中，预测框的分数，等于预测框的置信度与类别概率的积。

### 目标检测任务中的查准率（Precision）与召回率（Recall）

当给定一组分数阈值和 IOU 阈值，便可以求出一个类别中的 TP 与 FN，**TP+FN 等于该类别的预测框总数** 。同时 **TP+FP 等于该类别的真实框总数**，也是已知数据。所以给定两个阈值便可由 $Precision=\frac{TP}{TP+FN}$ 与 $Recall=\frac{TP}{TP+FP}$ 求得一组数据 $(r,p)$ 。

### AP 与 mAP

通常 IOU 阈值不变，被设定为 0.5 ，所以对于某一个类别而言，不同的分数阈值将对应不同的 $(r,p)$ ，给定 n 个分数阈值则可求出 n 个 $(r,p)$ 。将召回率（Recall）作为横轴，查准率（Precision）作为纵轴，n 个 $(r,p)$ 呈现的图像被称为 P-R 图，一个类别的 AP 就是 P-R 图上各点横纵坐标之积的和，为简化运算通常先将 P-R 图变化为单调递减再计算 AP

**如何得到 n 个  $(r,p)$  点**

1. 将总数为 N 的预测框按照分数进行降序排序，判断每个预测框是 TP 还是 FP
2. 取前 k 个预测框为一组，求这一组预测框的召回率（Recall）和查准率（Precision），k取 （1，2，...，N），得到 N 个 $(r,p)$  点

**如何计算 P-R 图上的 AP**

1. 补全区间，添加 $(0,1),(1,0)$ 两个  $(r,p)$  点

2. 将 P-R 图像变成单调递减的图像，参考以下代码：

    ```python
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    ```

    即，如果后一项比前一项大，则令前一项等于后一项。通过这种方式保证只能前一项大于等于后一项。

2. 计算 AP，参考以下代码：

    ```python
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
        i_list.append(i)
    
    ap = 0.0
    for i in i_list:
    	ap += ((mrec[i]-mrec[i-1])*mpre[i])
    ```

    即，如果rec的某项与前一项相比变化了，则记录该项索引，构建一个索引列表，最后根据索引列表，计算 AP

**如何计算 mAP**

mAP 是所有类别 AP 的平均值。

**参考链接**

https://blog.csdn.net/qq_35916487/article/details/89076570?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

https://blog.csdn.net/zdh2010xyz/article/details/54293298?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

https://blog.csdn.net/weixin_38106878/article/details/89199961

https://blog.csdn.net/plsong_csdn/article/details/89502117