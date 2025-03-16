# 数据预处理函数
'''
边界框可以用多种方式表示，常见的格式包括如下：
1.存储角点的坐标[xmin, ymin, xmax, ymax]
2.存储中心的坐标和长方体尺寸[x, y, width, height]
由于需要用到两种模式，所以编写用于在次两种格式之间转换的函数。
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def swap_xy(boxes):
    '''交换边界框的x和y坐标的顺序
    属性如下：
        boxes：形状为(num_boxes, 4)的张量，表示边界框
    返回值说明如下：
        交换了x和y坐标顺序后的边界框的形状与原边界框的形状相同。
    '''
    return tf.stack([boxes[:, 1], boxes[:, 0],
                     boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    '''转变为中心点长方体模式'''
    return tf.concat([(boxes[..., :2] + boxes[..., 2:]) / 2.0,
                      boxes[..., 2:] - boxes[..., :2]],
                      axis=-1,
                      )

def convert_to_corners(boxes):
    '''转变为角点坐标模式'''
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0,
         boxes[..., :2] + boxes[..., 2:] / 2.0],
         axis=-1
    )

def compute_iou(boxes1, boxes2):
    '''计算交并比'''
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    # 获取交集
    intersection = tf.maximum(0.0, rd-lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    # 获取并集
    union_area = tf.maximum(
        boxes1_area[:,None] + boxes2_area - intersection_area, 1e-8
    )
    # 返回交并比
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def visualize_detections(image, boxes, classes, scores, figsize=(7,7),
                         linewidth=1, color=[0,0,1]):
    '''可视化检测'''
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    # 获取当前轴对象
    ax = plt.gca()
    for box, _cls, score, in zip(boxes, classes, scores):
        text = f'{_cls}: {score:.2f}'
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False,
                              edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox = {"facecolor": color, "alpha": 0.4},
            clip_box = ax.clipbox, # 裁剪边界外元素
            clip_on = True,
        )
    plt.show()
    return ax

class AnchorBox:
    '''锚固箱类，是一组预定义的参考框，用于帮助模型快速定位和识别'''
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]    #锚框的长宽比（宽：高）
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]   #缩放因子（基础尺寸的倍数）

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)  # 每个位置生成的锚框数量
        self._strides = [2 ** i for i in range(3, 8)]   # 各层特征图的步长（对应FPN的P3-P7层）
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]   # 各层锚框的基础面积
        self._anchor_dims = self._compute_dims()    # 预计算所有锚框的宽高
    
    def _compute_dims(self):
        '''计算锚框的宽高尺寸'''
        anchor_dims_all = []
        for area in self._areas:    # 遍历每个层级
            anchor_dims = []
            for ratio in self.aspect_ratios:    # 遍历每个宽高比

                # 根据面积和宽高比计算基础宽高
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height

                # 将宽高组合为张量[1, 1, 2]
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1),
                    [1, 1, 2]
                )

                # 应用缩放因子（生成三个不同尺寸的锚框）
                for scale in self.scales:
                    anchor_dims.append(scale * dims)    # 每个宽高比生成3各缩放尺寸

            # 将当前层级的所有锚框尺寸堆叠：[1, 1, 9, 2](3 radios * 3 scales)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all  #形状：5个层级 * [1, 1, 9, 2]
    
    def _get_anchors(self, feature_height, feature_width, level):
        '''获取锚框'''
        # 生成特征图网络中心点坐标（原坐标系）
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5

        # 生成网格坐标，形状[feature_height, feature_width, 2]
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]

        # 拓展维度以匹配锚框数量，形状[H, W, 1, 2]
        centers = tf.expand_dims(centers, axis=-2)

        # 复制中心点以匹配锚框数量，形状[H, W, 1, 2]
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])

        # 复制锚框尺寸到每个位置，形状[H, W, 9, 2]
        dims = tf.tile(self._anchor_dims[level - 3],    # 选择当前层级的尺寸
                       [feature_height, feature_width, 1, 1]
                       )
        
        # 合成中心点和尺寸，形状[H, W, 9, 4](4 = 中心x，y + 宽，高)
        anchors = tf.concat([centers, dims], axis=-1)

        # 输出展平为[H * W * 9, 4]
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )
    
    def get_anchors(self, image_height, image_width):
        '''输出所有层级生成的锚框'''
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height // 2 ** i),
                tf.math.ceil(image_width // 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

def random_flip_horizontal(image, boxes):
    '''函数功能：以50%的概率水平翻转图像和目标框。
    输入参数如下：
        image：图像
        boxes：目标框
    返回值如下：
        随机翻转后的图像和目标框。
    '''
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1-boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

def resize_and_pad_image(image, min_side=800.0, max_side=1333.0,
                         jitter=[640, 1024], stride=128.0):
    '''
    函数功能如下：
        1.调整图像大小，使图像的短边长度等于min_side
        2.如果图像的长边长度大于max_side，就调整图像的长边长度等于max_side
        3.如果图像的形状不能被步长stride整除，则补0
    输入如下：
        image：图像
        min_side：如果jitter被设为空，则将图像的短边长度设为这个值
        max_side：如果调整大小后图像的长度超过此值，则调整图像的大小，使长边程度等于此值。
        jitter：包含缩放抖动的最小值和最大值的列表，如果该值不为空，图像较短一边的长度将调整为该范围内的随机值
        stride：在特征金字塔上的最小特征映射的步长
    返回值如下：
        image：调整大小后的图像
        image_shape：在补0以前的图像大小
        tatio：用于调整图像大小的比例因子
    '''
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

def preprocess_data(sample):
    '''
    函数功能：对单个样本进行预处理
    输入如下：
        sample：训练样本
    返回值如下：
        image：处理后的图像
        bbox：处理后的目标框，形状为(目标框个数, 4)， 每个目标框的格式为(x, y, width, height)。
        class_id：张量，表述图像中所有目标的种类编号，形状为(目标个数)
    '''
    image = sample['image']
    bbox = swap_xy(sample['objects']['bbox'])
    class_id = tf.cast(sample['objects']['label'], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [bbox[:, 0] * image_shape[1], bbox[:, 1] * image_shape[0],
         bbox[:, 2] * image_shape[1], bbox[:, 3] * image_shape[0]],
         axis=-1
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
