'''模型构建函数'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_backbone():
    backbone = keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=[None,None,3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output 
        for layer_name in [
            'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'
            ]
    ]
    return keras.model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

class FeaturePyramid(keras.layers.layer):
    '''功能：构建特征金字塔网络
    属性如下：
        num_classes:目标种类个数
        backbone:构建特征金字塔网络所需的前置网络，这里指ResNet50.'''
    
    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name='FeaturePyramid', **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, 'same')
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, 'same')
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, 'same')
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, 'same')
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, 'same')
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, 'same')
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, 'same')
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, 'same')
        self.upsample_2x = keras.layers.UpSampling2D(2)
    
    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    '''构建分类子网络和回归子网络
    属性如下：
        output_filters:最后一层的输出数
        bias_init:最后一层的bias_init
    返回结果如下：
        根据不同的output_filters返回分类自子网络和回归子网络'''
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandonNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=kernel_init)
        )
        head.add(keras.layers.RELU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding='same',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head
    
class RetinaNet(keras.Model):
    '''功能：构建RetinaNet
    属性如下：
        num_classes:目标种类个数
        backbone：构建特征金字塔网络的前置resnet50
    '''
    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name='RetinaNet', **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1-0.01)/0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, 'zeros')

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N,-1,4]))
            cls_outputs.append(self.cls_head(feature), [N,-1,self.num_classes])
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)

