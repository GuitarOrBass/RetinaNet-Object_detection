'''数据集编码'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pretreatment as pre
import tensorflow_datasets as tfds

class LabelEncoder:
    '''
    功能：将数据集原始标签转换为训练所需的数据
    属性如下：
    anchor_box：锚点生成器
    box_variance:锚点框的放大系数
    '''
    def __init__(self):
        self._anchor_box = pre.AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
    
    def _match_anchor_boxes(
            self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        '''函数功能：根据交并比将目标框映射到每个锚点上
            1.通过compute_iou函数计算iou_matrix， 形状为（锚点个数，目标框个数）
            2.计算iou_matrix每行中最大的交并比max_iou，形状为（锚点个数）
            3.计算iou_matrix每行中最大的交并比对应的索引matched_gt_idx， 形状为（锚点个数）
            4.max_iou中大于0.5的值置1，放在positive_mask中，形状为（锚点个数）
            5.max_iou中小于0.5的值置1，放在negative_mask中，形状为（锚点个数）
            6.max_iou中大于0.4且小于0.5的值置1，放在igore_mask中，形状为（锚点个数）
            函数输入如下：
                anchor_boxes：锚点框
                gt_boxes：目标框
                match_iou：正样本交并比阈值，如果锚点框与目标框的交并比超过该阈值，则判定为正样本。
                ignore_iou：忽略样本交并比阈值，锚点框与目标框的交并比在ignore_iou和match_iou之间，则判定
                为忽略样本，如果锚点框与目标框的交并比小于ignore_iou则判定为负样本。
            函数输出如下：
                matched_gt_idx：锚点所属的目标种类，形状为(目标个数)
                positive_mask：锚点是否属于正样本，形状为(锚点个数)，如果锚点为正样本则为1
                ignore_mask：锚点是否属于忽略样本，形状为(锚点个数)，如果毛带你为忽略样本则为1，否则为0.
            '''
        # 构建交并比矩阵
        iou_matrix = pre.compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )
    
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        '''DISTRIBUETEXT'''
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ], axis=-1
        )
        box_target = box_target / self._box_variance

        return box_target
    
    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        '''DISTRIBUETEXT'''
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)

        return label
    
    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        '''函数功能：按批创建用于训练的图像、目标的修正值和种类编号'''
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)

        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        
        batch_images = tf.keras.applications.resent.preprocess_input(batch_images)

        return batch_images, labels.stack()


        