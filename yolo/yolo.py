import numpy as np
import tensorflow as tf
import common.CommonFunctions as common
import darknet_53.backbone as backbone
from setting.config import Config as cfg

ANCHORS = [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
NUM_CLASS = 90
STRIDES = [8, 16, 32]
def YOLOv3(input_layer):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(cfg.NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)


    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(cfg.NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(cfg.NUM_CLASS +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def get_iou(bbox_pred, bbox_grtr):
    pred_reshaped = tf.reshape(bbox_pred, (cfg.BATCH_SIZE, -1, 1, 4))
    grtr_reshaped = tf.reshape(bbox_grtr, (cfg.BATCH_SIZE, 1, cfg.MAX_BOX_PER_IMAGE, 4))
    bbox_raw = pred_reshaped[..., :4] * cfg.SIZE_SQUARE
    gt_raw = grtr_reshaped[..., :4] * cfg.SIZE_SQUARE
    bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
    gt_area = gt_raw[..., 2] * gt_raw[..., 3]
    bbox_xymin = pred_reshaped[..., :2] - pred_reshaped[..., 2:] * 0.5
    bbox_xymax = pred_reshaped[..., :2] + pred_reshaped[..., 2:] * 0.5
    gt_xymin = grtr_reshaped[..., :2] - grtr_reshaped[..., 2:] * 0.5
    gt_xymax = grtr_reshaped[..., :2] + grtr_reshaped[..., 2:] * 0.5

    left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
    right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])

    intersection = tf.maximum(right_bottom - left_top, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = bbox_area + gt_area - inter_area
    iou = inter_area / union_area
    return iou[..., 1:]


def get_giou(bbox_pred, bbox_grtr):
    pred_reshaped = tf.reshape(bbox_pred, (cfg.BATCH_SIZE, -1, 1, 4))
    grtr_reshaped = tf.reshape(bbox_grtr, (cfg.BATCH_SIZE, 1, cfg.MAX_BOX_PER_IMAGE, 4))
    bbox_raw = pred_reshaped[..., :4] * cfg.SIZE_SQUARE
    gt_raw = grtr_reshaped[..., :4] * cfg.SIZE_SQUARE
    bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
    gt_area = gt_raw[..., 2] * gt_raw[..., 3]
    bbox_xymin = pred_reshaped[..., :2] - pred_reshaped[..., 2:] * 0.5
    bbox_xymax = pred_reshaped[..., :2] + pred_reshaped[..., 2:] * 0.5
    gt_xymin = grtr_reshaped[..., :2] - grtr_reshaped[..., 2:] * 0.5
    gt_xymax = grtr_reshaped[..., :2] + grtr_reshaped[..., 2:] * 0.5

    left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
    right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])

    intersection = tf.maximum(right_bottom - left_top, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = bbox_area + gt_area - inter_area
    iou = inter_area / union_area

    enclose_ltop = tf.minimum(bbox_xymin[..., :2], gt_xymin[..., :2])
    enclose_rbottom = tf.maximum(bbox_xymin[..., 2], gt_xymax[..., :2])
    enclose = tf.maximum(enclose_rbottom - enclose_ltop, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - ((enclose_area - union_area) / enclose_area)
    return giou


def get_loss(iou, bbox_pred ,gt_box):
    # iou.shape = (batch, h * w * 3, NUM_BOX)
    iou_flat = tf.reshape(iou, (cfg.BATCH_SIZE, -1, cfg.MAX_BOX_PER_IMAGE))

    # label_indices.shape = (batch, NUM_BOX)
    label_indices = tf.argmax(iou_flat, axis=1)

    # bbox_flat.shape = (batch, h * w * 3, 4)
    # bbox_best.shape = (batch, NUM_BOX, 4)
    bbox_flat = tf.reshape(bbox_pred, (cfg.BATCH_SIZE, -1, 4))
    bbox_best = tf.gather_nd(bbox_flat, label_indices, batch_dims=1)

    bbox_loss = (bbox_best - gt_box) ** 2
    return bbox_loss

