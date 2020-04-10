import numpy as np
import tensorflow as tf
import common.CommonFunctions as common
import darknet_53.backbone as backbone
from setting.config import Config as cfg

ANCHORS = [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
print(ANCHORS[0])
NUM_CLASS = 90
STRIDES = [8, 16, 32]
def YOLOv3(input_layer):
    route_1, route_2, conv = backbone.darknet53(input_layer)
    # print(route_1)
    # print(route_2.shape)
    # print(conv.shape)

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

    # lbbox = tf.reshape(conv_lbbox, (1, cfg.SIZE_LBOX, cfg.SIZE_LBOX, 3, cfg.NUM_CLASS+5))
    # lbbox_raw = lbbox[:,:,:,:,:4]
    # mbbox = tf.reshape(conv_mbbox, (1, cfg.SIZE_MBOX, cfg.SIZE_MBOX, 3, cfg.NUM_CLASS + 5))
    # mbbox_raw = mbbox[:, :, :, :, :4]
    # sbbox = tf.reshape(conv_sbbox, (1, cfg.SIZE_SBOX, cfg.SIZE_SBOX, 3, cfg.NUM_CLASS + 5))
    # sbbox_raw = sbbox[:, :, :, :, :4]
    # print("s",conv_sbbox.shape)
    # print("m",conv_mbbox.shape)
    # print(lbbox_raw)
    # return [sbbox_raw, mbbox_raw, lbbox_raw]
    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    # print(conv_shape)
    batch_size       = conv_shape[0]
    # print(batch_size)
    output_size      = conv_shape[1]
    # print(output_size)

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    # print(conv_output.shape)
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


def bbox_iou(boxes1, boxes2):
    bbox_raw = boxes1[..., :4] * cfg.SIZE_SQUARE
    gt_raw = boxes2[..., :4] * cfg.SIZE_SQUARE
    bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
    gt_area = gt_raw[..., 2] * gt_raw[..., 3]
    bbox_xymin = boxes1[..., :2] - boxes1[..., 2:] * 0.5
    bbox_xymax = boxes1[..., :2] + boxes1[..., 2:] * 0.5
    gt_xymin = boxes2[..., :2] - boxes2[..., 2:] * 0.5
    gt_xymax = boxes2[..., :2] + boxes2[..., 2:] * 0.5

    left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
    right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])

    intersection = tf.maximum(right_bottom - left_top, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = bbox_area + gt_area - inter_area
    iou = inter_area / union_area
    return iou


def bbox_giou(boxes1, boxes2):
    bbox_raw = boxes1[..., :4] * cfg.SIZE_SQUARE
    gt_raw = boxes2[..., :4] * cfg.SIZE_SQUARE
    bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
    gt_area = gt_raw[..., 2] * gt_raw[..., 3]
    bbox_xymin = boxes1[..., :2] - boxes1[..., 2:] * 0.5
    bbox_xymax = boxes1[..., :2] + boxes1[..., 2:] * 0.5
    gt_xymin = boxes2[..., :2] - boxes2[..., 2:] * 0.5
    gt_xymax = boxes2[..., :2] + boxes2[..., 2:] * 0.5
    left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
    right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])

    intersection = tf.maximum(right_bottom - left_top, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = bbox_area + gt_area - inter_area
    iou = inter_area / union_area

    enclose_ltop = tf.minimum(bbox_xymin[..., :2], gt_xymin[..., :2])
    enclose_rbottom = tf.maximum(bbox_xymin[...,2], gt_xymax[..., :2])
    enclose = tf.maximum(enclose_rbottom - enclose_ltop, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - ((enclose_area-union_area) / enclose_area)
    return giou


def get_loss(siou, miou, liou, bbox):
    # iou.shape = (batch, size, size, 3, NUM_CLASS)
    # s/m/liou_flat.shape = (batch, size * size, NUM_CLASS
    siou_flat = tf.reshape(siou, (cfg.BATCH_SIZE, -1, cfg.MAX_BOX_PER_IMAGE))
    miou_flat = tf.reshape(miou, (cfg.BATCH_SIZE, -1, cfg.MAX_BOX_PER_IMAGE))
    liou_flat = tf.reshape(liou, (cfg.BATCH_SIZE, -1, cfg.MAX_BOX_PER_IMAGE))

    # s/m/llabel_indices.shape = (batch, NUM_CLASS)
    slabel_indices = tf.argmax(siou_flat, axis=1)
    mlabel_indices = tf.argmax(miou_flat, axis=1)
    llabel_indices = tf.argmax(liou_flat, axis=1)

    # for gathering, transpose s/m/llabel_indices
    slabel_indices = tf.transpose(slabel_indices)
    mlabel_indices = tf.transpose(mlabel_indices)
    llabel_indices = tf.transpose(llabel_indices)

    # TODO: find out how to make them into same size
    """
    Because of the batch, tf.gather_nd doesn't work...
    Try to find out how to use tf.gather_nd properly or deal with batch dimension
    """
    bbox_flat = tf.reshape(bbox, (cfg.BATCH_SIZE, -1, 4))
    slabeled_bbox = tf.gather_nd(bbox_flat, slabel_indices)
    mlabeled_bbox = tf.gather_nd(bbox_flat, mlabel_indices)
    llabeled_bbox = tf.gather_nd(bbox_flat, llabel_indices)

    sbox_loss = (slabeled_bbox - bbox[:, :4]) ** 2
    mbox_loss = (mlabeled_bbox - bbox[:, :4]) ** 2
    lbox_loss = (llabeled_bbox - bbox[:, :4]) ** 2

    return sbox_loss, mbox_loss, lbox_loss

