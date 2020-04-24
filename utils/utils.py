import numpy as np
import tensorflow as tf
import common.CommonFunctions as common
import darknet_53.backbone as backbone
from setting.config import Config as cfg
import math

class Utils():
    def __init__(self, bbox_pred, bbox_grtr, category_grtr):
        self.bbox_pred = bbox_pred
        self.bbox_grtr = bbox_grtr
        self.category_grtr = category_grtr
        self.bbox_best = None
        self.label_indices = None
        self.iou_flat = None
        self.bbox_size = None

    def get_iou(self):
        pred_bbox = self.bbox_pred[:,:,:4]
        pred_reshaped = tf.reshape(pred_bbox, (cfg.BATCH_SIZE, -1, 1, 4))
        grtr_reshaped = tf.reshape(self.bbox_grtr, (cfg.BATCH_SIZE, 1, cfg.MAX_BOX_PER_IMAGE, 4))
        bbox_raw = pred_reshaped[..., :4] * cfg.SIZE_SQUARE
        gt_raw = grtr_reshaped[..., :4] * cfg.SIZE_SQUARE
        bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
        gt_area = gt_raw[..., 2] * gt_raw[..., 3]
        bbox_xymin = bbox_raw[..., :2] - bbox_raw[..., 2:] * 0.5
        bbox_xymax = bbox_raw[..., :2] + bbox_raw[..., 2:] * 0.5
        gt_xymin = gt_raw[..., :2] - gt_raw[..., 2:] * 0.5
        gt_xymax = gt_raw[..., :2] + gt_raw[..., 2:] * 0.5

        left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
        right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])
        intersection = tf.maximum(right_bottom - left_top, 0.0)
        inter_area = intersection[..., 0] * intersection[..., 1]
        union_area = bbox_area + gt_area - inter_area
        iou = inter_area / union_area
        return iou

    def get_giou(self):
        pred_bbox = self.bbox_pred[:,:,:4]
        pred_reshaped = tf.reshape(pred_bbox, (cfg.BATCH_SIZE, -1, 1, 4))
        grtr_reshaped = tf.reshape(self.bbox_grtr, (cfg.BATCH_SIZE, 1, cfg.MAX_BOX_PER_IMAGE, 4))
        bbox_raw = pred_reshaped[..., :4] * cfg.SIZE_SQUARE
        gt_raw = grtr_reshaped[..., :4] * cfg.SIZE_SQUARE
        bbox_area = bbox_raw[..., 2] * bbox_raw[..., 3]
        gt_area = gt_raw[..., 2] * gt_raw[..., 3]
        bbox_xymin = bbox_raw[..., :2] - bbox_raw[..., 2:] * 0.5
        bbox_xymax = bbox_raw[..., :2] + bbox_raw[..., 2:] * 0.5
        gt_xymin = gt_raw[..., :2] - gt_raw[..., 2:] * 0.5
        gt_xymax = gt_raw[..., :2] + gt_raw[..., 2:] * 0.5

        left_top = tf.maximum(bbox_xymin[..., :2], gt_xymin[..., :2])
        right_bottom = tf.minimum(bbox_xymax[..., :2], gt_xymax[..., :2])

        intersection = tf.maximum(right_bottom - left_top, 0.0)
        inter_area = intersection[..., 0] * intersection[..., 1]
        union_area = bbox_area + gt_area - inter_area
        iou = inter_area / union_area

        enclose_ltop = tf.minimum(bbox_xymin[..., :2], gt_xymin[..., :2])
        enclose_rbottom = tf.maximum(bbox_xymin[..., :2], gt_xymax[..., :2])
        enclose = tf.maximum(enclose_rbottom - enclose_ltop, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - ((enclose_area - union_area) / enclose_area)
        return giou

    def get_best_index(self, data, axis):
        index = tf.argmax(data, axis=axis)
        return index

    def get_best_data(self, data, label):
        best = tf.gather_nd(data, label, batch_dims=1)
        return best

    def bbox_loss(self, iou):
        # iou.shape = (batch, h, w, 3, NUM_BOX)
        # bbox_pred.shape = (batch, h * w * 3, 95)
        # gt_bbox.shape = (batch, NUM_BOX, 4)
        batch_size, bbox_size, boxinfo = self.bbox_pred.get_shape()
        self.bbox_size = math.sqrt(bbox_size / 3)
        gt_grid = self.bbox_grtr * self.bbox_size
        gt_whgrid = gt_grid[..., 2:4]
        gt_ltop = gt_grid[..., :2] - gt_whgrid / 2
        gt_pixel = tf.concat([gt_ltop, gt_whgrid], axis=-1)

        # iou_flat.shape = (batch, h * w * 3, NUM_BOX)
        self.iou_flat = tf.reshape(iou, (cfg.BATCH_SIZE, -1, cfg.MAX_BOX_PER_IMAGE))

        # label_indices.shape = (batch, NUM_BOX)
        self.label_indices = self.get_best_index(self.iou_flat, 1)
        self.label_indices = self.label_indices[..., tf.newaxis]

        # bbox_flat.shape = (batch, h * w * 3, 95)
        # bbox_best.shape = (batch, NUM_BOX, 95)
        self.bbox_best = self.get_best_data(self.bbox_pred, self.label_indices)
        bbox_raw = self.bbox_best * self.bbox_size
        bbox_raw = bbox_raw[:,:,tf.newaxis,:]
        gt_pixel = gt_pixel[:,tf.newaxis,:,:]
        bbox_xyloss = (bbox_raw[..., :2] - gt_pixel[..., :2]) ** 2
        bbox_whloss = (tf.sqrt(bbox_raw[..., 2:4]) - tf.sqrt(gt_pixel[..., 2:4])) ** 2
        bbox_loss = tf.concat([bbox_xyloss, bbox_whloss], axis=-1)

        return bbox_loss

    def category_loss(self, category_grtr):
        # category_best.shape = (batch, NUM_BOX, NUM_CLASS)
        # category_porb.shape = category_best.shape
        # gt_cate.shape = (batch, NUM_BOX, NUM_CLASS) >> NUM_CLASS = one-hot form
        category_best = self.bbox_best[..., 5:]
        category_prob = tf.nn.sigmoid(category_best)
        category_prob = category_prob[:, :, tf.newaxis, :]
        category_grtr = category_grtr[:, tf.newaxis, :, :]
        category_loss = tf.losses.categorical_crossentropy(category_grtr, category_prob)
        return category_loss

    def object_loss(self):
        label_indices = tf.sort(self.label_indices, axis=1, direction="ASCENDING")
        sorted_label = tf.keras.backend.eval(label_indices)
        zeros = tf.zeros([1, cfg.MAX_BOX_PER_IMAGE, 1])
        zeros = tf.cast(zeros, dtype=tf.int64)
        sorted_label = tf.concat([zeros, sorted_label], axis=-1)
        object_mask = tf.SparseTensor(sorted_label[0], [1] * cfg.MAX_BOX_PER_IMAGE, [1, (int(self.bbox_size**2))*3])
        object_mask = tf.sparse.to_dense(object_mask)
        bool_object_mask = tf.cast(object_mask, dtype=tf.bool)
        object_mask = tf.cast(object_mask, dtype=tf.float64)
        iou_best_idx = self.get_best_index(self.iou_flat, -1)
        iou_best_idx = iou_best_idx[:,:, tf.newaxis]
        tensor_idx = np.arange((int(self.bbox_size**2))*3)
        tensor_idx = np.reshape(tensor_idx, [1,int((self.bbox_size**2)*3),1])
        tensor_idx = tf.cast(tensor_idx, dtype=tf.int64)
        iou_best_idx = tf.concat([tensor_idx, iou_best_idx], axis=-1)
        iou_best = self.get_best_data(self.iou_flat, iou_best_idx)
        non_obj_mask = tf.less(iou_best, cfg.MIN_IOU)
        valid_mask = tf.logical_or(bool_object_mask, non_obj_mask)
        valid_mask = tf.cast(valid_mask, dtype=tf.float64)
        bbox_obj = self.bbox_pred[:,:,4]
        bbox_obj = bbox_obj * valid_mask
        bbox_obj = tf.sigmoid(bbox_obj)
        obj_loss = tf.losses.binary_crossentropy(bbox_obj, object_mask)
        return obj_loss
