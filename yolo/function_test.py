import numpy as np
import tensorflow as tf
import common.CommonFunctions as common
import darknet_53.backbone as backbone
from setting.config import Config as cfg
import math
from utils.utils import Utils as utils

np.random.seed(1)
def data():
    bbox_gather = np.arange(507 * 4)
    bbox_gather = tf.cast(bbox_gather, dtype=tf.int32)
    bbox_gather = tf.reshape(bbox_gather, [1, 507, 4])

    bbox_iou = np.random.rand(507*4)
    bbox_iou = np.reshape(bbox_iou, [1, 507, 4])
    bbox_iou = tf.cast(bbox_iou, dtype=tf.float64)

    rand_class = np.random.rand(507*90)
    rand_class = np.reshape(rand_class, [1, 507, 90])
    rand_class = tf.cast(rand_class, dtype=tf.float64)
    rand_obj = np.random.rand(507)
    rand_obj = np.reshape(rand_obj, [1, 507, 1])
    rand_obj = tf.cast(rand_obj, dtype=tf.float64)

    bbox_pred = tf.concat([bbox_iou, rand_obj, rand_class], axis=-1)

    grtr_iou = np.random.rand(30*4)
    grtr_iou = np.reshape(grtr_iou, [1,30,4])
    grtr_iou = tf.cast(grtr_iou, dtype=tf.float64)

    grtr_cate = np.arange(30)
    cate_zeros = [0] * 90
    cate_list = []
    for i in grtr_cate:
        cate = cate_zeros.copy()
        if int(i) == 0:
            cate[int(i)] = 1
        else:
            cate[int(i) - 1] = 1
        cate_list.append(cate)
    cate_list = np.array(cate_list)
    cate_list = np.expand_dims(cate_list, axis=0)
    cate_list = tf.cast(cate_list, dtype=tf.int32)
    utility = utils(bbox_pred, grtr_iou, grtr_cate)
    return utility, cate_list, bbox_pred, bbox_iou



def gather_nd_test(bbox_pred):
    print("Predicted bbox : ",bbox_pred)
    iou = np.random.rand(1, 507, 30)
    iou = tf.cast(iou, dtype=tf.float64)
    label_indices = tf.argmax(iou, axis=1)
    label_indices = label_indices[..., tf.newaxis]
    print("Best iou indices : ",label_indices)
    bbox_best = tf.gather_nd(bbox_pred, label_indices, batch_dims=1)
    print("Best bbox data : ",bbox_best)


def iou_test(utils):
    iou = utils.get_iou()
    return iou


def giou_test(utils):
    giou = utils.get_giou()
    return giou


def bbox_loss_test(utils, iou):
    bbox_loss = utils.bbox_loss(iou)
    return bbox_loss


def category_loss_test(utils, cate_pred):
    category_loss = utils.category_loss(cate_pred)
    return category_loss


def object_loss_test(utils):
    object_loss = utils.object_loss()
    return object_loss


def sparsetest(utils, iou):
    label_indices = utils.get_best_index(iou)
    label_indices = tf.sort(label_indices, axis=1, direction="ASCENDING")
    sorted_label = tf.keras.backend.eval(label_indices)
    sorted_label = sorted_label[..., tf.newaxis]
    best_bbox = utils.get_best_bbox(bbox_pred, sorted_label)
    zeros = tf.zeros([1,30,1])
    zeros = tf.cast(zeros, dtype=tf.int64)
    sorted_label = tf.concat([zeros, sorted_label], axis=-1)
    object_mask = tf.SparseTensor(sorted_label[0], [1]*30, [1,507])
    object_mask = tf.sparse.to_dense(object_mask)


def main():
    utility, cate_list, bbox_pred, bbox_iou = data()
    iou = iou_test(utility)
    bbox_loss = bbox_loss_test(utility, iou)
    category_loss = category_loss_test(utility, cate_list)
    object_loss = object_loss_test(utility)
    print("bbox_loss : ", bbox_loss)
    print("category_loss : ", category_loss)
    print("object_loss : ", object_loss)


if __name__ == "__main__":
    main()