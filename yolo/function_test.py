import numpy as np
import tensorflow as tf
import common.CommonFunctions as common
import darknet_53.backbone as backbone
from setting.config import Config as cfg
import math
from utils.utils import Utils as util

np.random.seed(1)


def iou_test():
    # preparing data
    utility = util()
    bbox1 = [0.4,0.4,0.8,0.8]
    bbox2 = [0.5,0.5,0.8,0.8]
    bbox3 = [0.6,0.6,0.8,0.8]
    for i in range(91):
        bbox1.append(0)
        bbox2.append(0)
        bbox3.append(0)
    bbox_pred = [bbox1, bbox2, bbox3]*169
    bbox_pred = np.array(bbox_pred)
    bbox_pred = np.reshape(bbox_pred, [1, 507, 95])
    bbox_pred = tf.cast(bbox_pred, dtype=tf.float64)
    bbox_grtr = [[0.6, 0.6, 0.8, 0.8]] * 30
    bbox_grtr = np.array(bbox_grtr)
    bbox_grtr = np.reshape(bbox_grtr, [1, 30, 4])
    bbox_grtr = tf.cast(bbox_grtr, dtype=tf.float64)

    # iou calculation by pre-made function
    iou = utility.get_iou(bbox_pred, bbox_grtr)

    # iou calculation manually with known answer
    manually_intersection1 = (416*0.6)**2
    manually_intersection2 = (416*0.7)**2
    manually_union1 = ((416*0.8) ** 2)*2 - manually_intersection1
    manually_union2 = ((416*0.8) ** 2)*2 - manually_intersection2
    manually_iou1 = manually_intersection1 / manually_union1
    manually_iou2 = manually_intersection2 / manually_union2
    manually_iou3 = 1
    manually_calculated = [[manually_iou1]*30, [manually_iou2]*30, [manually_iou3]*30] * 169
    manually_calculated = np.array(manually_calculated)
    manually_calculated = tf.cast(manually_calculated, dtype=tf.float64)
    manually_calculated = tf.reshape(manually_calculated, [1, 507, 30])

    assert np.isclose(iou.numpy(), manually_calculated.numpy()).all()
    print("Function(calculating iou) testing process is over. No errors occurred")


def bbox_loss_test():
    utility = util()
    # preparing data
    bbox1 = [0.3, 0.3, 0.6, 0.6]
    bbox2 = [0.4, 0.4, 0.6, 0.6]
    bbox3 = [0.6, 0.6, 0.8, 0.8]
    for i in range(91):
        bbox1.append(0)
        bbox2.append(0)
        bbox3.append(0)
    bbox_pred = [bbox1, bbox2, bbox3]*169
    bbox_pred = np.reshape(np.array(bbox_pred), [1, 507, 95])
    bbox_pred = tf.cast(bbox_pred, dtype=tf.float64)
    bbox_grtr = [[0.6, 0.6, 0.8, 0.8]]*30
    bbox_grtr = np.array(bbox_grtr)
    bbox_grtr = tf.cast(bbox_grtr, dtype=tf.float64)
    bbox_grtr = tf.reshape(bbox_grtr, [1, 30, 4])

    iou = np.zeros([1, 507, 30])
    certain_idx = np.arange(30)
    np.random.shuffle(certain_idx)
    for i, idx in enumerate(certain_idx):
        iou.tolist()
        iou[0][idx][i] = 1
    iou = tf.cast(iou, dtype=tf.int32)

    # bbox loss calculation by pre-made function
    bbox_loss = utility.bbox_loss(bbox_pred, bbox_grtr, iou)

    #  preparing manually calculated data
    bbox_list = bbox_pred[...,:4].numpy().tolist()
    manually_bbox = []
    for idx_manual in certain_idx:
        manually_bbox.append(bbox_list[0][idx_manual])
    manually_bbox = tf.cast(np.array(manually_bbox), dtype=tf.float64)
    manually_bbox = tf.reshape(manually_bbox, [1, 30, 4])
    manually_bbox_wh = manually_bbox[:,:,2:4]
    manually_bbox_xymin = manually_bbox[:,:,:2] - manually_bbox_wh/2
    manually_bbox_raw = tf.concat([manually_bbox_xymin, manually_bbox_wh], axis=-1)
    manually_bbox_raw = manually_bbox_raw * 13
    manually_grtr_wh = bbox_grtr[:,:,2:4]
    manually_grtr_xymin = bbox_grtr[:,:,:2] - manually_grtr_wh/2
    manually_grtr_raw = tf.concat([manually_grtr_xymin, manually_grtr_wh], axis=-1)
    manually_grtr_raw = manually_grtr_raw * 13
    manually_bbox_raw = manually_bbox_raw[:,:,tf.newaxis,:]
    manually_grtr_raw = manually_grtr_raw[:,tf.newaxis,:,:]
    xyloss = (manually_bbox_raw[...,:2] - manually_grtr_raw[...,:2]) **2
    whloss = (tf.sqrt(manually_bbox_raw[...,2:4]) - tf.sqrt(manually_grtr_raw[..., 2:4])) ** 2

    # bbox loss calculation manually with known answer
    manually_loss = tf.concat([xyloss, whloss], axis=-1)

    assert np.isclose(bbox_loss.numpy(), manually_loss.numpy()).all()
    print("Function(calculating bbox loss) testing process is over. No errors occurred")


def gather_nd_test():
    # preparing data
    bbox_iou = np.random.rand(507 * 4)
    bbox_iou = np.reshape(bbox_iou, [1, 507, 4])
    bbox_iou = tf.cast(bbox_iou, dtype=tf.float64)
    rand_class = np.random.rand(507 * 90)
    rand_class = np.reshape(rand_class, [1, 507, 90])
    rand_class = tf.cast(rand_class, dtype=tf.float64)
    rand_obj = np.random.rand(507)
    rand_obj = np.reshape(rand_obj, [1, 507, 1])
    rand_obj = tf.cast(rand_obj, dtype=tf.float64)
    bbox_pred = tf.concat([bbox_iou, rand_obj, rand_class], axis=-1)

    print("Predicted bbox : ",bbox_pred)
    iou = np.random.rand(1, 507, 30)
    iou = tf.cast(iou, dtype=tf.float64)
    label_indices = tf.argmax(iou, axis=1)
    label_indices = label_indices[..., tf.newaxis]
    print("Best iou indices : ",label_indices)
    bbox_best = tf.gather_nd(bbox_pred, label_indices, batch_dims=1)
    print("Best bbox data : ",bbox_best)


def main():
    gather_nd_test()
    iou_test()
    bbox_loss_test()

    
if __name__ == "__main__":
    main()