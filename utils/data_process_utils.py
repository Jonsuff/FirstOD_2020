import tensorflow as tf
import numpy as np
from setting.config import Config as cfg

def box_processing(grtr):
    # anchor_rate.shape = (9, 2)
    anchor_rate = cfg.ANCHORS / cfg.SIZE_SQUARE
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    lbbox = np.zeros((cfg.SIZE_LBOX, cfg.SIZE_LBOX, 3, 4))
    mbbox = np.zeros((cfg.SIZE_MBOX, cfg.SIZE_MBOX, 3, 4))
    sbbox = np.zeros((cfg.SIZE_SBOX, cfg.SIZE_SBOX, 3, 4))

    bbox_real = [lbbox, mbbox, sbbox]

    grtr_wh = grtr[:,2:4]
    grtr_wh_new = grtr_wh[:, tf.newaxis, :]
    # grtr_wh.shape = (NUM_BOX, 1, 2)
    # calculate : (N, 1, 2) vs (9, 2)
    # result : (N, 9, 2)
    mins = np.maximum(-grtr_wh_new / 2, -anchor_rate / 2)
    maxs = np.minimum(grtr_wh_new / 2, anchor_rate / 2)
    inter_wh = maxs-mins
    union_wh = (grtr_wh_new[:,:,0] * grtr_wh_new[:,:,1] + anchor_rate[:,:,0] * anchor_rate[:,:,1]
                - inter_wh[:,:,0] * inter_wh[:,:,1])
    iou = (inter_wh[:,:,0] * inter_wh[:,:,1]) / union_wh
    # best_box_idx.shape = (NUM_BOX, )
    best_box_idx = tf.argmax(iou, axis=1).numpy()
    scale = [cfg.SCALE_L, cfg.SCALE_M, cfg.SCALE_S]

    for i, idx in enumerate(best_box_idx):
        # if idx = 0, 1, 2 : small bbox -> group 2
        # if idx = 3, 4, 5 : medium bbox -> group 1
        # if idx = 6, 7, 8 : large bbox -> group 0
        group = 2 - idx // 3
        grid_x = int(np.floor(grtr[i, 0] / scale[group]))
        grid_y = int(np.floor(grtr[i, 1] / scale[group]))
        index_k = anchors_mask[group].index(idx)
        bbox_real[group][grid_y, grid_x ,index_k , :2] = grtr.numpy()[i, :2]
        bbox_real[group][grid_y, grid_x ,index_k , 2:4] = grtr_wh.numpy()[i]
    return bbox_real

