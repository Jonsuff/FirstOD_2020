import tensorflow as tf
import numpy as np
from setting.config import Config as cfg

def box_scale_processing(grtr):
    anchors = np.reshape(cfg.ANCHORS, [9,2])
    anchor_rate = anchors / cfg.SIZE_SQUARE
    bbox_scaled = np.zeros((cfg.BATCH_SIZE, 3, cfg.MAX_BOX_PER_IMAGE, 5))
    grtr_wh = grtr[:,2:4]
    grtr_wh_new = grtr_wh[:, tf.newaxis, :]
    # grtr_wh.shape = (NUM_BOX, 1, 2)
    # calculate : (NUM_BOX, 1, 2) vs (9, 2)
    # result : (NUM_BOX, 9, 2)
    mins = np.maximum(-grtr_wh_new / 2, -anchor_rate / 2)
    maxs = np.minimum(grtr_wh_new / 2, anchor_rate / 2)
    inter_wh = maxs-mins
    union_wh = (grtr_wh_new[:,:,0] * grtr_wh_new[:,:,1] + anchor_rate[:,0] * anchor_rate[:,1]
                - inter_wh[:,:,0] * inter_wh[:,:,1])
    iou = (inter_wh[:,:,0] * inter_wh[:,:,1]) / union_wh
    best_box_idx = tf.argmax(iou, axis=1).numpy()
    for i, idx in enumerate(best_box_idx):
        # if idx = 0, 1, 2 : group 2 -> small bbox
        # if idx = 3, 4, 5 : group 1 -> medium bbox
        # if idx = 6, 7, 8 : group 0 -> large bbox
        group = 2 - idx // 3
        bbox_scaled[:, group, i,  :2] = grtr[i, :2]
        bbox_scaled[:, group, i, 2:4] = grtr_wh[i]
        bbox_scaled[:, group, i, 4:5] = grtr[i, 4:5]
    return bbox_scaled


