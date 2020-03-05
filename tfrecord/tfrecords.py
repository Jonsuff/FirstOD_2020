import numpy as np
import tensorflow as tf
import json
import os.path as op
import io
from PIL import Image




class TfrecordMaker:
    def __init__(self, srcpath,filename_json, image_path):
        self.srcpath = srcpath
        self.filename_json = filename_json
        self.image_path = image_path
        self.image = None
        self.image_raw = None
        self.category_id = None
        self.category_ids = []
        self.box_x = None
        self.box_y = None
        self.box_w = None
        self.box_h = None
        self.box_total = []
        self.box_with_same_id_list = []
        self.box_resized = []
        self.box_resized_list = []
        self.im_h = None
        self.im_hs = []
        self.im_w = None
        self.im_ws = []
        self.im_id = None
        self.im_ids = []
        self.im_id_annotations = None
        self.data_file = open(op.join(self.srcpath, self.filename_json))
        self.data = json.load(self.data_file)
        self.annotations_list = self.data['annotations']
        self.image_list = self.data['images']


    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def open_image(self, filename_image):
        self.image = np.array(Image.open(filename_image))
        self.image_raw = self.image.tostring()

    def get_image_annotations(self):
        for object_images in self.image_list:
            self.im_h = object_images['height']
            self.im_hs.append(self.im_h)
            self.im_w = object_images['width']
            self.im_ws.append(self.im_w)
            self.im_id = object_images['id']
            self.im_ids.append(self.im_id)

    def get_bbox_annotations(self, annotations_list, frame):
        # for object_annotations in self.annotations_list:
        self.im_id_annotations = annotations_list[frame]['image_id']
        self.box_x = annotations_list[frame]['bbox'][0]
        self.box_y = annotations_list[frame]['bbox'][1]
        self.box_w = annotations_list[frame]['bbox'][2]
        self.box_h = annotations_list[frame]['bbox'][3]
        self.category_id = annotations_list[frame]['category_id']
        box_list = [self.im_id,self.box_y, self.box_x, self.box_h, self.box_w, self.category_id]
        self.box_total.append(box_list)
        return self.box_total

    def matching_with_im_box(self,im_ids, box_total):
        same_id_box_list = []
        for image_id in im_ids:
            for box_info in box_total:
                if image_id == box_info[0]:
                    del box_info[0]
                    same_id_box_list.append(box_info)
            box_with_same_id = same_id_box_list
            self.box_with_same_id_list.append(box_with_same_id)
            same_id_box_list = []
        return self.box_with_same_id_list

    def resize_box_list(self):
        MAX_BOX_PER_IMAGE = 65
        for box_list in self.box_with_same_id_list:
            box_list_array = np.array(box_list)
            self.box_resized = np.zeros((MAX_BOX_PER_IMAGE, 5), dtype=np.int32)
            box_count = len(box_list)
            if box_list_array.shape != (0,):
                self.box_resized[:box_count] = box_list_array
            self.box_resized_list.append(self.box_resized.tolist())





