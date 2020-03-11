import numpy as np
import tensorflow as tf
import json
import os.path as op
from PIL import Image


class TfrecordMaker:
    def __init__(self, srcpath, filename_json, image_path):
        self.srcpath = srcpath
        self.filename_json = filename_json
        self.image_path = image_path
        self.im_h = None
        self.im_w = None
        self.im_ids = []
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

    #
    def open_image(self, image_path,filename_image):
        image = Image.open(op.join(image_path, filename_image), 'r')
        self.im_w, self.im_h = image.size
        return image

    def get_image_annotations(self, frame):
        for i in range(frame):
            im_id = self.image_list[i]['id']
            self.im_ids.append(im_id)
        return self.im_ids

    def get_bbox_annotations(self):
        box_total = []
        for object_annotations in self.annotations_list:
            im_id_annotations = object_annotations['image_id']
            box_x = object_annotations['bbox'][0]
            box_y = object_annotations['bbox'][1]
            box_w = object_annotations['bbox'][2]
            box_h = object_annotations['bbox'][3]
            category_id = object_annotations['category_id']
            box_list = [im_id_annotations, box_y, box_x, box_h, box_w, category_id]
            box_total.append(box_list)
        return box_total

    def matching_with_im_box(self, im_ids, box_total):
        same_id_box_list = []
        box_with_same_id_list = []
        for image_id in im_ids:
            for box_info in box_total:
                if image_id == box_info[0]:
                    del box_info[0]
                    same_id_box_list.append(box_info)
            box_with_same_id = same_id_box_list
            box_with_same_id_list.append(box_with_same_id)
            same_id_box_list = []
        return box_with_same_id_list

    def resize_box_list(self, max_box, matched_box):
        box_resized_list = []
        for box_list in matched_box:
            box_list_array = np.array(box_list,dtype=np.int32)
            box_resized = np.zeros((max_box, 5), dtype=np.int32)
            box_count = len(box_list)
            if box_list_array.shape != (0,):
                box_resized[:box_count] = box_list_array
            box_resized_list.append(box_resized.tolist())
        return box_resized_list

    def resize_image_list(self, image_data):
        image_pix_array = np.array(image_data.getdata(), dtype=np.uint8)
        reshape_array = image_pix_array.reshape(self.im_h, self.im_w, 3)
        resized_array = np.zeros((640,640,3), dtype=np.uint8)
        if reshape_array.shape != (0,):
            resized_array[:self.im_h, :self.im_w] = reshape_array
        return resized_array



