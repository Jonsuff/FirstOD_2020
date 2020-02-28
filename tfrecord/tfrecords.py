import numpy as np
import tensorflow as tf
import json
import os.path as op
import io
from PIL import Image

tfrec_name = 'instances_val2017.tfrecords'


class TfrecordMaker:
    def __init__(self, srcpath,filename):
        self.srcpath = srcpath
        self.filename = filename
        self.image = None
        self.category_id = None
        self.category_ids = []
        self.box_total = []
        self.im_h = None
        self.im_hs = []
        self.im_w = None
        self.im_ws = []
        self.data_file = open(op.join(self.srcpath, self.filename))
        self.data = json.load(self.data_file)
        self.annotations_list = self.data['annotations']
        self.image_list = self.data['images']
        self.feature_dict = {}


    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def create_tf_example(self):
        with tf.io.gfile.GFile(self.srcpath, 'rb') as fid:
            encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            self.image = Image.open(encoded_jpg_io)

        for object_images in self.image_list:
            self.im_h = object_images['height']
            self.im_hs.append(self.im_h)
            self.im_w = object_images['width']
            self.im_ws.append(self.im_w)

        for object_annotations in self.annotations_list:
            [self.box_x, self.box_y, self.box_w, self.box_h] = list(object_annotations['bbox'])
            self.box_total.append((self.box_x, self.box_y, self.box_w, self.box_h))
            self.category_id = int(object_annotations['category_id'])
            self.category_ids.append(self.category_id)

        self.feature_dict = {
            'image height':
                self._int64_feature(self.im_hs),
            'image width':
                self._int64_feature(self.im_ws),
            'bbox info':
                self._float_feature(self.box_total),
            'category id':
                self._int64_feature(self.category_ids),
            }
