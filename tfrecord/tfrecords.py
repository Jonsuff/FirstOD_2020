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
        self.size_list = []
        self.shrink_rate_w = 1
        self.shrink_rate_w_list = []
        self.shrink_rate_h = 1
        self.shrink_rate_h_list = []
        self.im_ids = []
        self.box_resized_list = []
        self.data_file = open(op.join(self.srcpath, self.filename_json))
        self.data = json.load(self.data_file)
        self.annotations_list = self.data['annotations']
        self.annotations_dict = {}
        self.image_list = self.data['images']



    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def anno_dict_maker(self):
        for i in self.annotations_list:
            im_id = i['image_id']
            self.annotations_dict.update({f"{im_id}": [], "category_list": []})
        return self.annotations_dict

    def get_anno_dict(self, annotations_dict):
        for anno in self.annotations_list:
            im_id = anno['image_id']
            bbox = anno['bbox']
            box_x = bbox[0]
            box_y = bbox[1]
            box_w = bbox[2]
            box_h = bbox[3]
            ordered_bbox = [box_x, box_y, box_w, box_h]
            cate = anno['category_id']
            ordered_bbox.append(cate)
            if f"{im_id}" in annotations_dict.keys():
                annotations_dict[f"{im_id}"].append(ordered_bbox)
            else:
                annotations_dict.update({f"{im_id}": ordered_bbox})
        return annotations_dict

    # def get_anno_dict_yolo(self, annotations_dict):
    #     for anno in self.annotations_list:
    #         im_id = anno['image_id']
    #         bbox = anno['bbox']
    #         box_x = bbox[0]
    #         center_x =
    #         box_y = bbox[1]
    #         box_w = bbox[2]
    #         box_h = bbox[3]
    #         ordered_bbox = [box_h, box_w, box_y, box_x]
    #         cate = anno['category_id']
    #         ordered_bbox.append(cate)
    #         if f"{im_id}" in annotations_dict.keys():
    #             annotations_dict[f"{im_id}"].append(ordered_bbox)
    #         else:
    #             annotations_dict.update({f"{im_id}": ordered_bbox})
    #     return annotations_dict

    def resizing_bbox(self, annotations_dict):
        for i in annotations_dict.keys():
            box_array = np.array(annotations_dict[i])
            box_resized = np.zeros((65, 5), dtype=np.float64)
            box_count = len(annotations_dict[i])
            if box_array.shape != (0,):
                box_resized[:box_count] = box_array
            self.box_resized_list.append(box_resized.tolist())
        for i, ids in enumerate(annotations_dict):
            annotations_dict[ids] = self.box_resized_list[i]
        return annotations_dict

    def augmenting_info(self, annotations_dict):
        augmented_info = {}
        for iminfo in self.image_list:
            img_id = iminfo['id']
            if f"{img_id}" not in annotations_dict.keys():
                annotations_dict[f"{img_id}"] = np.zeros((65, 5), dtype=np.int32)
            else:
                newinfo = {iminfo['id']: {'imsize': [iminfo['height'], iminfo['width']],
                                          'bbox': annotations_dict[f"{img_id}"]}}
                augmented_info.update(newinfo)
        return augmented_info

    # def open_image(self, image_path, filename_image):
    #     image = Image.open(op.join(image_path, filename_image))
    #     self.im_w, self.im_h = image.size
    #     self.shrink_rate_w, self.shrink_rate_h = 224 / self.im_w, 224 / self.im_h
    #     return image

    def open_resize_image(self, image_path, filename_image):
        image = Image.open(op.join(image_path, filename_image))
        self.im_w, self.im_h = image.size
        self.shrink_rate_w, self.shrink_rate_h = 416 / self.im_w, 416 / self.im_h
        self.shrink_rate_w_list.append(self.shrink_rate_w)
        self.shrink_rate_h_list.append(self.shrink_rate_h)
        self.size_list.append([self.im_w, self.im_h])
        # print(self.size_list)
        im_resized = image.resize((416, 416))
        im_array = np.array(im_resized)
        if image.mode != "RGB":
            np.expand_dims(im_array, axis=0)
            im_array.resize(3,416,416)
        return im_array

    def get_usable_image_ids(self):
        for anno in self.annotations_list:
            image_id = anno['image_id']
            if image_id in self.im_ids:
                pass
            else:
                self.im_ids.append(image_id)
        return self.im_ids

    def get_image_annotations(self, frame):
        usable_ids = []
        self.get_usable_image_ids()
        for i in range(frame):
            usable_ids.append(self.im_ids[i])
        return usable_ids

