from setting.config import Config as cfg
import os.path as op
import numpy as np
import tensorflow as tf
from utils.data_process_utils import box_scale_processing as bsp
import json
import cv2


class TfrecordWriter:
    def __init__(self, data_root, filename):
        self.data_root = data_root
        self.data_file = filename
        self.image_box_dict = {}
        self.image_data = None

    def write_tfrecords(self, filename, frame):
        print("=" * 10, "Write tfrecords")
        bbox_per_im = self.get_scaled_bbox_data(frame)
        with tf.io.TFRecordWriter(filename) as writer:
            for i, id in enumerate(self.image_data.keys()):
                sbbox_array = bbox_per_im[f"{id}"][0][:,:,:,:4]
                mbbox_array = bbox_per_im[f"{id}"][1][:,:,:,:4]
                lbbox_array = bbox_per_im[f"{id}"][2][:,:,:,:4]
                scate_array = bbox_per_im[f"{id}"][0][:,:,:, 4]
                mcate_array = bbox_per_im[f"{id}"][1][:,:,:, 4]
                lcate_array = bbox_per_im[f"{id}"][2][:,:,:, 4]

                category_s = self.one_hot(scate_array)
                category_m = self.one_hot(mcate_array)
                category_l = self.one_hot(lcate_array)
                print(category_s.shape)

                # print(raw_bbox.shape)
                lbbox_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[lbbox_array.tostring()]))
                mbbox_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[mbbox_array.tostring()]))
                sbbox_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sbbox_array.tostring()]))
                lcate_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[category_l.tostring()]))
                mcate_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[category_m.tostring()]))
                scate_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[category_s.tostring()]))
                img_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.image_data[f"{id}"].tostring()]))
                example_dict = {"image": img_feature, "bbox_l": lbbox_feature, "bbox_m": mbbox_feature,
                                "bbox_s": sbbox_feature, "category_l": lcate_feature, "category_m": mcate_feature,
                                "category_s": scate_feature}
                # print(f"Parsing----------data--------[{i+1} / {frame}]-----------------")
                features = tf.train.Features(feature=example_dict)
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)

    def read_json(self):
        with open(op.join(self.data_root, self.data_file), "r") as instance_json:
            data = json.load(instance_json)

        annotations = data["annotations"]
        images = data["images"]
        return annotations, images

    def get_image_size(self, images):
        size_dict = {}
        for im in images:
            id = im["id"]
            height = im["height"]
            width = im["width"]
            size_dict.update({f"{id}": [height, width]})
        return size_dict

    def anno_dict_maker(self, images):
        bbox_wrapped = {}
        for im in images:
            im_id = im['id']
            bbox_wrapped.update({f"{im_id}": []})
        return bbox_wrapped

    def make_bbox_per_im_dict(self, annotations, bbox_wrapped, images):
        size = self.get_image_size(images)
        for anno in annotations:
            bbox = anno['bbox']
            imid_anno = anno["image_id"]
            category = anno["category_id"]
            height = size[f"{imid_anno}"][0]
            width = size[f"{imid_anno}"][1]
            box_x = bbox[0]
            box_y = bbox[1]
            box_w = bbox[2]
            box_h = bbox[3]
            center_x = box_x + (box_w / 2)
            center_y = box_y + (box_h / 2)
            yolo_x = center_x / width
            yolo_y = center_y / height
            yolo_w = box_w / width
            yolo_h = box_h / height
            bbox_with_cate = [yolo_x, yolo_y, yolo_w, yolo_h, category]
            if f"{imid_anno}" in bbox_wrapped.keys():
                if len(bbox_wrapped[f"{imid_anno}"]) < cfg.MAX_BOX_PER_IMAGE:
                    bbox_wrapped[f"{imid_anno}"].append(bbox_with_cate)
                    # print(len(bbox_wrapped[f"{imid_anno}"]))
                else:
                    pass
            else:
                bbox_wrapped.update({f"{imid_anno}": bbox_with_cate})
        return bbox_wrapped

    def get_bbox_data(self, annotations, bbox_wrapped, images):
        bbox_per_im = self.make_bbox_per_im_dict(annotations, bbox_wrapped, images)
        bbox_resized_list = []
        for i in bbox_per_im.keys():
            box_array = np.array(bbox_per_im[i])
            box_resized = np.zeros((cfg.MAX_BOX_PER_IMAGE, 5), dtype=np.float64)
            box_count = len(bbox_per_im[i])
            if box_array.shape != (0,):
                box_resized[:box_count] = box_array
            bbox_resized_list.append(box_resized.tolist())
        for i, ids in enumerate(bbox_per_im):
            bbox_per_im[ids] = bbox_resized_list[i]
        return bbox_per_im

    def get_scaled_bbox_data(self, frame):
        annotations, images = self.read_json()
        bbox_wrapped = self.anno_dict_maker(images)
        bbox_per_im = self.get_bbox_data(annotations, bbox_wrapped, images)
        self.image_data = self.get_image_data(bbox_per_im, frame)
        for i, id in enumerate(self.image_data.keys()):
            bbox_array = np.array(bbox_per_im[f"{id}"])
            bbox_scaled = bsp(bbox_array)
            bbox_per_im[f"{id}"] = bbox_scaled
        return bbox_per_im

    def get_image_data(self, bbox_per_im, frame):
        if frame == "all":
            frame = len(bbox_per_im.keys())
        img_dict = {}
        for i, im_id in enumerate(bbox_per_im.keys()):
            image_file_name = [file for file in cfg.FILE_LIST if file.endswith(("000" + f"{im_id}.jpg"))]
            image_data = cv2.imread(op.join(cfg.PATH_TO_IMAGES,image_file_name[0]))
            image_data = cv2.resize(image_data, (cfg.SIZE_H, cfg.SIZE_W))
            img_dict.update({f"{im_id}": image_data})
            img_dict.update({f"{im_id}": image_data})
            print(f"Writing----------image--------[{i+1} / {frame}]-----------------")
            if i == frame-1:
                break
        return img_dict

    def one_hot(self, category):
        h, w, c = category.shape
        cate_zeros = np.zeros([h, w, c, cfg.NUM_CLASS+1])
        category = np.reshape(category, [h * w * c])
        cate_onehot = cate_zeros.copy()
        cate_onehot = np.reshape(cate_onehot, [h * w * c, cfg.NUM_CLASS+1])

        for i, idx in enumerate(category):
            if idx != 0.0:
                cate_onehot[i][int(idx)] = 1
            else:
                pass

        cate_onehot = np.reshape(cate_onehot, [h, w, c, cfg.NUM_CLASS+1])
        return cate_onehot


def main():
    writer = TfrecordWriter(cfg.DATA_ROOT, cfg.INSTANCES_VAL2017)
    writer.write_tfrecords(cfg.TFRECORD_FILENAME, frame=100)


if __name__ == "__main__":
    main()
