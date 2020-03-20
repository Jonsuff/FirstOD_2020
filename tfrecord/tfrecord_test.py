import tensorflow as tf
import tfrecords
import os
import numpy as np
from PIL import Image


MAX_BOX_PER_IMAGE = 65
PATH_IMAGES_HOME = "/home/rilab/workspace/cocotry/val2017"
FILE_NAME_HOME = "instances_val2017.json"
PATH_ANNOTATIONS_HOME = "/home/rilab/workspace/cocotry/annotations/"
TF_functions = tfrecords.TfrecordMaker(PATH_ANNOTATIONS_HOME, FILE_NAME_HOME, PATH_IMAGES_HOME)
file_list = os.listdir(PATH_IMAGES_HOME)


def test_tfrecords_io(filename,frames):
    write_tfrecords(filename, frames)
    dataset = read_tfrecords(filename, 1, 1, False)
    print(dataset)
    for i, feature in enumerate(dataset):
        image = feature["image"]
        bbox = feature["bbox"]
        print(f"{i} image: shape={image.get_shape()}, value={image.numpy()}")
        print(f"{i} bbox:  shape={bbox.get_shape()}, value={bbox[0, 0].numpy()}")
        # save test image
        rs = image.numpy().reshape(224, 224, 3)
        result = Image.fromarray(rs)
        result.save(f'test{i}.jpeg')


def write_tfrecords(filename, frames):
    print("="*10, "Write tfrecords")
    anno_dict = TF_functions.get_anno_dict(TF_functions.anno_dict_maker())
    box_resized_anno = TF_functions.resizing_bbox(anno_dict)
    image_name_list = TF_functions.get_image_annotations(frames)
    box_resized_array = np.array(TF_functions.box_resized_list)
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(frames):
            image_file_name = [file for file in file_list if file.endswith(("0000"+f"{image_name_list[i]}.jpg"))]
            open_image = TF_functions.open_resize_image(PATH_IMAGES_HOME, image_file_name[0])
            print(open_image.shape)
            image_data = open_image.tostring()
            img_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
            box_info = box_resized_array[i][:, :-1]
            category_id = box_resized_array[i][:, -1]
            raw_box = box_info.tostring()
            raw_category = category_id.tostring()
            box_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_box]))
            category_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_category]))
            example_dict = {"image": img_feature, "bbox": box_feature, "category": category_feature}
            features = tf.train.Features(feature=example_dict)
            example = tf.train.Example(features=features)
            serialized= example.SerializeToString()
            writer.write(serialized)


def read_tfrecords(filename, epoch, batch, shuffle):
    print("="*10, "Read tfrecords")
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_example)
    # set epoch, batch size, shuffle
    return dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)


def parse_example(example):
    print("parsinggggggggggggggggggggggggggggggggggggggg")
    feature = tf.io.FixedLenFeature([], tf.string)
    feature_box = tf.io.FixedLenFeature([], tf.string)
    # feature_dict = {"image": feature, "bbox": feature_box}
    feature_dict = {"image": feature, "bbox": feature_box, "category": feature}
    # read data as string
    parsed = tf.io.parse_single_example(example, feature_dict)
    decoded = dict()
    # convert bytes to original type
    decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    decoded["bbox"] = tf.io.decode_raw(parsed["bbox"], tf.float64)
    decoded["category"] = tf.io.decode_raw(parsed["category"], tf.float64)
    decoded["image"] = tf.reshape(decoded["image"], shape=(224, 224, 3))
    decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(MAX_BOX_PER_IMAGE, 4))
    decoded["category"] = tf.reshape(decoded["category"], shape=(MAX_BOX_PER_IMAGE, 1))
    # you can do preprocess here, e.g. convert image type from uint8 (0~255) to float (0~1)
    # ...
    return decoded


def dataset_process(dataset, epoch, batch, shuffle):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size=batch, drop_remainder=True)
    return dataset


if __name__ == "__main__":
    test_tfrecords_io("sample.tfrecords",5)
