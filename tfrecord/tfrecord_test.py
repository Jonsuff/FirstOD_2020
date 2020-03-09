import tensorflow as tf
import tfrecords
import os
import numpy as np

MAX_BOX_PER_IMAGE = 65
PATH_IMAGES_HOME = "/home/rilab/workspace/cocotry/val2017"
FILE_NAME_HOME = "instances_val2017.json"
PATH_ANNOTATIONS_HOME = "/home/rilab/workspace/cocotry/annotations/"
TF_functions = tfrecords.TfrecordMaker(PATH_ANNOTATIONS_HOME, FILE_NAME_HOME, PATH_IMAGES_HOME)
file_list = os.listdir(PATH_IMAGES_HOME)



def test_tfrecords_io(filename, frames):
    write_tfrecords(filename, frames)
    dataset = read_tfrecords(filename, 1, 1, False)
    # image = dataset["image"]
    # bbox = dataset["bbox"]
    print(dataset)
    # print(image)
    for data in dataset:
        image = data["image"]
        bbox = data["bbox"]
        # print(f"{i} image: shape={image.get_shape()}, value={image[0, 0, 0:-1:50, 1].numpy()}")
        # print(f"{i} bbox:  shape={bbox.get_shape()}, value={bbox[0, 0].numpy()}")


def write_tfrecords(filename, frames):
    print("="*10, "Write tfrecords")
    image_id = TF_functions.get_image_annotations(frames)
    box_info = TF_functions.get_bbox_annotations()
    matched_box = TF_functions.matching_with_im_box(image_id,box_info)
    # print(matched_box)
    resized_box_annotations = TF_functions.resize_box_list(MAX_BOX_PER_IMAGE, matched_box)
    resized_box_annotations_flat = [items for sublist in resized_box_annotations for items in sublist]
    # print(resized_box_annotations)
    # resized_box_annotations_raw = np.array(resized_box_annotations).tostring()
    # print(len(resized_box_annotations_raw))
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(frames):
            image_name_list = TF_functions.im_ids
            image_file_name = [file for file in file_list if file.endswith(("0000"+f"{image_name_list[i]}.jpg"))]
            image_bytes_list = np.array(TF_functions.open_image(PATH_IMAGES_HOME, image_file_name[0])).tostring()
            img_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes_list]))
            print("-----------------------------------",np.array(resized_box_annotations[i]))
            # raw_box = np.array(resized_box_annotations[i]).tostring()
            # box_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_box]))
            box_feature_y = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(resized_box_annotations_flat[i][0]).tostring()))
            box_feature_x = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(resized_box_annotations_flat[i][1]).tostring()))
            box_feature_h = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(resized_box_annotations_flat[i][2]).tostring()))
            box_feature_w = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(resized_box_annotations_flat[i][3]).tostring()))
            box_feature_categoryId = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(resized_box_annotations_flat[i][4]).tostring()))
            example_dict = {"image": img_feature, "bbox_y": box_feature_y,
                            "bbox_x": box_feature_x, "bbox_h": box_feature_h, "bbox_w": box_feature_w,
                            "bbox_category": box_feature_categoryId}

            # example_dict = {"image": img_feature, "bbox": box_feature}
            features = tf.train.Features(feature=example_dict)
            example = tf.train.Example(features=features)
            # global serialized
            serialized= example.SerializeToString()
            # print(serialized)
            writer.write(serialized)




def read_tfrecords(filename, epoch, batch, shuffle):
    print("="*10, "Read tfrecords")
    dataset = tf.data.TFRecordDataset([filename])
    # each example is parsed in 'parse_example'
    print("mapping")
    dataset = dataset.map(parse_example)
    print("error?")
    # set epoch, batch size, shuffle
    return dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)
    # return dataset

def parse_example(example):
    print("parsinggggggggggggggggggggggggggggggggggggggg")
    feature = tf.io.FixedLenFeature((), tf.string, default_value="")
    feature_box = tf.io.FixedLenFeature([], tf.int64)
    features_dict = {"image": feature,
                     "bbox_y": feature_box,
                     "bbox_x": feature_box,
                     "bbox_h": feature_box,
                     "bbox_w": feature_box,
                     "bbox_category": feature_box}
    # features_dict = {"image": feature, "bbox_id": feature, "bbox_y": feature, "bbox_x": feature,
    #                 "bbox_h": feature, "bbox_w": feature, "bbox_category": feature}

    # read data as string
    parsed = tf.io.parse_single_example(example, features_dict)
    decoded = dict()
    # convert bytes to original type
    decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    decoded["bbox_y"] = tf.io.decode_raw(parsed["bbox_y"], tf.int64)
    decoded["bbox_x"] = tf.io.decode_raw(parsed["bbox_x"], tf.int64)
    decoded["bbox_h"] = tf.io.decode_raw(parsed["bbox_h"], tf.int64)
    decoded["bbox_w"] = tf.io.decode_raw(parsed["bbox_w"], tf.int64)
    decoded["bbox_category"] = tf.io.decode_raw(parsed["bbox_category"], tf.int64)

    # reshape flat array to multi dimensional array
    decoded["image"] = tf.reshape(decoded["image"], shape=(256, 256, 3))
    # print("len--------------------------------------------------",len(decoded['bbox']))
    # decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(MAX_BOX_PER_IMAGE, 5))
    decoded["bbox_y"] = tf.reshape(decoded["bbox_y"], shape=(MAX_BOX_PER_IMAGE, 1))
    decoded["bbox_x"] = tf.reshape(decoded["bbox_x"], shape=(MAX_BOX_PER_IMAGE, 1))
    decoded["bbox_h"] = tf.reshape(decoded["bbox_h"], shape=(MAX_BOX_PER_IMAGE, 1))
    decoded["bbox_w"] = tf.reshape(decoded["bbox_w"], shape=(MAX_BOX_PER_IMAGE, 1))
    decoded["bbox_category"] = tf.reshape(decoded["bbox_category"], shape=(MAX_BOX_PER_IMAGE, 1))
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
    test_tfrecords_io("sample.tfrecords", 5)

