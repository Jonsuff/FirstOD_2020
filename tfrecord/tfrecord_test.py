import tensorflow as tf
import tfrecords
import os


MAX_BOX_PER_IMAGE = 65
PATH_IMAGES_HOME = "C:\\Users\\JONSUFF\\Desktop\\dataset\\val2017\\"
FILE_NAME_HOME = "instances_val2017.json"
PATH_ANNOTATIONS_HOME = "C:\\Users\\JONSUFF\\Desktop\\dataset\\annotations_trainval2017\\"
TF_functions = tfrecords.TfrecordMaker(PATH_ANNOTATIONS_HOME, FILE_NAME_HOME, PATH_IMAGES_HOME)
file_list = os.listdir(PATH_IMAGES_HOME)


def test_tfrecords_io(filename, frames):
    write_tfrecords(filename, frames)
    dataset = read_tfrecords(filename, 2, 4, False)
    for i, feature in enumerate(dataset):
        image = feature["image"]
        bbox = feature["bbox"]
        print(f"{i} image: shape={image.get_shape()}, value={image[0, 0, 0:-1:50, 1].numpy()}")
        print(f"{i} bbox:  shape={bbox.get_shape()}, value={bbox[0, 0].numpy()}")


def write_tfrecords(filename, frames):
    print("="*10, "Write tfrecords")
    image_id = TF_functions.get_image_annotations(frames)
    box_info = TF_functions.get_bbox_annotations()
    matched_box = TF_functions.matching_with_im_box(image_id,box_info)
    resized_box_annotations = TF_functions.resize_box_list(MAX_BOX_PER_IMAGE, matched_box)
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(frames):
            image_name_list = TF_functions.im_ids
            image_file_name = [file for file in file_list if file.endswith(("0000"+image_name_list[i]))]
            image_bytes_list = [TF_functions.open_image(PATH_IMAGES_HOME, image_file_name[0])]
            img_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes_list[i]]))
            box_data = resized_box_annotations.tostring()
            box_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[box_data[i]]))
            example_dict = {"image": img_feature, "bbox": box_feature}
            features = tf.train.Features(feature=example_dict)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def read_tfrecords(filename, epoch, batch, shuffle):
    print("="*10, "Read tfrecords")
    dataset = tf.data.TFRecordDataset([filename])
    # each example is parsed in 'parse_example'
    dataset = dataset.map(parse_example)
    # set epoch, batch size, shuffle
    return dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)


def parse_example(example):
    feature = tf.io.FixedLenFeature((), tf.string, default_value="")
    features_dict = {"image": feature, "bbox": feature}
    # read data as string
    parsed = tf.io.parse_single_example(example, features_dict)
    decoded = dict()
    # convert bytes to original type
    decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    decoded["bbox"] = tf.io.decode_raw(parsed["bbox"], tf.int32)
    # reshape flat array to multi dimensional array
    decoded["image"] = tf.reshape(decoded["image"], shape=(256, 256, 3))
    decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(MAX_BOX_PER_IMAGE, 5))
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
    test_tfrecords_io("sample.tfrecords", 20)

