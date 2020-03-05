import tensorflow as tf
import numpy as np
import tfrecords as Tfmaker
import os.path as op
import os

MAX_BOX_PER_IMAGE = 10
TfFunc = Tfmaker.TfrecordMaker()
image_path = "/home/rilab/workspace/cocotry/val2017/"

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
    file_list = os.listdir(image_path)
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(frames):
            image_id = TfFunc.image_list[frames]['id']
            image_file_list = [file for file in file_list if file.endswith(f"000{image_id}.jpg")]
            image = TfFunc.open_image(op.join(image_path, image_file_list[0]))
            # bbox 5 columns: [category id, y, x, h, w]

            bbox = np.zeros((MAX_BOX_PER_IMAGE, 5), dtype=np.int32)
            value = np.random.randint(0, 256, (numbox, 5), dtype=np.int32)
            bbox[:numbox] = np.random.randint(0, 256, (numbox, 5), dtype=np.int32)
            print(f"{i} image:", image[0, 0:-1:50, 1])
            print(f"{i} bbox :", numbox, bbox[0])

            # convert data
            img_data = image.tostring()
            img_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data]))
            box_data = bbox.tostring()
            box_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[box_data]))
            example_dict = {"image": img_feature, "bbox": box_feature}
            features = tf.train.Features(feature=example_dict)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()

            # write to file
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

