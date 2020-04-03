import numpy as np
import tensorflow as tf
from setting.config import Config as cfg
import cv2

class TfrecordReader:
    def __init__(self):
        pass

    def get_dataset(self, filename, epoch, batch, shuffle):
        print("=" * 10, "Read tfrecords")
        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(self.parse_example)
        # set epoch, batch size, shuffle
        return self.dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)

    def parse_example(self, example):
        feature = tf.io.FixedLenFeature([], tf.string)
        feature_dict = {"image": feature, "bbox": feature}
        # read data as string
        parsed = tf.io.parse_single_example(example, feature_dict)
        decoded = dict()
        # convert bytes to original type
        decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
        decoded["bbox"] = tf.io.decode_raw(parsed["bbox"], tf.float64)
        decoded["image"] = tf.reshape(decoded["image"], shape=(416, 416, 3))
        decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(cfg.MAX_BOX_PER_IMAGE, 5))
        # decoded["category"] = tf.reshape(decoded["category"], shape=(MAX_BOX_PER_IMAGE, 1))
        # you can do preprocess here, e.g. convert image type from uint8 (0~255) to float (0~1)
        # ...
        return decoded

    def dataset_process(self, dataset, epoch, batch, shuffle):
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size=batch, drop_remainder=True)
        return dataset

def main():
    reader = TfrecordReader()
    dataset = reader.get_dataset(cfg.TFRECORD_FILENAME, 1, 1, False)


if __name__ == "__main__":
    main()

