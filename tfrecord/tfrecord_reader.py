import tensorflow as tf
from setting.config import Config as cfg


class TfrecordReader:
    def __init__(self, epoch, batch, shuffle, filename):
        self.epoch = epoch
        self.batch = batch
        self.shuffle = shuffle
        self.filename = filename

    def get_dataset(self):
        # print("=" * 10, "Read tfrecords")
        epoch = self.epoch
        batch = self.batch
        shuffle = self.shuffle
        dataset = tf.data.TFRecordDataset([self.filename])
        dataset = dataset.map(self.parse_example)
        # set epoch, batch size, shuffle
        return self.dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)

    def parse_example(self, example):
        feature = tf.io.FixedLenFeature([], tf.string)
        feature_dict = {"image": feature, "bbox": feature, "category": feature}
        parsed = tf.io.parse_single_example(example, feature_dict)
        decoded = dict()
        decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
        decoded["bbox"] = tf.io.decode_raw(parsed["bbox"], tf.float64)
        decoded["category"] = tf.io.decode_raw(parsed["category"], tf.float64)
        decoded["image"] = tf.reshape(decoded["image"], shape=(cfg.SIZE_H, cfg.SIZE_W, 3))
        decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(cfg.MAX_BOX_PER_IMAGE, 4))
        decoded["category"] = tf.reshape(decoded["category"], shape=(cfg.MAX_BOX_PER_IMAGE,))
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
    count = 0
    reader = TfrecordReader(cfg.EPOCH, cfg.BATCH_SIZE, False, cfg.TFRECORD_FILENAME)
    dataset = reader.get_dataset()
    print(dataset)
    for i in dataset:
        bbox = i["bbox"]
        print(bbox.numpy())
        count += 1
        print(count)


if __name__ == "__main__":
    main()

