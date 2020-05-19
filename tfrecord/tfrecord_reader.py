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
        feature_dict = {"image": feature, "bbox_l": feature, "bbox_m": feature, "bbox_s": feature,
                        "category_l": feature, "category_m": feature, "category_s": feature}
        parsed = tf.io.parse_single_example(example, feature_dict)
        decoded = dict()
        decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
        decoded["bbox_l"] = tf.io.decode_raw(parsed["bbox_l"], tf.float64)
        decoded["bbox_m"] = tf.io.decode_raw(parsed["bbox_m"], tf.float64)
        decoded["bbox_s"] = tf.io.decode_raw(parsed["bbox_s"], tf.float64)
        decoded["category_l"] = tf.io.decode_raw(parsed["category_l"], tf.float64)
        decoded["category_m"] = tf.io.decode_raw(parsed["category_m"], tf.float64)
        decoded["category_s"] = tf.io.decode_raw(parsed["category_s"], tf.float64)
        decoded["image"] = tf.reshape(decoded["image"], shape=(cfg.SIZE_H, cfg.SIZE_W, 3))
        decoded["bbox_l"] = tf.reshape(decoded["bbox_l"], shape=(cfg.MAX_BOX_PER_IMAGE, 4))
        decoded["bbox_m"] = tf.reshape(decoded["bbox_m"], shape=(cfg.MAX_BOX_PER_IMAGE, 4))
        decoded["bbox_s"] = tf.reshape(decoded["bbox_s"], shape=(cfg.MAX_BOX_PER_IMAGE, 4))
        # decoded["category"] = tf.reshape(decoded["category"], shape=(cfg.MAX_BOX_PER_IMAGE, ))
        decoded["category_l"] = tf.reshape(decoded["category_l"], shape=(cfg.SIZE_LBOX, cfg.SIZE_LBOX, 3,
                                                                         cfg.NUM_CLASS+1))
        decoded["category_m"] = tf.reshape(decoded["category_m"], shape=(cfg.SIZE_MBOX, cfg.SIZE_MBOX, 3,
                                                                         cfg.NUM_CLASS+1))
        decoded["category_s"] = tf.reshape(decoded["category_s"], shape=(cfg.SIZE_SBOX, cfg.SIZE_SBOX, 3,
                                                                         cfg.NUM_CLASS+1))
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
    reader = TfrecordReader(cfg.EPOCH, cfg.BATCH_SIZE, False, cfg.TFRECORD_FILENAME)
    dataset = reader.get_dataset()
    print(dataset)
    for i in dataset:
        lcate = i["category_l"]
        lbbox = i["bbox_l"]
        print(lbbox)


if __name__ == "__main__":
    main()

