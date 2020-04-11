import os

class Config:
    DATA_ROOT = "/home/jon/data/annotations/"
    INSTANCES_VAL2017 = "instances_val2017.json"
    PATH_TO_IMAGES = "/home/jon/data/val2017/"
    FILE_LIST = os.listdir(PATH_TO_IMAGES)
    TFRECORD_FILENAME = "Input_data.tfrecords"
    SIZE_W = 416
    SIZE_H = 416
    SIZE_SQUARE = 416
    SCALE_S = 1/8
    SCALE_M = 1/16
    SCALE_L = 1/32
    ANCHORS = [1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875,
               6.1875, 11.65625, 10.1875]
    NUM_CLASS = 90
    MAX_BOX_PER_IMAGE = 65

    SIZE_LBOX = 13
    SIZE_MBOX = 26
    SIZE_SBOX = 52
    BATCH_SIZE = 1
    EPOCH = 1