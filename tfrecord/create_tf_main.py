import numpy as np
import tensorflow as tf
import json
import os.path as op
import io
from PIL import Image
import tfrecords
import os



def convert_to_tfrecords():
    src_path = "/home/rilab/workspace/cocotry/annotations/"
    filename_json = "instances_val2017.json"
    img_path = "/home/rilab/workspace/cocotry/val2017/"
    Tfmaker = tfrecords.TfrecordMaker(src_path, filename_json, img_path)
    file_list = os.listdir(img_path)
    Tfmaker.get_image_annotations()
    box_annotations = Tfmaker.matching_with_im_box(Tfmaker.im_ids, Tfmaker.get_bbox_annotations())
    print(box_annotations)
    print(len(box_annotations))




convert_to_tfrecords()
