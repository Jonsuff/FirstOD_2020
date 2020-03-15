# import tensorflow as tf
import numpy as np
import os.path as op
# import tfrecords
import os
import json
from PIL import Image

#
#
# MAX_BOX_PER_IMAGE = 65
PATH_ANNOTATIONS_HOME = "C:\\Users\\JONSUFF\\Desktop\\dataset\\val2017\\annotations"
PATH_IMAGES_HOME = "C:\\Users\\JONSUFF\\Desktop\\dataset\\val2017\\val2017"
FILE_NAME_HOME = "instances_val2017.json"
file_list = os.listdir(PATH_IMAGES_HOME)

data_file = open(op.join(PATH_ANNOTATIONS_HOME, FILE_NAME_HOME))
data = json.load(data_file)
annotations = data["annotations"]
image_info = data['images']
annotations_dict = {}
image_data = 1
data_dict = {"id": [], "bbox": []}
count = 0
bbox_list = []
id_list = []
################{id: box}##############################
for i in annotations:
    im_id = i['image_id']
    cate = i["category_id"]
    annotations_dict.update({f"{im_id}": []})

box_with_same_id = []
for anno in annotations:
    im_id = anno['image_id']
    id_list.append(im_id)
    bbox = anno['bbox']
    cate = anno['category_id']
    bbox.append(cate)
    if f"{im_id}" in annotations_dict.keys():
        annotations_dict[f"{im_id}"].append(bbox)
    else:
        annotations_dict.update({f"{im_id}": bbox})

box_resized = []
box_array = []
box_resized_list = []
for i in annotations_dict.keys():
    box_array = np.array(annotations_dict[i])
    box_resized = np.zeros((65,5), dtype=np.int32)
    box_count = len(annotations_dict[i])
    if box_array.shape != (0,):
        box_resized[:box_count] = box_array
    box_resized_list.append(box_resized.tolist())


for i,ids in enumerate(annotations_dict):
    annotations_dict[ids] = box_resized_list[i]


augmented_image_info = []
for iminfo in image_info:
    img_id = iminfo['id']
    if f"{img_id}" not in annotations_dict.keys():
        annotations_dict[f"{img_id}"] = np.zeros((65,5),dtype=np.int32)
    else:
        newinfo = {iminfo['id']: {'imsize': [iminfo['height'], iminfo['width']], 'bbox': annotations_dict[f"{img_id}"]}}
        augmented_image_info.append(newinfo)
