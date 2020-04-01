import numpy as np
import tensorflow as tf
import yolo
import cv2
import shutil
import os


FILE_NAME = "instances_val2017.json"
PATH_ANNOTATIONS = "/home/rilab/workspace/cocotry/annotations/"
image_file = "/home/rilab/workspace/cocoDataset/val2017/000000078823.jpg"
img = cv2.imread(image_file)

height, width, channel = img.shape

size = 416
rate_w = size / width
rate_h = size / height
new_w = int(width * rate_w)
new_h = int(height * rate_h)

new_img = cv2.resize(img, (new_h,new_w))


new_img = np.float32(new_img)
new_img = np.expand_dims(new_img, axis=0)

logdir = "./data/log"
input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = yolo.YOLOv3(input_tensor)
small_box, medium_box, large_box = conv_tensors


output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = yolo.decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)


model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

