import numpy as np
import tensorflow as tf
import yolo
import cv2
import os
import os.path as op
import tfrecords

ANCHORS = [[1.25,1.625], [2.0,3.75], [4.125,2.875]]
NUM_CLASS = 90
STRIDES = [8, 16, 32]
MAX_BOX_PER_IMAGE = 65
PATH_IMAGES_HOME = "/home/rilab/workspace/cocotry/val2017"
FILE_NAME_HOME = "instances_val2017.json"
PATH_ANNOTATIONS_HOME = "/home/rilab/workspace/cocotry/annotations/"
TF_functions = tfrecords.TfrecordMaker(PATH_ANNOTATIONS_HOME, FILE_NAME_HOME, PATH_IMAGES_HOME)
file_list = os.listdir(PATH_IMAGES_HOME)

size = 416
rate_list = []
img_list = []
box_list = []
img_id_list = []
real_box = []
new_box_list = []
resized_box_list = []
xmin_list = []
xmax_list = []
ymin_list = []
ymax_list = []
image_list = []





def test_tfrecords_io(filename):
    dataset = read_tfrecords(filename, 1, 1, False)
    for i, feature in enumerate(dataset):
        image = feature["image"]
        bbox = feature["bbox"]
        bbox = np.array(bbox)
        bbox = np.squeeze(bbox, axis=0)
        # print(type(bbox))
        box_list.append(bbox)
        id = feature["image_id"]
        id = id.numpy()
        id = int(id)
        id = str(id)
        img_id_list.append(id)
        print(image.shape)
        size = feature["size"]
        # print(f"{i+1} bbox : ", bbox)
    box_array = np.array(box_list)
    return box_array

def reshaping(array):
    array = np.reshape(array, (65,1))
    return array

def box_image(box_array):
    for box_per_img in box_array:
        box_h = reshaping(box_per_img[:,0])
        box_w = reshaping(box_per_img[:,1])
        box_ymin = reshaping(box_per_img[:,2])
        box_xmin = reshaping(box_per_img[:,3])
        box_xmax = reshaping(box_xmin + box_w)
        box_ymax = reshaping(box_ymin + box_h)
        box_per_img = np.concatenate((box_xmin, box_ymin, box_xmax, box_ymax), axis=1)
        new_box_list.append(box_per_img.tolist())
    return new_box_list




def show_box(box_list):
    while True:
        image_file_name = [file for file in file_list if file.endswith(("000" + f"{img_id_list[7]}.jpg"))]
        print(image_file_name)
        img = cv2.imread(op.join(PATH_IMAGES_HOME, image_file_name[0]))
        img = cv2.resize(img, (size, size))
        for j in range(65):
            xmin = int(box_list[7][j][0])
            xmin_list.append(xmin)
            ymin = int(box_list[7][j][1])
            ymin_list.append(ymin)
            xmax = int(box_list[7][j][2])
            xmax_list.append(xmax)
            ymax = int(box_list[7][j][3])
            ymax_list.append(ymax)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(f"testing", img)

        if cv2.waitKey(0) == 113:
            break
        cv2.destroyAllWindows()
    return img




def resizing_data(frame, box_list):
    new_array = np.array(box_list)
    for i in range(frame):
        image_file_name = [file for file in file_list if file.endswith(("000" + f"{img_id_list[i]}.jpg"))]
        img = cv2.imread(op.join(PATH_IMAGES_HOME, image_file_name[0]))
        height, width, channel = img.shape
        rate_h = size / height
        rate_w = size / width
        rate = [rate_w, rate_h, rate_w, rate_h]
        rate_list.append(rate)
        new_h = int(height * rate_h)
        new_w = int(width * rate_w)
        img_list.append(cv2.resize(img, (new_h,new_w)))
    rate_array = np.array(rate_list)
    for j in range(frame):
        resized_box = new_array[j] * rate_array[j]
        resized_box_list.append(resized_box.tolist())
    return resized_box_list
    # return img_list



def read_tfrecords(filename, epoch, batch, shuffle):
    print("="*10, "Read tfrecords")
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_example)
    # set epoch, batch size, shuffle
    return dataset_process(dataset, epoch=epoch, batch=batch, shuffle=shuffle)


def parse_example(example):
    feature = tf.io.FixedLenFeature([], tf.string)
    feature_box = tf.io.FixedLenFeature([], tf.string)
    feature_dict = {"image_id": feature, "image": feature, "size": feature, "bbox": feature_box, "category": feature}
    # read data as string
    parsed = tf.io.parse_single_example(example, feature_dict)
    decoded = dict()
    # convert bytes to original type
    decoded["image_id"] = tf.io.decode_raw(parsed["image_id"], tf.int64)
    decoded["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    decoded["size"] = tf.io.decode_raw(parsed["size"], tf.int64)
    decoded["bbox"] = tf.io.decode_raw(parsed["bbox"], tf.float64)
    decoded["category"] = tf.io.decode_raw(parsed["category"], tf.float64)
    decoded["image"] = tf.reshape(decoded["image"], shape=(416, 416, 3))
    decoded["bbox"] = tf.reshape(decoded["bbox"], shape=(MAX_BOX_PER_IMAGE, 4))
    decoded["category"] = tf.reshape(decoded["category"], shape=(MAX_BOX_PER_IMAGE, 1))
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
    count = 1
    box_array = test_tfrecords_io("sample.tfrecords")
    # print(box_array)
    new_box = box_image(box_array)
    resizing_data(8, new_box)
    show_box(resizing_data(8, new_box))


