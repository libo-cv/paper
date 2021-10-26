from get_features import *
import numpy as np
import tensorflow as tf
import keras

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_label_img_dict(img_address):
    """
    将数据集图片与标签对应，并按照标签大小排序
    :param img_address:
    :return:
    """
    lines = [x.strip() for x in open(img_address, 'r').readlines()]
    label_img_map = []
    for line in lines:
        tl = line.split('/')
        label_img_map.append([int(tl[0]), tl[1]])
    label_img_map.sort(reverse=False)
    return label_img_map


path = 'C:/Users/12055/Desktop/开始搞研究/databases/holidays/pictures/'
label_img_map = get_label_img_dict('holiday_features/img.txt')


# for i in range(len(label_img_map)):
#     file = open('label.txt', 'a')
#     file.write(str(label_img_map[i][0]) + '\n')
#     file.close()

# get_VGG16Features(path, label_img_map)
# get_VGG19Features(path, label_img_map)
# get_ResNet50Features(path, label_img_map)
# get_XceptionFeatures(path, label_img_map)
# get_InceptionResNetV2Features(path, label_img_map)
# get_NASNetLargeFeatures(path, label_img_map)
# get_colorFeatures(path, label_img_map, Qs=10, Qv=10)
# get_MobileNetV2Features(path, label_img_map)
get_DenseNetFeatures(path, label_img_map)
get_EfficientNetB0Features(path, label_img_map)


# a = np.load('VGG16_feature.npy')
# print(a)
