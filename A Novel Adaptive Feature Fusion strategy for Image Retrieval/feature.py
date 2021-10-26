import numpy as np
import tensorflow as tf
import keras
from get_features import *


path = 'C:/Users/12055/Desktop/开始搞研究/databases/Corel-1k/test1/image.orig/'

# lines = [x.strip() for x in open('wang_label.txt', 'r').readlines()]

label_img_map = []
for i in range(1000):
    label_img_map.append([i, str(i) + '.jpg'])

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
