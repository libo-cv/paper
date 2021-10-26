from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet50V2 as ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet import preprocess_input as preprocess_input_resnet50
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import math
import torch
from torchvision import models
import tensorflow as tf
import numpy as np
import keras

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def get_VGG16Features(path, label_img_map):
    """
    提取数据集彩色、灰度图的VGG16特征
    :return:
    """
    base_model = VGG16(weights='imagenet')
    # 查看VGG16源码，inculde_top是全连接层进行分类。但是我们需要提取到全连接层的4096输出，是fc2层
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)     # 4096维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_vgg16(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('VGG16: ', i)

    np.save('VGG16_feature.npy', features)
    return 0


def get_VGG19Features(path, label_img_map):
    """
    提取数据集彩色、灰度图像的VGG19特征
    :return:
    """
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)    # 4096维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_vgg19(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('VGG19: ', i)

    np.save('VGG19_feature.npy', features)
    return 0


def get_ResNet50Features(path, label_img_map):
    """
    提取数据集彩色、灰度图像的ResNet50特征
    :return:
    """
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs= base_model.input, outputs=base_model.get_layer('avg_pool').output)  # 2048维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_resnet50(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('ResNet50: ', i)

    np.save('ResNet50_feature.npy', features)
    return 0


def get_XceptionFeatures(path, label_img_map):
    """
    提取数据集彩色、灰度图像的ResNet50特征
    :return:
    """
    base_model = Xception(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)  # 2048维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_xception(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('Xception: ', i)

    np.save('Xception_feature.npy', features)
    return 0


def get_InceptionResNetV2Features(path, label_img_map):
    """
    提取数据集彩色、灰度图像的InceptionResNetV2特征
    :return:
    """
    base_model = InceptionResNetV2(weights='imagenet')
    model = Model(inputs= base_model.input, outputs=base_model.get_layer('avg_pool').output)  # 1536维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_inception_resnet_v2(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('InceptionResNetV2: ', i)

    np.save('InceptionResNetV2_feature.npy', features)
    return 0


def get_NASNetLargeFeatures(path, label_img_map):
    """
    提取数据集彩色、灰度图像的NASNetLarge特征
    :return:
    """
    base_model = NASNetLarge(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)  # 4032维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(331, 331))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_nasnet(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('NASNetLarge: ', i)

    np.save('NASNetLarge_feature.npy', features)
    return 0


def get_colorFeatures(path, label_img_map, Qs, Qv):
    """
    提取颜色特征
    :param path:
    :param label_img_map:
    :param Qs:
    :param Qv:
    :return:
    """
    print('Start to extract Color Features................')
    features = []
    for num in range(len(label_img_map)):
        img_path = path + label_img_map[num][1]
        img = image.load_img(img_path, target_size=(500, 500))
        x = image.img_to_array(img)
        R = x[:, :, 0] / 255
        G = x[:, :, 1] / 255
        B = x[:, :, 2] / 255
        Cmax = np.max(x, axis=2) / 255
        Cmin = np.min(x, axis=2) / 255
        delta = Cmax - Cmin
        H = S = np.zeros((500, 500))
        V = Cmax
        feature = np.zeros(2000)
        for i in range(R[0].size):
            for j in range(R[1].size):
                if delta[i][j] == 0:
                    H[i][j] = 0
                elif Cmax[i][j] == R[i][j]:
                    H[i][j] = 60 * ((G[i][j] -B[i][j]) / delta[i][j] + 0)
                elif Cmax[i][j] == G[i][j]:
                    H[i][j] = 60 * ((B[i][j] - R[i][j]) / delta[i][j] + 2)
                else:
                    H[i][j] = 60 * ((R[i][j] - G[i][j]) / delta[i][j] + 4)
                if H[i][j] < 0:
                    H[i][j] = H[i][j] + 360
                if Cmax[i][j] == 0:
                    S[i][j] = 0
                else:
                    S[i][j] = delta[i][j] / Cmax[i][j]
                H[i][j] = math.floor(H[i][j] / 18)              # 向下取整
                S[i][j] = math.floor(S[i][j] * 10)
                V[i][j] = math.floor(V[i][j] * 10)
        G = H * Qs * Qv + S * Qv + V
        for k in range(2000):
            feature[k] = np.sum(G == k) / (R[0].size * R[1].size)

        features.append(feature)

        if num % 100 == 0:
            print('Color: ', num)

    np.save('Color_feature.npy', features)
    return 0


def get_MobileNetV2Features(path, label_img_map):
    """
    提取数据集彩色、灰度图像的MobileNetV2特征
    :return:
    """
    base_model = MobileNetV2(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)  # 2048维
    model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_mobilenet_v2(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('MobileNetV2: ', i)

    np.save('MobileNetV2_feature.npy', features)
    return 0

def get_DenseNetFeatures(path, label_img_map):
    """
    提取数据集彩色、灰度图像的MobileNetV2特征
    :return:
    """

    base_model = DenseNet(weights='imagenet', blocks=[6, 12, 48, 32])
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)  # 2048维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_densenet(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('DenseNet: ', i)

    np.save('DenseNet_feature.npy', features)
    return 0

def get_EfficientNetB0Features(path, label_img_map):
    """
    提取数据集彩色、灰度图像的MobileNetV2特征
    :return:
    """

    base_model = EfficientNetB0(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('top_dropout').output)  # 2048维
    # model.summary()

    features = []
    for i in range(len(label_img_map)):
        img_path = path + label_img_map[i][1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        gray = np.mean(x, axis=2)
        gray = np.transpose([gray, gray, gray], (1, 2, 0))
        y = np.array([x, gray])
        y = preprocess_input_efficientnet(y)
        feature = model.predict(y)
        # z = features[0, :]
        feature = np.concatenate((0.57 * feature[0, :], 0.43 * feature[1, :]))
        features.append(feature)
        if i % 100 == 0:
            print('EfficientNetB0: ', i)

    np.save('EfficientNetB0_feature.npy', features)
    return 0


# get_VGG16Features()
# get_VGG19Features()
# get_ResNet50Features()
# get_XceptionFeatures()
# get_InceptionResNetV2Features()
# get_NASNetLargeFeatures(0, 0)
# get_colorFeatures(0, 0, 10, 10)
# get_AlexNetFeatures(0, 0)
# get_MobileNetV2Features(0, 0)
# get_DenseNetFeatures(0, 0)
# get_EfficientNetB0Features(0, 0)

