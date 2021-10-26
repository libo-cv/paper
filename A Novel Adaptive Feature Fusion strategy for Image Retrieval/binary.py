"""
对特征进行 hash编码
推测"**.npy"为数组格式第一个维度n为样本数量，第二个维度m为特征维度
"""

import numpy as np
# import matplotlib.pyplot as plt
import os

# root = '/share/home/math4/oldcaffe/wangjiaojuan/caffe-master/Apagerank/Holidays/'


def decimal2binary(root, save_path):
    feature = np.load(root)
    n = feature.shape[0]
    m = feature.shape[1]
    kk = np.zeros((n, m))
    for i in range(n):
        n_mean = np.sum(feature[i, :]) / m
        for j in range(m):
            if feature[i, j] >= n_mean:
                kk[i, j] = 1
            else:
                kk[i, j] = 0
    np.save(save_path, kk)
    return 0


path = 'E:/python_program/ImageRetrival/An-adaptive-weight-method-for-image-retrieval-based-multi-feature-fusion-master/RS28/'
num = 200


decimal2binary(path + 'Color_' + str(num) + '_features.npy', path + 'Color_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'VGG16_' + str(num) + '_features.npy', path + 'VGG16_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'VGG19_' + str(num) + '_features.npy', path + 'VGG19_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'Xception_' + str(num) + '_features.npy', path + 'Xception_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'ResNet50_' + str(num) + '_features.npy', path + 'ResNet50_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'NASNetLarge_' + str(num) + '_features.npy', path + 'NASNetLarge_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'InceptionResNetV2_' + str(num) + '_features.npy', path + 'InceptionResNetV2_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'MobileNetV2_' + str(num) + '_features.npy', path + 'MobileNetV2_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'AlexNet_' + str(num) + '_features.npy', path + 'AlexNet_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'LBP_' + str(num) + '_features.npy', path + 'LBP_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'gist_' + str(num) + '_features.npy', path + 'gist_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'DenseNet_' + str(num) + '_features.npy', path + 'DenseNet_binary_' + str(num) + '_features.npy')
decimal2binary(path + 'EfficientNetB0_' + str(num) + '_features.npy', path + 'EfficientNetB0_binary_' + str(num) + '_features.npy')