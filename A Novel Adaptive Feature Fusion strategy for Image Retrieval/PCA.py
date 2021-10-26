#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 主成分分析 降维
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# 参数初始化
# inputfile = 'input.xls'
# data = pd.read_excel(inputfile, header=None)  # 读入数据

# 此为注释部分，按需打开即可
'''
pca = PCA()
pca.fit(data)
print(pca.components_) #返回模型的各个特征向量
print(pca.explained_variance_ratio_) #返回各个成分各自的方差百分比
'''


def pca(root, path, num):
    """

    :param root:
    :param path:
    :return:
    """
    data = np.load(root)
    pca = PCA(n_components=num)
    pca.fit(data)
    low_d = pca.transform(data)  # 用它来降低维度
    print(sum(pca.explained_variance_ratio_))
    np.save(path, low_d)
    return 0


# pd.DataFrame(low_d).to_excel('output.xls')  # 保存结果
# pca.inverse_transform(low_d)     #必要时可以用inverse_transform()函数来复原数据，按需应用
path = 'E:/python_program/ImageRetrival/An-adaptive-weight-method-for-image-retrieval-based-multi-feature-fusion-master/RS28/'
num = 200

pca(path + 'Color_feature.npy', path + 'Color_' + str(num) + '_features.npy', num)
pca(path + 'InceptionResNetV2_feature.npy', path + 'InceptionResNetV2_' + str(num) + '_features.npy', num)
pca(path + 'MobileNetV2_feature.npy', path + 'MobileNetV2_' + str(num) + '_features.npy', num)
pca(path + 'NASNetLarge_feature.npy', path + 'NASNetLarge_' + str(num) + '_features.npy', num)
pca(path + 'ResNet50_feature.npy', path + 'ResNet50_' + str(num) + '_features.npy', num)
pca(path + 'VGG16_feature.npy', path + 'VGG16_' + str(num) + '_features.npy', num)
pca(path + 'VGG19_feature.npy', path + 'VGG19_' + str(num) + '_features.npy', num)
pca(path + 'Xception_feature.npy', path + 'Xception_' + str(num) + '_features.npy', num)
pca(path + 'AlexNet_feature.npy', path + 'AlexNet_' + str(num) + '_features.npy', num)
pca(path + 'gist_feature.npy', path + 'gist_' + str(num) + '_features.npy', num)
pca(path + 'LBP_feature.npy', path + 'LBP_' + str(num) + '_features.npy', num)
pca(path + 'DenseNet_feature.npy', path + 'DenseNet_' + str(num) + '_features.npy', num)
pca(path + 'EfficientNetB0_feature.npy', path + 'EfficientNetB0_' + str(num) + '_features.npy', num)

# data = np.load('VGG16_feature.npy')
# print(data)