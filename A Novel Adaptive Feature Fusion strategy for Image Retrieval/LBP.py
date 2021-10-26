import os
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数

path = r"C:/Users/12055/Desktop/开始搞研究/databases/Corel-1k/test1/image.orig/"  # 图像所在的文件夹
ls = []
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        ls.append(os.path.join(dirpath, filename))


def get_lbp_feature(ls):
    d = []
    for i in ls:
        t0 = time()
        L = np.zeros(256)
        image = cv2.imread(i)  # 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(image, n_points, radius)
        for k in range(256):
            L[k] = np.sum(lbp == k) / (lbp[0].size * lbp[1].size)
        d.append(L)
        print(time() - t0)

    kk = np.array(d)
    np.save("LBP_feature.npy", kk)


get_lbp_feature(ls)
