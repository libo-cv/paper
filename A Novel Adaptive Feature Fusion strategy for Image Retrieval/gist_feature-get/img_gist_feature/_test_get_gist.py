import os
import cv2
import sys
from PIL import Image
import numpy as np
sys.path.append("./img_gist_feature/")
from utils_gist import *

path= r"C:/Users/Administrator/Desktop/dataset/UCMerced_LandUse/"
ls=[]
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        ls.append(os.path.join(dirpath,filename))

def dist_feature(ls):
    L=[]
    num=0
    for s_img_url in ls:    
        gist_helper = GistUtils()
        np_img = cv2.imread(s_img_url, -1)
        np_gist_rgb = gist_helper.get_gist_vec(np_img, mode="rgb")
        np_gist_gray = gist_helper.get_gist_vec(np_img, mode="gray")
        feature=np.concatenate((np_gist_rgb.flatten(),np_gist_gray.flatten()),axis=0)
        L.append(feature)
        num+=1
        print(num)
    kk=np.array(L)
    np.save("gist_feature",kk)

dist_feature(ls)


data=np.load("gist_feature.npy")
print(data)
print(data.shape)