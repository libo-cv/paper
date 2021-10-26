from __future__ import division
from relation_function import *
import numpy as np
import os
from time import *
from numba import vectorize, cuda
import cupy as cp


# @vectorize(['float32(float32, float32)'], target='cuda')
def compare(root):
    # get all_image label
    filepath = root + 'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN1_feature = np.load(root + 'VGG19_binary_features.npy')
    CNN_feature = np.load(root + 'ResNet50_binary_features.npy')
    color_feature = np.load(root + 'MobileNetV2_binary_features.npy')

    ###################################################################################################################################
    # get retrieval image and database image
    path = 0
    re = []  # 将每类的第一个作为查询样本
    database = []  # 将每类的第二个至最后一个作为数据样本
    while path < origion_line:
        re.append(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        num = end - start + 1
        start = start + 1
        while start <= end:
            database.append(start)
            start = start + 1
        path = path + num
    ############################################################################################################################
    ac1 = 0
    ac2 = 0
    ac3 = 0
    ac5 = 0
    ac6 = 0
    ac7 = 0
    #########################################################################################################################################
    r = 0.4
    o = 5
    #########################
    for i in range(len(re)):
        path = re[i]
        # print path
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        ac_label = []  # 找出正确的标签
        start = start + 1
        while start <= end:
            ac_label.append(start)
            start = start + 1
            ################################### weight
        start_time = time()
        CNN_distance = haming_distance(CNN_feature, path, database)
        result_CNN = query_result(CNN_distance, ac_label, database)
        color_distance = haming_distance(color_feature, path, database)
        result_color = query_result(color_distance, ac_label, database)
        CNN1_distance = haming_distance(CNN1_feature, path, database)
        result_CNN1 = query_result(CNN1_distance, ac_label, database)
        ac1 = ac1 + result_CNN
        ac2 = ac2 + result_color
        ac3 = ac3 + result_CNN1
        #############################
        fix_ac = fix_query(CNN1_distance, CNN_distance, color_distance, ac_label, database)
        reback_ac = reback_query(result_CNN, result_color, result_CNN1, CNN_distance, color_distance, CNN1_distance,
                                 ac_label, database)
        ac5 = ac5 + fix_ac
        ac6 = ac6 + reback_ac
        print(i, "th ac.................is", result_CNN, result_color, result_CNN1, fix_ac, reback_ac)
        #########################################
    ac1 = ac1 / len(re)
    ac2 = ac2 / len(re)
    ac3 = ac3 / len(re)
    ac5 = ac5 / len(re)
    ac6 = ac6 / len(re)
    #########################################################################################################"
    print("........................................comparision is")
    print("CNN,color,CNN1,AVG,RF.................is", ac1, ac2, ac3, ac5, ac6)


# @vectorize(['float32(float32, float32)'], target='cuda')
def with_Entropy(root):
    # get all_image label
    filepath = root + 'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN1_feature = np.load(root + 'VGG19_binary_features.npy')
    CNN_feature = np.load(root + 'ResNet50_binary_features.npy')
    color_feature = np.load(root + 'MobileNetV2_binary_features.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path = 0
    re = []
    database = []
    while path < origion_line:
        re.append(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        num = end - start + 1
        start = start + 1
        while start <= end:
            database.append(start)
            start = start + 1
        path = path + num
    ############################################################################################################################
    ac1 = 0
    ac2 = 0
    ac3 = 0
    #########################################################################################################################################
    r = 0.4
    o = 5
    #########################
    for i in range(len(re)):
        path = re[i]
        # print(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        ac_label = []
        start = start + 1
        while start <= end:
            ac_label.append(start)
            start = start + 1
            ################################### weight
        # start_time = time.perf_counter()
        CNN_distance = Entropy_haming_distance(CNN_feature, path, database)
        result_CNN = query_result(CNN_distance, ac_label, database)
        color_distance = Entropy_haming_distance(color_feature, path, database)
        result_color = query_result(color_distance, ac_label, database)
        CNN1_distance = Entropy_haming_distance(CNN1_feature, path, database)
        result_CNN1 = query_result(CNN1_distance, ac_label, database)
        ac1 = ac1 + result_CNN
        ac2 = ac2 + result_color
        ac3 = ac3 + result_CNN1
        #########################################
    ac1 = ac1 / len(re)
    ac2 = ac2 / len(re)
    ac3 = ac3 / len(re)
    ########################################################################################################"
    print("........................................with Entropy")
    print("CNN,color,CNN1.................is", ac1, ac2, ac3)


# @vectorize(['float32(float32, float32)'], target='cuda')
def unsupervised(root):
    # get all_image label
    filepath = root + 'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN1_feature = np.load(root + 'VGG19_binary_features.npy')
    CNN_feature = np.load(root + 'ResNet50_binary_features.npy')
    color_feature = np.load(root + 'MobileNetV2_binary_features.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path = 0
    re = []
    database = []
    while path < origion_line:
        re.append(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        num = end - start + 1
        start = start + 1
        while start <= end:
            database.append(start)
            start = start + 1
        path = path + num
    ############################################################################################################################
    ac7 = 0
    #########################################################################################################################################
    r = 0.4
    o = 5
    #########################
    start_time = time()
    for i in range(len(re)):
        path = re[i]
        # print(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        ac_label = []
        start = start + 1
        while start <= end:
            ac_label.append(start)
            start = start + 1
            ################################### weight
        CNN_distance = Entropy_haming_distance(CNN_feature, path, database)
        result_CNN = query_result(CNN_distance, ac_label, database)
        color_distance = Entropy_haming_distance(color_feature, path, database)
        result_color = query_result(color_distance, ac_label, database)
        CNN1_distance = Entropy_haming_distance(CNN1_feature, path, database)
        result_CNN1 = query_result(CNN1_distance, ac_label, database)
        ours_ac = unsupervised_fusion(result_CNN, result_color, result_CNN1, CNN_distance, color_distance,
                                      CNN1_distance, ac_label, database, r, o)
        ac7 = ac7 + ours_ac
        print(i, "th ac.................is", result_CNN, result_color, result_CNN1, ours_ac)
        #########################################
    ac7 = ac7 / len(re)
    end_time = time()
    #########################################################################################################"
    print("........................................unsupervised")
    print("our.................!!!!!!!!!!!!!!!!!!!is", ac7, "time is ", end_time - start_time)


# @vectorize(['float32(float32, float32)'], target='cuda')
def supervised(root, num, data_name, find_num, top):
    # get all_image label
    filepath = root + 'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    Color_feature = np.load(root + 'Color_binary_' + str(num) + '_features.npy')
    InceptionResNetV2_feature = np.load(root + 'InceptionResNetV2_binary_' + str(num) + '_features.npy')
    MobileNetV2_feature = np.load(root + 'MobileNetV2_binary_' + str(num) + '_features.npy')
    NASNetLarge_feature = np.load(root + 'NASNetLarge_binary_' + str(num) + '_features.npy')
    ResNet50_feature = np.load(root + 'ResNet50_binary_' + str(num) + '_features.npy')
    VGG16_feature = np.load(root + 'VGG16_binary_' + str(num) + '_features.npy')
    VGG19_feature = np.load(root + 'VGG19_binary_' + str(num) + '_features.npy')
    Xception_feature = np.load(root + 'Xception_binary_' + str(num) + '_features.npy')
    AlexNet_feature = np.load(root + 'AlexNet_binary_' + str(num) + '_features.npy')
    gist_feature = np.load(root + 'gist_binary_' + str(num) + '_features.npy')
    LBP_feature = np.load(root + 'LBP_binary_' + str(num) + '_features.npy')
    DenseNet_feature = np.load(root + 'DenseNet_binary_' + str(num) + '_features.npy')
    EfficientNetB0_feature = np.load(root + 'EfficientNetB0_binary_' + str(num) + '_features.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path = 0
    re = []  # 查询图像
    database = []  # 数据集中的图像
    while path < origion_line:
        index = 1
        while index <= find_num:
            re.append(path)
            path += 1
            index += 1
        name_map = ac_img_label(origion_file)  # 返回各类标签所在的区间
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        num = end - start + 1  # 第path类有num张图片
        start = start + find_num
        while start <= end:
            database.append(start)
            start = start + 1
        path = start
    ############################################################################################################################

    ac1 = ac2 = ac3 = ac4 = ac5 = ac6 = ac7 = ac8 = ac9 = ac10 = ac11 = ac12 = ac13 = ac14 = ac15 = ac16 = 0
    #########################################################################################################################################
    r = 0.4
    o = 5
    #########################
    start_time = time()
    for i in range(len(re)):
        path = re[i]
        # print(path)
        name_map = ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start, end = name_map[re_image]
        ac_label = []
        start = start + find_num
        while start <= end:
            ac_label.append(start)
            start = start + 1
            ################################### weight

        result = []
        Color_distance = Entropy_haming_distance(Color_feature, path, database)
        result_Color = query_result(Color_distance, ac_label, database, top)
        result.append([result_Color, Color_distance])

        InceptionResNetV2_distance = Entropy_haming_distance(InceptionResNetV2_feature, path, database)
        result_InceptionResNetV2 = query_result(InceptionResNetV2_distance, ac_label, database, top)
        result.append([result_InceptionResNetV2, InceptionResNetV2_distance])

        MobileNetV2_distance = Entropy_haming_distance(MobileNetV2_feature, path, database)
        result_MobileNetV2 = query_result(MobileNetV2_distance, ac_label, database, top)
        result.append([result_MobileNetV2, MobileNetV2_distance])

        NASNetLarge_distance = Entropy_haming_distance(NASNetLarge_feature, path, database)
        result_NASNetLarge = query_result(NASNetLarge_distance, ac_label, database, top)
        result.append([result_NASNetLarge, NASNetLarge_distance])

        ResNet50_distance = Entropy_haming_distance(ResNet50_feature, path, database)
        result_ResNet50 = query_result(ResNet50_distance, ac_label, database, top)
        result.append([result_ResNet50, ResNet50_distance])

        VGG16_distance = Entropy_haming_distance(VGG16_feature, path, database)
        result_VGG16 = query_result(VGG16_distance, ac_label, database, top)
        result.append([result_VGG16, VGG16_distance])

        VGG19_distance = Entropy_haming_distance(VGG19_feature, path, database)
        result_VGG19 = query_result(VGG19_distance, ac_label, database, top)
        result.append([result_VGG19, VGG19_distance])

        Xception_distance = Entropy_haming_distance(Xception_feature, path, database)
        result_Xception = query_result(Xception_distance, ac_label, database, top)
        result.append([result_Xception, Xception_distance])

        AlexNet_distance = Entropy_haming_distance(AlexNet_feature, path, database)
        result_AlexNet = query_result(AlexNet_distance, ac_label, database, find_num)
        result.append([result_AlexNet, AlexNet_distance])

        gist_distance = Entropy_haming_distance(gist_feature, path, database)
        result_gist = query_result(gist_distance, ac_label, database, find_num)
        result.append([result_gist, gist_distance])

        LBP_distance = Entropy_haming_distance(LBP_feature, path, database)
        result_LBP = query_result(LBP_distance, ac_label, database, find_num)
        result.append([result_LBP, LBP_distance])

        DenseNet_distance = Entropy_haming_distance(DenseNet_feature, path, database)
        result_DenseNet = query_result(DenseNet_distance, ac_label, database, find_num)
        result.append([result_DenseNet, DenseNet_distance])

        EfficientNetB0_distance = Entropy_haming_distance(EfficientNetB0_feature, path, database)
        result_EfficientNetB0 = query_result(EfficientNetB0_distance, ac_label, database, find_num)
        result.append([result_EfficientNetB0, EfficientNetB0_distance])

        result.sort(key=lambda x: (x[0]), reverse=True)


        fix_accuracy = fix_query(result[0][1], result[1][1], result[2][1], result[3][1], result[4][1], ac_label, database, top)

        # reback_accuracy = reback_query(result[0][0], result[1][0], result[2][0], result[3][0], result[4][0],
        #                         result[0][1], result[1][1], result[2][1], result[3][1], result[4][1],
        #                          ac_label, database, top)

        # Entropy_accuracy, mix_accuracy = unsupervised_fusion(result[0][1], result[1][1], result[2][1], result[3][1], result[4][1],
        #                          ac_label, database, r, o, top)

        mix_accuracy = ours_query(result[0][0], result[1][0], result[2][0], result[3][0], result[4][0],
                                  result[0][1], result[1][1], result[2][1], result[3][1], result[4][1],
                                  ac_label, database, r, o, top)

        # print(i, 'th result: ', result_Color, result_InceptionResNetV2, result_MobileNetV2, result_NASNetLarge,
        #       result_ResNet50, result_VGG16, result_VGG19, result_Xception,
        #       result_AlexNet, result_gist, result_LBP, result_DenseNet, result_EfficientNetB0,
        #       fix_accuracy, mix_accuracy)
        # result_AlexNet, result_gist, result_LBP,
        ac1 += result_Color
        ac2 += result_InceptionResNetV2
        ac3 += result_MobileNetV2
        ac4 += result_NASNetLarge
        ac5 += result_ResNet50
        ac6 += result_VGG16
        ac7 += result_VGG19
        ac8 += result_Xception
        ac9 += result_AlexNet
        ac10 += result_gist
        ac11 += result_LBP
        ac12 += result_DenseNet
        ac13 += result_EfficientNetB0
        ac14 += fix_accuracy
        # ac15 += Entropy_accuracy
        ac16 += mix_accuracy
        #########################################

    ac1 = ac1 / len(re)
    ac2 = ac2 / len(re)
    ac3 = ac3 / len(re)
    ac4 = ac4 / len(re)
    ac5 = ac5 / len(re)
    ac6 = ac6 / len(re)
    ac7 = ac7 / len(re)
    ac8 = ac8 / len(re)
    ac9 = ac9 / len(re)
    ac10 = ac10 / len(re)
    ac11 = ac11 / len(re)
    ac12 = ac12 / len(re)
    ac13 = ac13 / len(re)
    ac14 = ac14 / len(re)
    # ac15 = ac15 / len(re)
    ac16 = ac16 / len(re)
    end_time = time()
    #########################################################################################################"
    print("........................................supervised")
    print("our.................!!!!!!!!!!!!!!!!!!!is \n",
          ac1, '\n', ac2, '\n', ac3, '\n', ac4, '\n', ac5, '\n', ac6, '\n', ac7, '\n', ac8, '\n', ac9, '\n', ac10, '\n',
          ac11, '\n', ac12, '\n', ac13, '\n', ac14, '\n', ac15, '\n', ac16, '\n', "time is '\n' ", end_time - start_time)

    # file = open('accuracy_loss.txt', 'a')
    # file.write("epoch:" + str(epoch) + '\n')
    # file.write("Train accuracy is: " + str(train_accuracy) + "%; Train loss is: " + str(train_loss) + '\n')
    # #         file.write("Val accuracy is: " + str(val_accuracy) + "%; Val loss is: " + str(val_loss) + '\n')
    # file.write("Test accuracy is: " + str(test_accuracy) + "%; Test loss is: " + str(test_loss) + '\n')
    # file.close()


#################################################
# root='/share/home/math4/oldcaffe/wangjiaojuan/caffe-master/Apagerank/Holidays/'
root = 'E:/python_program/ImageRetrival/An-adaptive-weight-method-for-image-retrieval-based-multi-feature-fusion-master/UC21/'
num = 100
data_name = 'holiday'
find_number = 80
top = 10
# compare(root)
# with_Entropy(root)
# unsupervised(root)
# print('——————————————————————200dimension———top10————————————————————————————')
# supervised(root, 200, data_name, find_number, 10)
# print('——————————————————————200dimension———top20————————————————————————————')
# supervised(root, 200, data_name, find_number, 20)
# print('——————————————————————200dimension———top30————————————————————————————')
# supervised(root, 200, data_name, find_number, 30)
# print('——————————————————————200dimension———top40————————————————————————————')
# supervised(root, 200, data_name, find_number, 40)
# print('——————————————————————200dimension———top50————————————————————————————')
# supervised(root, 200, data_name, find_number, 50)
#
# print('——————————————————————100dimension———top10————————————————————————————')
# supervised(root, 100, data_name, find_number, 10)
# print('——————————————————————100dimension———top20————————————————————————————')
# supervised(root, 100, data_name, find_number, 20)
# print('——————————————————————100dimension———top30————————————————————————————')
# supervised(root, 100, data_name, find_number, 30)
# print('——————————————————————100dimension———top40————————————————————————————')
# supervised(root, 100, data_name, find_number, 40)
# print('——————————————————————100dimension———top50————————————————————————————')
# supervised(root, 100, data_name, find_number, 50)
#
# print('——————————————————————50dimension———top10————————————————————————————')
# supervised(root, 50, data_name, find_number, 10)
# print('——————————————————————50dimension———top20————————————————————————————')
# supervised(root, 50, data_name, find_number, 20)
# print('——————————————————————50dimension———top30————————————————————————————')
# supervised(root, 50, data_name, find_number, 30)
# print('——————————————————————50dimension———top40————————————————————————————')
# supervised(root, 50, data_name, find_number, 40)
# print('——————————————————————50dimension———top50————————————————————————————')
# supervised(root, 50, data_name, find_number, 50)
#
# print('——————————————————————40dimension———top10————————————————————————————')
# supervised(root, 40, data_name, find_number, 10)
# print('——————————————————————40dimension———top20————————————————————————————')
# supervised(root, 40, data_name, find_number, 20)
# print('——————————————————————40dimension———top30————————————————————————————')
# supervised(root, 40, data_name, find_number, 30)
# print('——————————————————————40dimension———top40————————————————————————————')
# supervised(root, 40, data_name, find_number, 40)
# print('——————————————————————40dimension———top50————————————————————————————')
# supervised(root, 40, data_name, find_number, 50)
#
# print('——————————————————————30dimension———top10————————————————————————————')
# supervised(root, 30, data_name, find_number, 10)
# print('——————————————————————30dimension———top20————————————————————————————')
# supervised(root, 30, data_name, find_number, 20)
# print('——————————————————————30dimension———top30————————————————————————————')
# supervised(root, 30, data_name, find_number, 30)
# print('——————————————————————30dimension———top40————————————————————————————')
# supervised(root, 30, data_name, find_number, 40)
# print('——————————————————————30dimension———top50————————————————————————————')
# supervised(root, 30, data_name, find_number, 50)
#
# print('——————————————————————20dimension———top10————————————————————————————')
# supervised(root, 20, data_name, find_number, 10)
# print('——————————————————————20dimension———top20————————————————————————————')
# supervised(root, 20, data_name, find_number, 20)
# print('——————————————————————20dimension———top30————————————————————————————')
# supervised(root, 20, data_name, find_number, 30)
# print('——————————————————————20dimension———top40————————————————————————————')
# supervised(root, 20, data_name, find_number, 40)
# print('——————————————————————20dimension———top50————————————————————————————')
# supervised(root, 20, data_name, find_number, 50)

# print('——————————————————————50dimension———top10————————————————————————————')
# supervised(root, 50, data_name, find_number, 10)
# print('——————————————————————10dimension———top20————————————————————————————')
# supervised(root, 10, data_name, find_number, 20)
# print('——————————————————————10dimension———top30————————————————————————————')
# supervised(root, 10, data_name, find_number, 30)
# print('——————————————————————10dimension———top40————————————————————————————')
# supervised(root, 10, data_name, find_number, 40)
# print('——————————————————————10dimension———top50————————————————————————————')
# supervised(root, 10, data_name, find_number, 50)

# print('——————————————————————9dimension———top10————————————————————————————')
# supervised(root, 9, data_name, find_number, 10)
# print('——————————————————————9dimension———top20————————————————————————————')
# supervised(root, 9, data_name, find_number, 20)
# print('——————————————————————9dimension———top30————————————————————————————')
# supervised(root, 9, data_name, find_number, 30)
# print('——————————————————————9dimension———top40————————————————————————————')
# supervised(root, 9, data_name, find_number, 40)
# print('——————————————————————9dimension———top50————————————————————————————')
# supervised(root, 9, data_name, find_number, 50)


# print('——————————————————————50dimension———top10————————————————————————————')
# supervised(root, 50, 50, data_name, find_number, 10)
# print('——————————————————————100dimension———top10————————————————————————————')
# supervised(root, 100, 100, data_name, find_number, 10)
# print('——————————————————————150dimension———top10————————————————————————————')
# supervised(root, 150, 150, data_name, find_number, 10)
# print('——————————————————————200dimension———top10————————————————————————————')
# supervised(root, 200, 200, data_name, find_number, 10)
# print('——————————————————————250dimension———top10————————————————————————————')
# supervised(root, 250, 250, data_name, find_number, 10)
# print('——————————————————————300dimension———top10————————————————————————————')
# supervised(root, 300, 250, data_name, find_number, 10)
# print('——————————————————————350dimension———top10————————————————————————————')
# supervised(root, 350, 250, data_name, find_number, 10)
# print('——————————————————————400dimension———top10————————————————————————————')
# supervised(root, 400, 250, data_name, find_number, 10)
# print('——————————————————————500dimension———top10————————————————————————————')
# supervised(root, 500, 250, data_name, find_number, 10)


dimension_num = [10, 20, 30, 40, 50, 100, 200]
top_num = [10, 20, 30, 40, 50]
for i in range(len(dimension_num)):
    for j in range(len(top_num)):
        print('***********************', dimension_num[i], 'dimension, top', top_num[j], '******************')
        supervised(root, dimension_num[i], data_name, find_number, top_num[j])
