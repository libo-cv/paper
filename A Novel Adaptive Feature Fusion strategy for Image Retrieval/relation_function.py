"""
定义了一些相关的函数
"""

from __future__ import division
import numpy as np
import math
import cupy as cp
from numba import vectorize, cuda


# @vectorize(['float32(float32, float32)'], target='cuda')
def ac_img_label(origion_file):
    """
    返回数据集每类样本所占的区间
    :param origion_file:标签文件
    :return:第i类样本在第m个到第n个位置
    """
    origion_line = len(origion_file)
    name_map = {}
    for j in range(origion_line):
        image_name = origion_file[j]
        class_name = image_name.split('/')[0]
        if class_name in name_map.keys():
            name_map[class_name][1] += 1
        else:
            name_map[class_name] = [j, j]
    return name_map


# @vectorize(['float32(float32, float32)'], target='cuda')
def haming_distance(CNN_feature, path, database):
    """
    :param CNN_feature: 数据集特征
    :param path: 查询样本编号
    :param database: 数据集（什么格式的？尺寸如何？）
    :return: 查询样本与数据集各个样本之间的L1距离
    """
    re_feature = CNN_feature[path, :]  # path应该是查询样本的编号吧？re_feature为查询样本特征
    distance = np.zeros(len(database))  # 长度为数据集样本数、元素为0的一维数组
    for j in range(len(database)):
        im_feature = CNN_feature[database[j], :]  # 找出数据集中第j个样本的特征
        dist = np.sum(np.abs(np.subtract(re_feature, im_feature)))  # 计算第j个样本与查询样本之间的距离，各个特征分量作差取绝对值之后求和
        distance[j] = dist  # 存储到distance数组中
    distance = np.divide(distance, (np.sum(distance)))  # 距离进行归一化
    return distance  # 返回查询样本与数据集中各个样本之间的距离


# @vectorize(['float32(float32, float32)'], target='cuda')
def feature_Entropy(feature):
    """
    :param feature: 特征数组n*m，m：特征维度；n：数据集样本数
    :return:特征各个分量的权重
    """
    n = feature.shape[0]  # n为数据集样本数
    m = feature.shape[1]  # m为特征维度
    H_Entropy = []
    for s in range(m):  # 对特征集第s维特征进行归一化
        feature[:, s] = feature[:, s] / np.sum(feature[:, s])
    for s in range(m):
        HE = 0  # 初始化熵为0
        for t in range(n):
            if feature[t, s] > 0:
                HE += feature[t, s] * math.log(feature[t, s], 2)   # 计算第s维特征的特征熵
        HHE = HE * ((-1) / math.log(n, 2))  # 对熵进行归一化
        HHE = math.exp(1 - HHE)
        H_Entropy.append(HHE)  # 对指数化后的熵进行增添到熵数组中
    H_Entropy = H_Entropy / np.sum(H_Entropy)  # 进行softmax
    return H_Entropy  # 返回权重，m维向量


# @vectorize(['float32(float32, float32)'], target='cuda')
def Entropy_haming_distance(CNN_feature, path, database):
    """

    :param CNN_feature:数据集特征
    :param path:查询样本编号
    :param database:数据集
    :return:查询样本与数据集中各个样本之间的加权距离
    """
    H_Entropy = feature_Entropy(CNN_feature)  # 调用feature_Entropy()函数，输出特征各个分量的权重
    re_feature = CNN_feature[path, :]  # 查询样本特征
    distance = np.zeros(len(database))  # 长度为数据集样本数、元素为0的一维数组
    for j in range(len(database)):
        im_feature = CNN_feature[database[j], :]  # 找出数据集中第j个样本的特征
        dist = 0  # 初始化距离为0
        A = abs(re_feature - im_feature)  # A为查询样本的特征与数据集中第j个样本的特征差的绝对值，是一个m维向量
        for yh in range(len(A)):
            if A[yh] != 0:
                dist += H_Entropy[yh]  # 如果特征差的第yh个分量不为零，那么距离累加其权重，因为hash编码之后的特征差要么是1要么是0
        distance[j] = dist  # 将各个分量权重的累加，作为数据集第j个样本与查询样本的距离
    distance = distance / (np.sum(distance))  # 将距离归一化
    return distance


# @vectorize(['float32(float32, float32)'], target='cuda')
def mean_average_precison(ac_label, result):
    """
    专门计算mAP的函数
    :param ac_label: 正确的标签
    :param result: 模型得到的模型结果
    :return:mAP
    """
    similar_num = 0  # 初始化相似和为0
    mean_average_precison = 0  # 初始化平均准确率为0
    for j in range(len(result)):
        if (result[j] in ac_label):
            similar_num += 1
            mean_average_precison += (float(similar_num) / (j + 1))  # 第similar_num个相似结果在第j+1个位置，j+1是因为j从0开始计数
            # print mean_average_precison
    mean_average_precison = mean_average_precison / len(ac_label)  # ？？？？为啥要除以查询样本的数量，不是应该除以相似结果的数量嘛？
    # mean_average_precison = mean_average_precison / similar_num          # 自己改的
    return mean_average_precison


def Precision_Ratio(ac_label, result, find_num):
    similar_num = 0
    for j in range(find_num):
        if result[j] in ac_label:
            similar_num += 1
    precision_ratio = similar_num / find_num
    return precision_ratio

# @vectorize(['float32(float32, float32)'], target='cuda')
def query_result(distance, ac_label, database, find_num):
    """
    根据单一特征距离，计算某一次查询的mAP
    :param distance: 查询样本与数据集中各个样本之间的距离
    :param ac_label: 查询样本的标签
    :param database: 数据集
    :return:mAP
    """
    sub_distance = sorted(distance)  # 对距离进行升序排列
    listc = []
    for t in range(len(sub_distance)):  # 依照距离从小到大的顺序，对数据集进行排序
        for h in range(len(distance)):
            if (sub_distance[t] == distance[h]) and (not database[h] in listc):
                listc.append(database[h])
    # ac = mean_average_precison(ac_label, listc)  # 返回mAP
    ac = Precision_Ratio(ac_label, listc, find_num)
    return ac


# @vectorize(['float32(float32, float32)'], target='cuda')
def fix_query(CNN1_distance, CNN_distance, color_distance, CNN2_distance, CNN3_distance, ac_label, database, find_num):
    """
    根据三个特征距离加权平均之后特征，计算某一次查询的mAP
    :param CNN1_distance: CNN1特征距离
    :param CNN_distance: CNN特征距离
    :param color_distance: 颜色特征距离
    :param ac_label: 查询样本的标签
    :param database: 数据集
    :return: mAP
    """
    w_CNN = 1 / 5
    w_color = 1 / 5
    w_CNN1 = 1 / 5
    w_CNN2 = 1 / 5
    w_CNN3 = 1 / 5
    distance = CNN1_distance * w_CNN1 + CNN_distance * w_CNN + color_distance * w_color + CNN2_distance * w_CNN2 + CNN3_distance * w_CNN3
    sub_distance = sorted(distance)
    listc = []
    for t in range(len(sub_distance)):
        for h in range(len(distance)):
            if (sub_distance[t] == distance[h]) and (not database[h] in listc):
                listc.append(database[h])
    # ac = mean_average_precison(ac_label, listc)
    ac = Precision_Ratio(ac_label, listc, find_num)
    return ac


# @vectorize(['float32(float32, float32)'], target='cuda')
def reback_query(result_CNN, result_color, result_CNN1, result_CNN2, result_CNN3,
                 CNN_distance, color_distance, CNN1_distance, CNN2_distance, CNN3_distance, ac_label, database, find_num):
    """
    特征权重固定时，查询样本的mAP
    :param result_CNN:
    :param result_color:
    :param result_CNN1:
    :param CNN_distance:
    :param color_distance:
    :param CNN1_distance:
    :param ac_label:
    :param database:
    :return:
    """
    global w_CNN, w_color, w_CNN1
    w_sum = result_CNN + result_color + result_CNN1 + result_CNN2 + result_CNN3
    if w_sum == 0:
        w_CNN = 1 / 5
        w_color = 1 / 5
        w_CNN1 = 1 / 5
        w_CNN2 = 1 / 5
        w_CNN3 = 1 / 5
    else:
        w_CNN = result_CNN / w_sum
        w_color = result_color / w_sum
        w_CNN1 = result_CNN1 / w_sum
        w_CNN2 = result_CNN2 / w_sum
        w_CNN3 = result_CNN3 / w_sum
    distance = CNN1_distance * w_CNN1 + CNN_distance * w_CNN + color_distance * w_color + CNN2_distance * w_CNN2 + CNN3_distance * w_CNN3
    sub_distance = sorted(distance)
    listc = []
    for t in range(len(sub_distance)):
        for h in range(len(distance)):
            if (sub_distance[t] == distance[h]) and (not database[h] in listc):
                listc.append(database[h])
    # ac = mean_average_precison(ac_label, listc)
    ac = Precision_Ratio(ac_label, listc, find_num)
    return ac


# @vectorize(['float32(float32, float32)'], target='cuda')
def ours_query(result_CNN, result_color, result_CNN1, result_CNN2, result_CNN3, CNN_distance, color_distance, CNN1_distance,
               CNN2_distance, CNN3_distance, ac_label, database, r, o, find_num):
    """
    权重迭代优化时，查询样本的mAP
    :param result_CNN:
    :param result_color:
    :param result_CNN1:
    :param CNN_distance:
    :param color_distance:
    :param CNN1_distance:
    :param ac_label:
    :param database:
    :param r: 权重迭代优化算法中的参数，取值范围为0到1
    :param o: 在计算转移矩阵时的参数，取值范围大于等于1
    :return:
    """
    w_sum = result_CNN + result_color + result_CNN1
    if w_sum == 0:
        B = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
    else:
        pre1 = result_CNN  # CNN特征的信任
        pre2 = result_color  # 颜色特征的信任
        pre3 = result_CNN1  # CNN1特征的信任
        pre4 = result_CNN2
        pre5 = result_CNN3
        B = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
        A = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
        HH = [[0, pre1 - pre2, pre1 - pre3, pre1 - pre4, pre1 - pre5],
              [pre2 - pre1, 0, pre2 - pre3, pre2 - pre4, pre2 - pre5],
              [pre3 - pre1, pre3 - pre2, 0, pre3 - pre4, pre3 - pre5],
              [pre4 - pre1, pre4 - pre2, pre4 - pre3, 0, pre4 - pre5],
              [pre5 - pre1, pre5 - pre2, pre5 - pre3, pre5 - pre4, 0]]
        # HH = [[0, np.subtract(pre1, pre2), np.subtract(pre1, pre3), np.subtract(pre1, pre4), np.subtract(pre1, pre5)],
        #       [np.subtract(pre2, pre1), 0, np.subtract(pre2, pre3), np.subtract(pre2, pre4), np.subtract(pre2, pre5)],
        #       [np.subtract(pre3, pre1), np.subtract(pre3, pre2), 0, np.subtract(pre3, pre4), np.subtract(pre3, pre5)],
        #       [np.subtract(pre4, pre1), np.subtract(pre4, pre2), np.subtract(pre4, pre3), 0, np.subtract(pre4, pre5)],
        #       [np.subtract(pre5, pre1), np.subtract(pre5, pre2), np.subtract(pre5, pre3), np.subtract(pre5, pre4), 0]]
        for ss in range(len(HH)):  # 构造转移矩阵H
            for tt in range(len(HH)):
                HH[ss][tt] = (HH[ss][tt] + 1) / 2
                # if (HH[ss][tt] >= 0):
                #     HH[ss][tt] = math.exp(o * HH[ss][tt])
                # else:
                #     HH[ss][tt] = abs(HH[ss][tt])
        # HH = np.array(HH)

        for ss in range(len(HH)):  # 归一化
            pro_sum = np.sum(x[ss] for x in HH)
            for tt in range(len(HH)):
                HH[tt][ss] = HH[tt][ss] / pro_sum
        # 权重的迭代优化
        kkk = 10000
        while (kkk > 0.0004):  # 迭代误差不超过0.0005
            A[0] = r * B[0] + (1 - r) * (HH[0][0] * B[0] + HH[0][1] * B[1] + HH[0][2] * B[2] + HH[0][3] * B[3] + HH[0][4] * B[4])
            A[1] = r * B[1] + (1 - r) * (HH[1][0] * B[0] + HH[1][1] * B[1] + HH[1][2] * B[2] + HH[1][3] * B[3] + HH[1][4] * B[4])
            A[2] = r * B[2] + (1 - r) * (HH[2][0] * B[0] + HH[2][1] * B[1] + HH[2][2] * B[2] + HH[2][3] * B[3] + HH[2][4] * B[4])
            A[3] = r * B[3] + (1 - r) * (HH[3][0] * B[0] + HH[3][1] * B[1] + HH[3][2] * B[2] + HH[3][3] * B[3] + HH[3][4] * B[4])
            A[4] = r * B[4] + (1 - r) * (HH[4][0] * B[0] + HH[4][1] * B[1] + HH[4][2] * B[2] + HH[4][3] * B[3] + HH[4][4] * B[4])
            sss = A[0] + A[1] + A[2] + A[3] + A[4]
            A[0] = A[0] / sss
            A[1] = A[1] / sss
            A[2] = A[2] / sss
            A[3] = A[3] / sss
            A[4] = A[4] / sss
            B[0] = r * A[0] + (1 - r) * (HH[0][0] * A[0] + HH[0][1] * A[1] + HH[0][2] * A[2] + HH[0][3] * A[3] + HH[0][4] * A[4])
            B[1] = r * A[1] + (1 - r) * (HH[1][0] * A[0] + HH[1][1] * A[1] + HH[1][2] * A[2] + HH[1][3] * A[3] + HH[1][4] * A[4])
            B[2] = r * A[2] + (1 - r) * (HH[2][0] * A[0] + HH[2][1] * A[1] + HH[2][2] * A[2] + HH[2][3] * A[3] + HH[2][4] * A[4])
            B[3] = r * A[3] + (1 - r) * (HH[3][0] * A[0] + HH[3][1] * A[1] + HH[3][2] * A[2] + HH[3][3] * A[3] + HH[3][4] * A[4])
            B[4] = r * A[4] + (1 - r) * (HH[4][0] * A[0] + HH[4][1] * A[1] + HH[4][2] * A[2] + HH[4][3] * A[3] + HH[4][4] * A[4])
            sss = B[0] + B[1] + B[2] + B[3] + B[4]
            B[0] = B[0] / sss
            B[1] = B[1] / sss
            B[2] = B[2] / sss
            B[3] = B[3] / sss
            B[4] = B[4] / sss

            # A = np.add(np.multiply(r, B), np.multiply((1 - r), np.matmul(HH, B)))
            # A = np.divide(A, np.sum(A))
            # B = np.add(np.multiply(r, A), np.multiply((1 - r), np.matmul(HH, A)))
            # B = np.divide(B, np.sum(B))


            kkk =np.sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) + (A[2] - B[2]) * (A[2] - B[2]) + (A[3] - B[3]) * (A[3] - B[3]) + (A[4] - B[4]) * (A[4] - B[4]))
            # kkk = np.sqrt(np.sum(np.square(A - B)))
    # print("weight...............is", B[0], B[1], B[2], B[3], B[4])
    distance = CNN1_distance * B[2] + CNN_distance * B[0] + color_distance * B[1] + CNN2_distance * B[3] + CNN3_distance * B[4]
    sub_distance = sorted(distance)
    listc = []
    # lo=0
    for t in range(len(sub_distance)):
        for h in range(len(distance)):
            if (sub_distance[t] == distance[h]) and (not database[h] in listc):
                listc.append(database[h])
                # while (lo<4):
                #      print 1-distance[h]
                #      lo=lo+1
    # print listc[0:4]
    # ac = mean_average_precison(ac_label, listc)
    ac = Precision_Ratio(ac_label, listc, find_num)
    return ac


# @vectorize(['float32(float32, float32)'], target='cuda')
def weight_Entropy(ddist):
    """

    :param ddist:
    :return:ddist归一化后的熵
    """
    HE = 0
    ddist = ddist /np.sum(ddist)  # ddist归一化，使其变成概率
    # pdb.set_trace()
    for s in range(len(ddist)):
        if ddist[s] > 0:
            HE = HE + ddist[s] * math.log(ddist[s], 2)  # 熵求和
    HHE = HE * ((-1) / math.log(len(ddist), 2))  # 熵归一化
    # HHEE=math.exp(1-HHE)
    return HHE


# @vectorize(['float32(float32, float32)'], target='cuda')
def unsupervised_fusion(distance1, distance2, distance3, distance4, distance5, ac_label, database, r, o, find_num):
    """
    将不同特征的距离熵作为特征的信任，由此求得特征的权重，计算mAP
    :param distance1:
    :param distance2:
    :param distance3:
    :param ac_label:
    :param database:
    :param r:
    :param o:
    :return: mAP
    """
    # 将熵作为特征的信任

    pre1 = weight_Entropy(distance1)
    pre2 = weight_Entropy(distance2)
    pre3 = weight_Entropy(distance3)
    pre4 = weight_Entropy(distance4)
    pre5 = weight_Entropy(distance5)
    pre_sum = pre5 + pre4 + pre3 + pre2 + pre1
    B = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
    A = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
    HH = [[0, pre1 - pre2, pre1 - pre3, pre1 - pre4, pre1 - pre5],
          [pre2 - pre1, 0, pre2 - pre3, pre2 - pre4, pre2 - pre5],
          [pre3 - pre1, pre3 - pre2, 0, pre3 - pre4, pre3 - pre5],
          [pre4 - pre1, pre4 - pre2, pre4 - pre3, 0, pre4 - pre5],
          [pre5 - pre1, pre5 - pre2, pre5 - pre3, pre5 - pre4, 0]]
    # HH = [[0, np.subtract(pre1, pre2), np.subtract(pre1, pre3), np.subtract(pre1, pre4), np.subtract(pre1, pre5)],
    #       [np.subtract(pre2, pre1), 0, np.subtract(pre2, pre3), np.subtract(pre2, pre4), np.subtract(pre2, pre5)],
    #       [np.subtract(pre3, pre1), np.subtract(pre3, pre2), 0, np.subtract(pre3, pre4), np.subtract(pre3, pre5)],
    #       [np.subtract(pre4, pre1), np.subtract(pre4, pre2), np.subtract(pre4, pre3), 0, np.subtract(pre4, pre5)],
    #       [np.subtract(pre5, pre1), np.subtract(pre5, pre2), np.subtract(pre5, pre3), np.subtract(pre5, pre4), 0]]
    for ss in range(len(HH)):  # 构造转移矩阵H
        for tt in range(len(HH)):
            # HH[ss][tt] = (HH[ss][tt] + 1) / 2
            if (HH[ss][tt] >= 0):
                HH[ss][tt] = math.exp(o * HH[ss][tt])
            else:
                HH[ss][tt] = 1-abs(HH[ss][tt])
    # HH = np.array(HH)

    for ss in range(len(HH)):  # 归一化
        pro_sum = np.sum(x[ss] for x in HH)
        for tt in range(len(HH)):
            HH[tt][ss] = HH[tt][ss] / pro_sum
    # 权重的迭代优化
    kkk = 10000
    while (kkk > 0.0004):  # 迭代误差不超过0.0005
        A[0] = r * B[0] + (1 - r) * (
                    HH[0][0] * B[0] + HH[0][1] * B[1] + HH[0][2] * B[2] + HH[0][3] * B[3] + HH[0][4] * B[4])
        A[1] = r * B[1] + (1 - r) * (
                    HH[1][0] * B[0] + HH[1][1] * B[1] + HH[1][2] * B[2] + HH[1][3] * B[3] + HH[1][4] * B[4])
        A[2] = r * B[2] + (1 - r) * (
                    HH[2][0] * B[0] + HH[2][1] * B[1] + HH[2][2] * B[2] + HH[2][3] * B[3] + HH[2][4] * B[4])
        A[3] = r * B[3] + (1 - r) * (
                    HH[3][0] * B[0] + HH[3][1] * B[1] + HH[3][2] * B[2] + HH[3][3] * B[3] + HH[3][4] * B[4])
        A[4] = r * B[4] + (1 - r) * (
                    HH[4][0] * B[0] + HH[4][1] * B[1] + HH[4][2] * B[2] + HH[4][3] * B[3] + HH[4][4] * B[4])
        sss = A[0] + A[1] + A[2] + A[3] + A[4]
        A[0] = A[0] / sss
        A[1] = A[1] / sss
        A[2] = A[2] / sss
        A[3] = A[3] / sss
        A[4] = A[4] / sss
        B[0] = r * A[0] + (1 - r) * (
                    HH[0][0] * A[0] + HH[0][1] * A[1] + HH[0][2] * A[2] + HH[0][3] * A[3] + HH[0][4] * A[4])
        B[1] = r * A[1] + (1 - r) * (
                    HH[1][0] * A[0] + HH[1][1] * A[1] + HH[1][2] * A[2] + HH[1][3] * A[3] + HH[1][4] * A[4])
        B[2] = r * A[2] + (1 - r) * (
                    HH[2][0] * A[0] + HH[2][1] * A[1] + HH[2][2] * A[2] + HH[2][3] * A[3] + HH[2][4] * A[4])
        B[3] = r * A[3] + (1 - r) * (
                    HH[3][0] * A[0] + HH[3][1] * A[1] + HH[3][2] * A[2] + HH[3][3] * A[3] + HH[3][4] * A[4])
        B[4] = r * A[4] + (1 - r) * (
                    HH[4][0] * A[0] + HH[4][1] * A[1] + HH[4][2] * A[2] + HH[4][3] * A[3] + HH[4][4] * A[4])
        sss = B[0] + B[1] + B[2] + B[3] + B[4]
        B[0] = B[0] / sss
        B[1] = B[1] / sss
        B[2] = B[2] / sss
        B[3] = B[3] / sss
        B[4] = B[4] / sss

        # A = np.add(np.multiply(r, B), np.multiply((1 - r), np.matmul(HH, B)))
        # A = np.divide(A, np.sum(A))
        # B = np.add(np.multiply(r, A), np.multiply((1 - r), np.matmul(HH, A)))
        # B = np.divide(B, np.sum(B))

        kkk = np.sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) + (A[2] - B[2]) * (A[2] - B[2]) + (
                    A[3] - B[3]) * (A[3] - B[3]) + (A[4] - B[4]) * (A[4] - B[4]))
        # kkk = np.sqrt(np.sum(np.square(A - B)))


    print("weight...............is", B[0], B[1], B[2], B[3], B[4])


    pre1 = pre1 / pre_sum
    pre2 = pre2 / pre_sum
    pre3 = pre3 / pre_sum
    pre4 = pre4 / pre_sum
    pre5 = pre5 / pre_sum
    print(pre1, pre2, pre3, pre4, pre5)
    distance_Entropy = distance1 * pre1 + distance2 * pre2 + distance3 * pre3 + distance4 * pre4 + distance5 * pre5
    sub_distance = sorted(distance_Entropy)
    listc_Entropy = []
    for t in range(len(sub_distance)):
        for h in range(len(distance_Entropy)):
            if (sub_distance[t] == distance_Entropy[h]) and (not database[h] in listc_Entropy):
                listc_Entropy.append(database[h])
    # ac_Entropy = mean_average_precison(ac_label, listc_Entropy)
    ac_Entropy = Precision_Ratio(ac_label, listc_Entropy, find_num)


    distance = distance1 * B[0] + distance2 * B[1] + distance3 * B[2] + distance4 * B[3] + distance5 * B[4]
    sub_distance = sorted(distance)
    listc = []
    for t in range(len(sub_distance)):
        for h in range(len(distance)):
            if (sub_distance[t] == distance[h]) and (not database[h] in listc):
                listc.append(database[h])
    # ac = mean_average_precison(ac_label, listc)
    ac = Precision_Ratio(ac_label, listc, find_num)

    return ac_Entropy, ac
