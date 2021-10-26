import os
from time import time
import numpy as np
import torch
import torch.nn
from PIL import Image
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

from torchvision import transforms as T





path = r"C:\Users\12055\Desktop\开始搞研究\databases\UCMerced_LandUse\UCMerced_LandUse\UCMerced_LandUse\Images\UCMerced_LandUse"
ls = []
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        ls.append(os.path.join(dirpath, filename))
##################################################################################
### AlexNet特征提取器
model = models.alexnet(pretrained=True)
feature = torch.nn.Sequential(*list(model.children()))


def make_model():
    model = models.alexnet(pretrained=True)  # 定位
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    return model


def Alex_feature(imgpath):
    model = make_model()
    TARGET_IMG_SIZE = 448
    img_to_tensor = transforms.ToTensor()
    model.eval()  # 必须要有，不然会影响特征提取结果
    img = Image.open(imgpath)  # 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    gray_img = T.Grayscale(num_output_channels=3)(img)
    # gray_img = np.array(gray_img)
    # img = np.array(img)
    # y = np.array([img, gray_img])
    tensor = img_to_tensor(img)  # 将图片转化成tensor
    gray = img_to_tensor(gray_img)
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    gray = Variable(torch.unsqueeze(gray, dim=0).float(), requires_grad=False)

    result = model(Variable(tensor))
    result_gray = model(Variable(gray))

    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    result_gray = result_gray.data.cpu().numpy()

    result = np.concatenate((result_npy[0], result_gray[0]))

    return result  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]


##############################################################################################################

if __name__ == "__main__":
    ld = []
    num = 0
    for imgpath in ls:
        num += 1
        t0 = time()
        tmp = Alex_feature(imgpath)
        ld.append(np.array(tmp))
        print(num)
    kk = np.array(ld)
    np.save("AlexNet_feature.npy", kk)
