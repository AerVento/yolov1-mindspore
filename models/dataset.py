import os
import cv2
import mindspore
import mindspore.numpy as mnp
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset


def generator(images_path, labels_path):
    # 枚举所有的图片
    for img_name in os.listdir(images_path):
        img_name = img_name[:-4]  # 移除后缀.jpg
        # 读取图像数据
        image_path = os.path.join(images_path, img_name + ".jpg")
        image = cv2.imread(image_path)
        # 图像预处理：调整大小和归一化
        image = cv2.resize(image, (448, 448))
        image = image.astype(np.float32) / 255.0
        image_tensor = Tensor(image, mindspore.float32)
        image_tensor = mnp.transpose(image_tensor, (2, 1, 0))  # 将张量通道从(1440, 2560, 3)改为(3, 2560, 1440)
        image_tensor = mnp.expand_dims(image_tensor, axis=0)
        # 读取标签数据
        label_path = os.path.join(labels_path, img_name + ".txt")
        label = []
        if os.path.exists(label_path):  # 如果存在标签文件，就读取添加
            with open(label_path, "r") as label_file:
                for label_line in label_file:
                    pair = label_line.strip().split(" ")
                    # 一个标注框
                    rect_info = [float(pair[i]) for i in range(5)]
                    # 转为Tensor，并加入总标签
                    label.append(mnp.array(rect_info))

        label_tensor = mnp.array(label)
        yield image_tensor, label_tensor


def create_dataset(dir_path):
    images_path = os.path.join(dir_path, "images")
    labels_path = os.path.join(dir_path, "labels")

    # 读取所有的labels列表
    classes = []
    with open(os.path.join(labels_path, "classes.txt"), "r") as file:
        # 逐行读取文件内容
        for line in file:
            classes.append(line.strip())

    # 数据集迭代器
    dataset = GeneratorDataset(source=list(generator(images_path, labels_path)), column_names=["data", "label"])
    return dataset, classes
