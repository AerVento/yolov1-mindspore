import os
import cv2
import mindspore
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor


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
    def generator():
        # 枚举所有的图片
        for img_name in os.listdir(images_path):
            img_name = img_name[:-4]  # 移除后缀.jpg
            # 读取图像数据
            image_path = os.path.join(images_path, img_name + ".jpg")
            image = cv2.imread(image_path)
            # 图像预处理：调整大小和归一化
            image = cv2.resize(image, (2560, 1440))
            image = image.astype(np.float32) / 255.0
            image_tensor = Tensor(image, mindspore.float32)
            # 读取标签数据
            label_path = os.path.join(labels_path, img_name + ".txt")
            label = []
            if os.path.exists(label_path):  # 如果存在标签文件，就读取添加
                with open(label_path, "r") as label_file:
                    for label_line in label_file:
                        pair = label_line.strip().split(" ")
                        label.append({
                            "type": int(pair[0]),
                            "rect": [float(pair[1]), float(pair[2]), float(pair[3]), float(pair[4])]
                        })

            yield image_tensor, label

    dataset = ds.GeneratorDataset(generator, column_names=["image", "label"])
    return dataset, classes
