import mindspore
import numpy as np
from mindspore import Tensor

from models.yolo import yolo_loss, Yolo1
import mindspore as ms
import cv2
# yolo_loss(ms.Tensor([1, 2, 3, 4, 5]), 1)

image = cv2.imread("videos/images/img_0.jpg")
# 图像预处理：调整大小和归一化
image = cv2.resize(image, (2560, 1440))
image = image.astype(np.float32) / 255.0
image_tensor = Tensor(image, mindspore.float32)

net = Yolo1()

output = net(image_tensor)

print(output)
