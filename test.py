import mindspore
from mindspore import Tensor
import mindspore.numpy as np
from mindspore.dataset import GeneratorDataset

from models.yolo import yolo_loss, Yolo1, LeNet5
import mindspore as ms
import cv2


# Generator
def my_generator(start, end):
    for i in range(start, end):
        yield i, "str"


# since a generator instance can be only iterated once, we need to wrap it by lambda to generate multiple instances
dataset = GeneratorDataset(source=lambda: my_generator(3, 6), column_names=["i","str"])

for data in dataset.create_tuple_iterator():
    i, string = data
    print(i)
    print(string)
