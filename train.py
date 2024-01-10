import mindspore
from mindspore import context, nn
import mindspore.numpy as mnp
from models.yolo import Yolo1
from models.dataset import create_dataset

mindspore.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 网络
net = Yolo1()

# 优化器
opt = nn.Adam(net.trainable_params())

# 损失函数
loss = None

# 数据集
dataset, classes = create_dataset("data")

it = dataset.create_tuple_iterator()
for data in it:
    image, label = data
    print(f"{image.shape}")
    image = mnp.expand_dims(image, axis=0)
    print(f"{image.shape} {net(image)}")
    break
# epoch = 200
# for _ in range(epoch):
#     it = dataset.create_tuple_iterator()
#     i = 0
#     for data in it:
#         image, label = data
#         print(f"{i}:{image.shape}")
#         image = mnp.expand_dims(image, axis=0)
#         print(f"{i}:{image.shape} {net(image)}")
#         i += 1
#         break
#     break
