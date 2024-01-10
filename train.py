from mindspore import context, nn

from models.yolo import Yolo1
from models.dataset import create_dataset

# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 网络
net = Yolo1()

# 优化器
opt = nn.Adam(net.trainable_params())

# 损失函数
loss = None

# 数据集
batch = 8
dataset, classes = create_dataset("data")

# 训练
epoch = 200
for _ in range(epoch):
    iter = dataset.create_dict_iterator(output_numpy=True)
    for _ in range(batch):
        for data in iter:
            image_data = data
            label_data = data['label']

