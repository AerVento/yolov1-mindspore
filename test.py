import numpy as np
from mindspore import nn, ops
from mindspore.train import Model, LossMonitor
from mindspore.dataset import GeneratorDataset


def get_data(num, w=2.0, b=3.0):
    """生成数据及对应标签"""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16):
    """加载数据集"""
    dataset = GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset


class MAELoss(nn.LossBase):
    """自定义损失函数MAELoss"""
    def construct(self, base, target):
        x = ops.abs(base - target)
        return self.get_loss(x)  # 返回loss均值


train_dataset = create_dataset(num_data=160)
network = nn.Dense(1, 1)
loss_fn = MAELoss()
optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.005, momentum=0.9)

# 使用model接口将网络、损失函数和优化器关联起来
model = Model(network, loss_fn, optimizer)
model.train(10, train_dataset, callbacks=[LossMonitor(10)])