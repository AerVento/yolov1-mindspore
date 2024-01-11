import mindspore
from mindspore import context, nn, Model, LossMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.numpy as mnp
from models.yolo import Yolo1, YoloLoss
from models.dataset import create_dataset
# mindspore.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 检查点
ckpt_save_dir = "checkpoint"

# 网络
net = Yolo1()

# 优化器
opt = nn.Adam(net.trainable_params())

# 损失函数
criterion = YoloLoss()

# 数据集
dataset, classes = create_dataset("data")
size = dataset.get_dataset_size()
print(f"{size} images collected.")

# 检查点
config = CheckpointConfig(save_checkpoint_steps=size)

# 模型
model = Model(net, criterion, opt)
model.fit(1, train_dataset=dataset, callbacks=[
    ModelCheckpoint(prefix="yolo", directory="./checkpoint", config=config),
    LossMonitor(1)
])

# 训练
epoch = 2
model.train(epoch, dataset)

# for _ in range(epoch):
#     it = dataset.create_tuple_iterator()
#     i = 0
#     for data in it:
#         image, label = data
#         print(f"{i}:{image.shape}")
#         image = mnp.expand_dims(image, axis=0)
#         print(f"{i}:{image.shape} {net(image)}")
#         i += 1
#     break
