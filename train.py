import mindspore
from mindspore import context, nn, Model, LossMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.numpy as mnp
from models.yolo import Yolo1, YoloLoss
from models.dataset import create_dataset

# 检查点
ckpt_save_dir = "checkpoint"
load_checkpoint = "yolo_4-1_399.ckpt"
param_dict = mindspore.load_checkpoint(f"{ckpt_save_dir}/{load_checkpoint}")

# 网络
net = Yolo1()
param_not_load, _ = mindspore.load_param_into_net(net, param_dict)
print(f"params not loaded:{param_not_load}")

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
epoch = 1
model = Model(net, criterion, opt)
model.fit(epoch, train_dataset=dataset, callbacks=[
    ModelCheckpoint(prefix="yolo", directory="./checkpoint", config=config),
    LossMonitor(1)
])

# 训练
model.train(epoch, dataset)
print("Training completed.")
