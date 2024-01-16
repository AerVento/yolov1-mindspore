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
# 损失函数
criterion = YoloLoss()
# 测试数据集
dataset, classes = create_dataset("data")
size = dataset.get_dataset_size()
print(f"{size} images collected.")

i = 0
for image, label in dataset.create_tuple_iterator():
    loss = criterion(net(image), label)
    print(f"image_{i}: loss = {loss}")
    i += 1

print("Validate completed.")
