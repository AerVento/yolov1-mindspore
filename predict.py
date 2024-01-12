import cv2
import mindspore
import numpy as np
from mindspore import Tensor
import mindspore.numpy as mnp
from models.yolo import Yolo1, compute_iou

# 检查点
ckpt_save_dir = "checkpoint"
load_checkpoint = "yolo_4-1_399.ckpt"
param_dict = mindspore.load_checkpoint(f"{ckpt_save_dir}/{load_checkpoint}")

# 网络
net = Yolo1()
param_not_load, _ = mindspore.load_param_into_net(net, param_dict)
print(f"params not loaded:{param_not_load}")

image_path = "data/images/img_7.jpg"
image = cv2.imread(image_path)
# 图像预处理：调整大小和归一化
image = cv2.resize(image, (448, 448))
image = image.astype(np.float32) / 255.0
image_tensor = Tensor(image, mindspore.float32)
image_tensor = mnp.transpose(image_tensor, (2, 1, 0))  # 将张量通道从(1440, 2560, 3)改为(3, 2560, 1440)
image_tensor = mnp.expand_dims(image_tensor, axis=0)
output = net(image_tensor)

# 输出处理
type_val = 0.5 # 分类的阈值
conf_val = 0.5 # 置信度阈值
iou_val = 0.6 # iou阈值
divide = [[], [], [], []]
maxes = [0, 0, 0, 0]
max_scores = [0, 0, 0, 0]
for i in range(7):
    for j in range(7):
        tensor = output[i][j]
        types = tensor[10:]

        detected = -1
        for k in range(4):
            if types[k] > type_val:
                detected = k


        if detected == -1:
            continue

        rect_1 = tensor[:4]
        conf_1 = tensor[4:][:1]
        rect_2 = tensor[5:][:4]
        conf_2 = tensor[9:][:1]

        if conf_1 > max_scores[detected]:
            max_scores[detected] = conf_1
            maxes[detected] = rect_1

        if conf_2 > max_scores[detected]:
            max_scores[detected] = conf_2
            maxes[detected] = rect_2

        divide[detected].append((i,j))

for k in range(4):
    for i, j in divide[k]:
        tensor = output[i][j]
        rect_1 = tensor[:4]
        rect_2 = tensor[5:][:4]

        iou_1 = compute_iou(rect_1, maxes[k])
        iou_2 = compute_iou(rect_1, maxes[k])
        if iou_1 > iou_val or iou_2 > iou_val:
            divide[k].remove((i, j))

for k in range(4):
    print(f"type {k}:\n")
    for i, j in divide[k]:
        tensor = output[i][j]
        rect_1 = tensor[:4]
        rect_2 = tensor[5:][:4]
        print(f"{rect_1}\n")
        print(f"{rect_2}\n")

