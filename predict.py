import cv2
import mindspore
import numpy as np
from mindspore import Tensor
import mindspore.numpy as mnp
from models.yolo import Yolo1, compute_iou

classes = [
    'T_body',
    'T_head',
    'CT_body',
    'CT_head'
]

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
conf_thres = 0.38  # 置信度阈值
iou_thres = 0.01  # iou阈值

# 所有置信度达到阈值的框
total = []
for i in range(7):
    for j in range(7):
        tensor = output[i][j]
        rect_1 = tensor[:4]
        conf_1 = tensor[4:][:1]
        rect_2 = tensor[5:][:4]
        conf_2 = tensor[9:][:1]
        types: Tensor = tensor[10:]

        if conf_1 > conf_2 and conf_1 > conf_thres:
            # 某一个类别的总概率
            types_pr = types * conf_1
            total.append((rect_1, conf_1, types_pr))
        if conf_2 > conf_1 and conf_2 > conf_thres:
            types_pr = types * conf_2
            total.append((rect_2, conf_2, types_pr))

total = sorted(total, key=lambda x: x[1], reverse=True)  # 按置信度从大到小排序
filtered = []  # 经过非极大化抑制之后剩下来的框
while len(total) > 0:
    # 将置信度最大的元素放进去
    max_elem = total.pop(0)
    rect_max, conf_max, types_max = max_elem
    filtered.append((rect_max, types_max))
    # 移除交并比过高的元素
    i = 0
    while i < len(total):
        rect, conf, types = total[i]
        iou = compute_iou(rect_max, rect)

        # 大于iou阈值，移除
        if iou > iou_thres:
            total.pop(i)
        i += 1

for rect, types in filtered:
    index = -1
    for i in range(types.size):
        if index == -1 or types[i] > types[index]:
            index = i

    print(f"rect: {rect}, type: {classes[index]}")


