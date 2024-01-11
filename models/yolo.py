import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore.numpy as mnp

class Yolo1(nn.Cell):
    def __init__(self):
        super(Yolo1, self).__init__()

        # 第一层
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=7, stride=2, pad_mode="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        )

        # 第二层
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, pad_mode="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        )

        # 第三层
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, pad_mode="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        )

        # 第四层
        self.conv4 = nn.SequentialCell(
            *[nn.SequentialCell(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, pad_mode="same")
            ) for _ in range(4)],
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        )

        # 第五层
        self.conv5 = nn.SequentialCell(
            *[nn.SequentialCell(
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, pad_mode="same"),
            ) for _ in range(2)],
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, pad_mode="same"),
            nn.LeakyReLU()
        )

        # 第六层
        self.conv6 = nn.SequentialCell(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.LeakyReLU()
        )

        # 展平层，为全连接层做准备
        self.flatten = nn.Flatten()

        # 第七层：全连接层
        self.fc1 = nn.SequentialCell(
            nn.Dense(in_channels=7*7*1024, out_channels=4096, activation="sigmoid")
        )

        # 第八层：全连接层
        self.fc2 = nn.SequentialCell(
            nn.Dense(in_channels=4096, out_channels=14*7*7, activation="sigmoid")
        )

        # 重整层：将输出的全连接层转化为7x7并有30个通道
        self.reshape = ops.Reshape()

    def construct(self, x):
        # 第一层
        x = self.conv1(x)

        # 第二层
        x = self.conv2(x)

        # 第三层
        x = self.conv3(x)
        # 第四层
        x = self.conv4(x)

        # 第五层
        x = self.conv5(x)

        # 第六层
        x = self.conv6(x)

        # 第七层：全连接层
        x = self.flatten(x)
        x = self.fc1(x)

        # 第八层：全连接层
        x = self.fc2(x)
        x = self.reshape(x, (7, 7, 14))
        return x


class YoloLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(YoloLoss, self).__init__(reduction)
        self.abs = ops.Abs()

    def compute_iou(self, box1: Tensor, box2: Tensor):  # box1(2,4)  box2(1,4)
        center_x_1, center_y_1, w_1, h_1 = box1
        center_x_2, center_y_2, w_2, h_2 = box2
        area1 = w_1 * h_1
        area2 = w_2 * h_2

        delta_x = np.abs(center_x_1 - center_x_2)
        delta_y = np.abs(center_y_1 - center_y_2)

        if delta_x > w_1 + w_2 or delta_y > h_1 + h_2:
            return 0

        inter_x = w_1 + w_2 - delta_x
        inter_y = h_1 + h_2 - delta_y
        inter = inter_x * inter_y

        iou = inter / (area1 + area2 - inter)
        return iou

    def construct(self, predict: Tensor, target: Tensor):
        S = 7
        lambda_coord = 5
        lambda_noobj = 0.5
        loss = 0
        for i in range(7):
            for j in range(7):
                for label in range(target.shape[0]):
                    rect = target[label][1:]
                    # 当前单元格的张量输出
                    predict_tensor = predict[i][j]  # 长度为30的张量
                    # 预测框1
                    pred_rect_1 = predict_tensor[:4]
                    conf_1 = predict_tensor[4]
                    # 预测框2
                    pred_rect_2 = predict_tensor[5:][:4]
                    conf_2 = predict_tensor[9]

                    iou_1 = self.compute_iou(rect, pred_rect_1)
                    iou_2 = self.compute_iou(rect, pred_rect_2)

                    center_x, center_y, w, h = rect
                    if iou_1 > iou_2:
                        # 1 是正样本、2 是负样本
                        pred_center_x, pred_center_y, pred_w, pred_h = pred_rect_1
                        # 定位损失：中心
                        loss += lambda_coord * (mnp.square(center_x - pred_center_x) + mnp.square(center_y - pred_center_y))
                        # 定位损失：宽高
                        loss += lambda_coord * (mnp.square(mnp.sqrt(w) - mnp.sqrt(pred_w)) + mnp.square(mnp.sqrt(h) - mnp.sqrt(pred_h)))
                        # 置信度损失
                        loss += mnp.square(conf_1 - iou_1)
                        loss += lambda_noobj * mnp.square(conf_2)
                    else:
                        # 2 是正样本、1 是负样本
                        pred_center_x, pred_center_y, pred_w, pred_h = pred_rect_2
                        # 定位损失：中心
                        loss += lambda_coord * (
                                    mnp.square(center_x - pred_center_x) + mnp.square(center_y - pred_center_y))
                        # 定位损失：宽高
                        loss += lambda_coord * (mnp.square(mnp.sqrt(w) - mnp.sqrt(pred_w)) + mnp.square(
                            mnp.sqrt(h) - mnp.sqrt(pred_h)))
                        # 置信度损失
                        loss += mnp.square(conf_2 - iou_2)
                        loss += lambda_noobj * mnp.square(conf_1)

        # 类别损失
        for label in range(target.shape[0]):
            type = [0, 0, 0, 0]
            type[int(target[label][0])] = 1
            type = mnp.array(type)

            center_x, center_y = target[label][1:][:2]
            grid_x = np.floor(S * center_x)
            grid_y = np.floor(S * center_y)

            pred_type = predict[grid_x][grid_y][10:]

            delta = mnp.square(type - pred_type)
            loss += mnp.sum(delta)
        return loss
