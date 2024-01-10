import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops


class Yolo1(nn.Cell):
    def __init__(self):
        super(Yolo1, self).__init__()
        # 激活函数
        self.leaky_relu = nn.LeakyReLU()
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=7, stride=2, pad_mode="same")

        # 第二层
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, pad_mode="same")

        # 第三层
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, pad_mode="same")
        )

        # 第四层
        self.conv4 = nn.SequentialCell(
            *[nn.SequentialCell(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, pad_mode="same")
            ) for _ in range(4)],
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, pad_mode="same")
        )

        # 第五层
        self.conv5 = nn.SequentialCell(
            *[nn.SequentialCell(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, pad_mode="same"),
            ) for _ in range(2)],
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, pad_mode="same")
        )

        # 第六层
        self.conv6 = nn.SequentialCell(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, pad_mode="same")
        )

        # 展平层，为全连接层做准备
        self.flatten = nn.Flatten()

        # 第七层：全连接层
        self.fc1 = nn.SequentialCell(
            nn.Dense(in_channels=7*7*1024, out_channels=4096),
            self.leaky_relu
        )

        # 第八层：全连接层
        self.fc2 = nn.SequentialCell(
            nn.Dense(in_channels=4096, out_channels=7*7*30),
            self.leaky_relu
        )

        # 重整层：将输出的全连接层转化为7x7并有30个通道
        self.reshape = ops.Reshape()

    def construct(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.pool(x)

        # 第二层
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)

        # 第三层
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.pool(x)

        # 第四层
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.pool(x)

        # 第五层
        x = self.conv5(x)
        x = self.leaky_relu(x)

        # 第六层
        x = self.conv6(x)
        x = self.leaky_relu(x)

        # 第七层：全连接层
        x = self.flatten(x)
        x = self.fc1(x)

        # 第八层：全连接层
        x = self.fc2(x)
        x = self.reshape(x, (7, 7, 30))
        return x


def yolo_loss(output: Tensor, label):
    tensor_shape = output.shape
    print(tensor_shape)
