import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)  # (20, 128, 8, 8)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)  # (20, 64, 4, 4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)  # (20, 32, 2, 2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # (20, 64, 4, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # (20, 128, 8, 8)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # (20, 256, 16, 16)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        return x


class AutoPromptGenerator2D(nn.Module):
    def __init__(self, image_embedding_size: Tuple[int, int], num_points: int = 3):
        super(AutoPromptGenerator2D, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.num_points = num_points
        self.image_embedding_size = image_embedding_size

    def forward(self, x):
        # 编码和解码阶段
        x = self.encoder(x)
        x = self.decoder(x)  # 输出 (batch_size, 256, 16, 16)

        # 生成 point prompt
        batch_size, _, height, width = x.shape

        # 随机选择坐标
        point_coords = torch.randint(0, height, (batch_size, self.num_points, 2), device=x.device)

        # 随机生成 point 标签，范围是 -1（无效点）, 0（负样本）, 1（正样本）
        point_labels = torch.randint(-1, 2, (batch_size, self.num_points), device=x.device)

        return point_coords, point_labels


# 测试代码
if __name__ == "__main__":
    model = AutoPromptGenerator2D(image_embedding_size=(16, 16))
    input_tensor = torch.randn(20, 256, 16, 16)  # 输入的特征图 (20, 256, 16, 16)
    point_coords, point_labels = model(input_tensor)
    print(f"point坐标: {point_coords.shape}")  # (20, 3, 2)
    print(f"point标签: {point_labels.shape}")  # (20, 3)