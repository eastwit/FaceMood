import torch
import torch.nn as nn

class XceptionModule(nn.Module):
    """ 实现代码中重复的 Module 结构 """
    def __init__(self, in_channels, out_channels):
        super(XceptionModule, self).__init__()
        
        # 残差分支 (1x1 Conv with stride 2)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 主分支
        self.conv_block = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv_block(x)
        return x + residual # layers.add([x, residual])


class MiniXception(nn.Module):
    def __init__(self, input_shape, num_classes, l2_regularization=0.01):
        super(MiniXception, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # --- Modules ---
        self.module1 = XceptionModule(8, 16)
        self.module2 = XceptionModule(16, 32)
        self.module3 = XceptionModule(32, 64)
        self.module4 = XceptionModule(64, 128)

        # --- Output ---
        self.conv_out = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.base(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        
        x = self.conv_out(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        return x

class SeparableConv2d(nn.Module):
    """ 实现深度可分离卷积 """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


