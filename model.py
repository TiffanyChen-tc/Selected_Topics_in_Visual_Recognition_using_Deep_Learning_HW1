import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SEBlock(nn.Module):
    """通道維度的 Squeeze-and-Excitation 注意力機制。"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        s = torch.sigmoid(self.fc2(self.relu(self.fc1(s)))).view(b, c, 1, 1)
        return x * s


class MyBottleneck(nn.Module):
    """自定義 Bottleneck，在殘差相加後整合 SEBlock。"""

    def __init__(self, original_block, se_reduction=16):
        super(MyBottleneck, self).__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.relu
        self.downsample = original_block.downsample
        self.stride = original_block.stride
        self.se_extra = SEBlock(original_block.conv3.out_channels, reduction=se_reduction)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return self.se_extra(out)


class GeMPooling(nn.Module):
    """Generalized Mean Pooling，提升細粒度辨識能力的池化層。"""

    def __init__(self, p_init=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), output_size=1
        ).pow(1. / self.p).flatten(1)


class PMGConvBlock(nn.Module):
    """PMG 框架中用於特徵降維的卷積塊。"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(PMGConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PMGClassifier(nn.Module):
    """用於單一粒度特徵的分類頭，含 Dropout 正則化。"""

    def __init__(self, in_features, feature_size, num_classes, dropout=0.3):
        super(PMGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class VRModel(nn.Module):
    """視覺辨識模型主體，結合 ResNeXt-101 與 PMG 多尺度結構。"""

    def __init__(self, num_classes=100, feature_size=512):
        super(VRModel, self).__init__()
        base = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = nn.Sequential(*[MyBottleneck(b) for b in base.layer4])

        self.conv_block1 = PMGConvBlock(512, feature_size, feature_size)
        self.conv_block2 = PMGConvBlock(1024, feature_size, feature_size)
        self.conv_block3 = PMGConvBlock(2048, feature_size, feature_size)

        self.pool = GeMPooling()

        self.classifier1 = PMGClassifier(feature_size, feature_size, num_classes)
        self.classifier2 = PMGClassifier(feature_size, feature_size, num_classes)
        self.classifier3 = PMGClassifier(feature_size, feature_size, num_classes)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(feature_size * 3),
            nn.Linear(feature_size * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_size, num_classes),
        )
        self._init_new_layers()

    def _init_new_layers(self):
        """初始化自定義層的權重。"""
        new_modules = [
            self.conv_block1, self.conv_block2, self.conv_block3,
            self.classifier1, self.classifier2, self.classifier3,
            self.classifier_concat,
        ]
        for mod in new_modules:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        nn.init.uniform_(m.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.stem(x)
        f = self.layer1(f)
        f2 = self.layer2(f)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        xl1 = self.pool(self.conv_block1(f2))
        xc1 = self.classifier1(xl1)

        xl2 = self.pool(self.conv_block2(f3))
        xc2 = self.classifier2(xl2)

        xl3 = self.pool(self.conv_block3(f4))
        xc3 = self.classifier3(xl3)

        x_concat = self.classifier_concat(torch.cat([xl1, xl2, xl3], dim=1))
        return xc1, xc2, xc3, x_concat


def get_model():
    """外部調用接口，返回模型實例。"""
    return VRModel(num_classes=100)