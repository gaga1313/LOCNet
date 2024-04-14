import torch
import torch.nn as nn
import torchvision

import timm
from timm.models._registry import register_model

from src.models.locnet import Bridge, UpBlockForUNetWithResNet50, UNetWithResnet50Encoder


class AutoEncoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=1):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(weights=None)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048, 16)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=128,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
        )
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=64,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        x = self.input_block(x)
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue

        cls_pred = self.avg_pool(x)
        cls_pred = torch.flatten(cls_pred, start_dim=1)
        cls_pred = self.dropout(cls_pred)
        cls_pred = self.classifier(cls_pred)

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            x = block(x, None)
        output_feature_map = x
        x = self.out(x)
        if with_output_feature_map:
            return x, cls_pred, output_feature_map
        else:
            return x, cls_pred


__all__ = []


@register_model
def autoencoder(pretrained=False, **kwargs):
    model = AutoEncoder(n_classes=1)
    return model
