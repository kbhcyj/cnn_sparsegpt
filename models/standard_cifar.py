from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16_bn, ResNet18_Weights, VGG16_BN_Weights

class ResNet18_CIFAR(nn.Module):
    """
    CIFAR-10 (32x32) 이미지에 맞게 수정된 ResNet-18 모델.
    Standard ResNet은 ImageNet(224x224)용이라 CIFAR-10에서는 정보 손실이 큼.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # Pretrained weights 로드 (ImageNet 기준이지만 전이 학습용으로 사용)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)
        
        # 1. 첫 번째 Conv 레이어 수정
        # ImageNet: 7x7 kernel, stride 2, padding 3
        # CIFAR-10: 3x3 kernel, stride 1, padding 1 (이미지가 작으므로)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Pretrained 가중치가 있다면 3x3 중심부 등으로 초기화 시도 가능하나,
        # 여기서는 단순 구조 변경만 수행하고 초기화는 랜덤 혹은 별도 학습에 맡김.
        # (단, 사용자가 pretrained=True를 원할 경우 차원 불일치로 conv1은 초기화됨)
        
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        # 2. MaxPool 제거 (CIFAR-10은 이미지가 작아서 초반 pooling 제외)
        self.maxpool = nn.Identity() 
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGG16_CIFAR(nn.Module):
    """
    CIFAR-10 (32x32) 이미지에 맞게 수정된 VGG-16 (Batch Norm 포함) 모델.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = vgg16_bn(weights=weights)
        
        self.features = base_model.features
        
        # Original VGG has AdaptiveAvgPool2d((7, 7)) -> leads to 512*7*7 inputs
        # For CIFAR (32x32), final feature map is 1x1. We enforce 1x1 output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # CIFAR-10 32x32 이미지가 VGG features 통과 후 크기: 512 x 1 x 1
        # 따라서 classifier 입력 차원이 512 * 7 * 7 (ImageNet)이 아닌 512임.
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

