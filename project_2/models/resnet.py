import torch
import torch.nn as nn
torch.manual_seed(0)

class ResNetBlock(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid_channels = channels // expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ResNetDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.downsample(x))

class ResNet(nn.Module):
    def __init__(self, depth=[3,4,6,3], width=[64,256,512,1024,2048], num_classes=1, input_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width[0], kernel_size=9, stride=2, padding=4, padding_mode="circular", bias=False),
            nn.BatchNorm2d(width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Define layers for ResNet-50
        self.stage1 = self._make_stage(width[0], width[1], num_blocks=depth[0], first_stride=1)
        self.stage2 = self._make_stage(width[1], width[2], num_blocks=depth[1])
        self.stage3 = self._make_stage(width[2], width[3], num_blocks=depth[2])
        self.stage4 = self._make_stage(width[3], width[4], num_blocks=depth[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(width[4], num_classes))

    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride=2):
        layers = [ResNetDownsampleBlock(in_channels, out_channels, stride=first_stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

import torch
import torch.nn as nn

class PreActResNetBlock(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid_channels = channels // expansion
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        return x + out  # No activation here

class PreActResNetDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = self.downsample(self.relu1(self.bn1(x)))

        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))

        return identity + out  # No activation here

class ResNetV2(nn.Module):
    def __init__(self, depth=[3,4,6,3], width=[64,256,512,1024,2048], num_classes=1, input_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width[0], kernel_size=9, stride=2, padding=4, padding_mode="circular", bias=False),
            nn.BatchNorm2d(width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = self._make_stage(width[0], width[1], num_blocks=depth[0], first_stride=1)
        self.stage2 = self._make_stage(width[1], width[2], num_blocks=depth[1])
        self.stage3 = self._make_stage(width[2], width[3], num_blocks=depth[2])
        self.stage4 = self._make_stage(width[3], width[4], num_blocks=depth[3])

        self.bn_last = nn.BatchNorm2d(width[4])
        self.relu_last = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width[4], num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride=2):
        layers = [PreActResNetDownsampleBlock(in_channels, out_channels, stride=first_stride)]
        for _ in range(num_blocks - 1):
            layers.append(PreActResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.relu_last(self.bn_last(x))  # Final activation
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def ResNet50V2():
    return ResNetV2(depth=[3,4,6,3], width=[64,256,512,1024,2048])

def ResNet50():
    return ResNet(depth=[3,4,6,3], width=[64,256,512,1024,2048])

def ResNet101():
    return ResNet(depth=[3,4,23,3], width=[64,256,512,1024,2048])


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ResNet50().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())

