import torch
import torch.nn as nn
torch.manual_seed(0)


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.downsample = None
#         if in_channels != out_channels or stride != 1:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self,x):
#         identity = x
#         # print(x.shape)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out+=identity
#         out = self.relu(out)
#         return out

# class ResNet(nn.Module):
#     """
#     CNN architecture based on the ResNet
#     """
#     def __init__(self, image_size = 128):
#         super().__init__()

#         '''Stem Layer:'''
#         self.stem = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )

#         '''Block layers:'''
#         self.stage_1 = nn.Sequential(*[ResidualBlock(64, 64)] * 3)
#         self.stage_2 = nn.Sequential(ResidualBlock(64, 128, stride=2), *[ResidualBlock(128, 128)] * 4)
#         self.stage_3 = nn.Sequential(ResidualBlock(128, 256, stride=2), *[ResidualBlock(256, 256)] * 6)
#         self.stage_4 = nn.Sequential(ResidualBlock(256, 512, stride=2), *[ResidualBlock(512, 512)] * 3)


#         '''Dummy forward pass to compute output size'''
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 1, image_size, image_size)
#             dummy_output = self.stem(dummy_input)
#             dummy_output = self.stage_1(dummy_output)
#             dummy_output = self.stage_2(dummy_output)
#             dummy_output = self.stage_3(dummy_output)
#             dummy_output = self.stage_4(dummy_output)
#             dummy_output = nn.AdaptiveAvgPool2d((1, 1))(dummy_output)
#             flattened_size = dummy_output.numel()
        
#         '''Out layer'''
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.out = nn.Sequential(nn.Linear(flattened_size,1),nn.Dropout(p=0.2))


#     def forward(self,x):
#         B,C,M,N = x.shape

#         # Stem layer:
#         # print(x.shape)
#         x = self.stem(x)

#         x = self.stage_1(x)
#         x = self.stage_2(x)
#         x = self.stage_3(x)
#         x = self.stage_4(x)

#         x = self.global_avg_pool(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.out(x)
#         # x = nn.ReLU(inplace=True)(x)
#         return x



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
    def __init__(self, num_classes=1, input_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Define layers for ResNet-50
        self.stage1 = self._make_stage(64, 256, num_blocks=3, first_stride=1)
        self.stage2 = self._make_stage(256, 512, num_blocks=4)
        self.stage3 = self._make_stage(512, 1024, num_blocks=6)
        self.stage4 = self._make_stage(1024, 2048, num_blocks=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(2048, num_classes))

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


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ResNet().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())

