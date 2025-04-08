import torch
import torch.nn as nn
torch.manual_seed(0)

# class ConvNeXtBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=7,stride=stride,padding=3,bias=False)
#         self.ln1 = nn.LayerNorm(out_channels,eps=1e-6)
#         self.bn = nn.BatchNorm2d(out_channels)

#         self.leakyrelu = nn.LeakyReLU()
#         # self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
#         # self.ln2 = nn.LayerNorm(out_channels)
#         # self.out_channels = out_channels
#         self.mlp = nn.Sequential(
#             nn.Linear(out_channels, out_channels * 4),
#             nn.LeakyReLU(),
#             nn.Linear(out_channels * 4, out_channels)
#         )
#         self.downsample = None if stride == 1 else nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_channels))

#     def forward(self,x):
#         identity = x
#         # print(x.shape)
#         out = self.conv1(x)
#         # out = out.permute(0,2,3,1)
#         out = self.bn(out)
#         out = out.permute(0,2,3,1)
#         # out = self.gelu(out)
#         out = self.mlp(out)
#         out = out.permute(0,3,1,2)
#         # out = self.conv2(out)
#         # out = self.ln2(out.permute(0,2,3,1)).permute(0,3,1,2)

#         # out = out.flatten(1)
#         # print(out.shape)
#         # batch_size, channels, height, width = out.shape
#         # out = out.view(batch_size, channels*height*width)
#         # print(out.shape)
#         # print(self.out_channels)
#         # print(out.shape)
#         # out = out.view_as(x)
#         # out = out.view(batch_size, channels, height, width)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out+=identity
#         out = self.leakyrelu(out)
#         return out


# class ConvNeXt(nn.Module):
#     def __init__(self, image_size = 128):
#         super().__init__()

#         '''Stem Layer:'''
#         self.stem = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # print('stem made')
#         '''Block layers:'''
#         self.stage_1 = nn.Sequential(*[ConvNeXtBlock(64,64)]*3)
#         # print('1 made')
#         self.stage_2 = nn.Sequential(ConvNeXtBlock(64,128,stride=2),*[ConvNeXtBlock(128,128)]*3)
#         # print('2 made')
#         self.stage_3 = nn.Sequential(ConvNeXtBlock(128,256,stride=2),*[ConvNeXtBlock(256,256)]*9)
#         # print('3 made')
#         self.stage_4 = nn.Sequential(ConvNeXtBlock(256,512,stride=2),*[ConvNeXtBlock(512,512)]*3)
#         # print('4 made')

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


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)                          # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                   # → (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)                   # → (B, C, H, W)
        return x + shortcut


class ConvNeXtStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
        ) if in_dim != out_dim else nn.Identity()

        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_dim) for _ in range(depth)])

    def forward(self, x):
        if not isinstance(self.downsample, nn.Identity):
            # permute before LayerNorm
            x = x.permute(0, 2, 3, 1)
            x = self.downsample[0](x)  # LayerNorm
            x = x.permute(0, 3, 1, 2)
            x = self.downsample[1](x)  # Conv2d
        return self.blocks(x)


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]  # ConvNeXt-Tiny

        # Patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True)
        )

        self.stage1 = ConvNeXtStage(dims[0], dims[0], depths[0])
        self.stage2 = ConvNeXtStage(dims[0], dims[1], depths[1])
        self.stage3 = ConvNeXtStage(dims[1], dims[2], depths[2])
        self.stage4 = ConvNeXtStage(dims[2], dims[3], depths[3])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem[0](x)               # patchify conv
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)               # layernorm
        x = x.permute(0, 3, 1, 2)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.head(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ConvNeXtTiny().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())