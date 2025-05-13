import torch
import torch.nn as nn
torch.manual_seed(0)


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
        x = self.dwconv(x) # (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
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


class ConvNeXt(nn.Module):
    def __init__(self,dims = [96, 192, 384, 768],depths = [3, 3, 9, 3], in_channels=1, num_classes=4):
        super().__init__()
        
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

def ConvNeXtTiny():
    return ConvNeXt(dims = [96, 192, 384, 768],depths = [3, 3, 9, 3])

def ConvNeXtSmall():
    return ConvNeXt(dims = [96, 192, 384, 768],depths = [3, 3, 27, 3])

def ConvNeXtXL():
    return ConvNeXt(dims = [256, 512, 1024, 2048],depths = [3, 3, 27, 3])

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ConvNeXtXL().to(device)
    print(model)
    print(model(x).cpu().detach().numpy())