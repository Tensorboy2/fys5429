"""Module for Vision Transformer"""
import torch
import torch.nn as nn
torch.manual_seed(0)
import os
path = os.path.dirname(__file__)

class Attention(nn.Module):
    """
    Standard Self-Attention layer.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence (patches), Channels

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        out = self.proj(out)
        return self.proj_drop(out)

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Patchify(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # -> (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # -> (B, N_patches, embed_dim)
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_classes, dropout=0.0):
        super().__init__()
        self.name = ""
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2

        self.patchify = Patchify(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patchify(x)  # (B, N, embed_dim)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.head(x)

def ViT_B16(image_size=128, num_classes=4, patch_size=16, pre_trained = False):
    """
    Base ViT with 12 layers, 12 heads, 768 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = "ViT_B16"
    if pre_trained:
        weights_path = os.path.join(path, f'{model.name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_B8(image_size=128, num_classes=4, patch_size=8, pre_trained = False):
    """
    Base ViT with 12 layers, 12 heads, 768 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = "ViT_B8"
    if pre_trained:
        weights_path = os.path.join(path, f'{model.name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model
def ViT_B4(image_size=128, num_classes=4, patch_size=4, pre_trained = False):
    """
    Base ViT with 12 layers, 12 heads, 768 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = "ViT_B4"
    if pre_trained:
        weights_path = os.path.join(path, f'{model.name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_L16(image_size=128, num_classes=4, pre_trained = False):
    """
    Large ViT with 24 layers, 16 heads, 1024 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = "ViT_L16"

    if pre_trained:
        weights_path = os.path.join(path, f'{model.name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_H16(image_size=128, num_classes=4, pre_trained = False):
    """
    Huge ViT with 32 layers, 16 heads, 1280 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        num_classes=num_classes,
        mlp_ratio=4,
    )
    model.name = "ViT_H16"
    if pre_trained:
        weights_path = os.path.join(path, f'{model.name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model


if __name__ == "__main__":
    x = torch.rand((2, 1, 128, 128))
    # model = ViT(image_size=128, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
    model = ViT_B16(pre_trained=True)
    out = model(x)
    print(out.shape)  # Should be (2, 1)
    print(model(x).cpu().detach().numpy())
