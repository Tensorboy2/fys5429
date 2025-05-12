import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyMAE(nn.Module):
    def __init__(self, embed_dim=32, patch_size=4, image_size=16, mode="mae"):
        super().__init__()
        self.mode = mode  # "mae" or "head"
        self.patchify = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Decoder for MAE
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size * patch_size)  # reconstruct pixel patch
        )

        # Head for classification/regression
        self.head = nn.Linear(embed_dim, 1)

        # Set mode
        if mode == "mae":
            self.forward = self.forward_mae
        elif mode == "head":
            self.forward = self.forward_head
        else:
            raise ValueError("Unknown mode")

    def forward_mae(self, x):
        patches = self.patchify(x)  # (B, C, H/patch, W/patch)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, C)
        encoded = self.encoder(patches)
        recon = self.decoder(encoded)  # (B, N, patch*patch)
        recon = recon.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 1, H, W)
        return recon

    def forward_head(self, x):
        patches = self.patchify(x)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        encoded = self.encoder(patches)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)

# Toy image data: 4 samples of 1x16x16 "images"
x = torch.randn(4, 1, 16, 16)

# Pretraining with MAE mode
model = ToyMAE(mode="mae")
# print(model)
recon = model(x)
print("MAE reconstruction shape:", recon.shape)

# Save pretrained encoder weights
temp = model.state_dict()

# Reload in head mode
finetune_model = ToyMAE(mode="head")
finetune_model.load_state_dict(temp)
# print(finetune_model)
out = finetune_model(x)
print("Finetune head output shape:", out.shape)
