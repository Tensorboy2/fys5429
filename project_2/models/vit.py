"""Module for Vision Transformer"""
import torch
import torch.nn as nn


"""
Plan: Keep the class modular with subclasses for things such as the attention layer, the patchify layer and the mlp layer
"""

class Attention(nn.Module):
    """
    Standard Self attention layer
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads ==0, "Embedding dimension must be divisible by number of heads."

        self. scale = self.head_dim ** -0.5

        self.qvk = nn.Linear(embed_dim,embed_dim*3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)


    def forward(self,x):
        B, N, C = x.shape # batch, num_patches, embed_dim

        qvk = self.qvk(x) # B, N, 3 * C
        qvk = qvk.reshape(B,N,3,self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q,k,v = qvk[0], qvk[1], qvk[2] # (B, heads, N, head_dim), (B, heads, N, head_dim), (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2,-1)) * self.scale # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v # (B, heads, N, head_dim)
        out = out.transpose(1,2).reshape(B,N,C) 
        out = self.proj(out)
        return self.proj_drop(out)
    

class MLP(nn.Module):
    """
    Standard multilayer perceptron layer
    """
    def __init__(self,embed_dim, hidden_size, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_linear = nn.Linear(embed_dim,hidden_size)
        self.gelu = nn.GELU()
        self.out_linear = nn.Linear(hidden_size,embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        x = self.in_linear(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.out_linear(x)
        x = self.drop(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim,num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim*mlp_ratio)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x)) 
        return x

class Patchify(nn.Module):
    def __init__(self, in_channels=1, 
                 embed_dim=128, 
                 patch_size=16, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.proj =  nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=patch_size,
                      stride=patch_size,
                      bias=False,
                      padding=0)
    def forward(self, x):
        # x has shape (B,C,H,W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class ViT(nn.Module):
    """
    Vision transformer class.
    """
    def __init__(self, image_size, patch_size, embed_dim, depth,
                 num_heads, mlp_ration,
                  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stem = Patchify(in_channels=1,
                             embed_dim=embed_dim,
                             patch_size=patch_size)

        self.layers = self._make_stage(depth, embed_dim, num_heads, mlp_ration)

        self.out = nn.Linear(embed_dim,1)

    def _make_stage(self,depth,embed_dim, num_heads, mlp_ration):
        return nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ration) for _ in range(depth)]
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layers(x)
        print(x.shape)
        x = self.out(x)
        return x
    

if __name__ == "__main__":
    x = torch.rand((2,1,128,128))
    model = ViT(128,16,512,12,12,6)
    output = model.forward(x)
    print(output.squeeze())