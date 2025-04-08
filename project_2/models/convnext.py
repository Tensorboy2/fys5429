import torch
import torch.nn as nn
torch.manual_seed(0)

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.ln1 = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.ln2 = nn.LayerNorm(out_channels)
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        )
        self.downsample = None if stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self,x):
        identity = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.ln1(out.permute(0,2,3,1)).permute(0,3,1,2)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.ln2(out.permute(0,2,3,1)).permute(0,3,1,2)

        # out = out.flatten(1)
        # print(out.shape)
        # batch_size, channels, height, width = out.shape
        # out = out.view(batch_size, channels*height*width)
        # print(out.shape)
        # print(self.out_channels)
        out = self.mlp(out)
        # print(out.shape)
        # out = out.view_as(x)
        # out = out.view(batch_size, channels, height, width)

        if self.downsample is not None:
            identity = self.downsample(x)

        out+=identity
        out = self.gelu(out)
        return out


class ConvNeXt(nn.Module):
    def __init__(self, image_size = 128):
        super().__init__()

        '''Stem Layer:'''
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # print('stem made')
        '''Block layers:'''
        self.stage_1 = nn.Sequential(*[ConvNeXtBlock(64,64)]*3)
        # print('1 made')
        self.stage_2 = nn.Sequential(ConvNeXtBlock(64,128,stride=2),*[ConvNeXtBlock(128,128)]*3)
        # print('2 made')
        self.stage_3 = nn.Sequential(ConvNeXtBlock(128,256,stride=2),*[ConvNeXtBlock(256,256)]*9)
        # print('3 made')
        self.stage_4 = nn.Sequential(ConvNeXtBlock(256,512,stride=2),*[ConvNeXtBlock(512,512)]*3)
        # print('4 made')

        '''Dummy forward pass to compute output size'''
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size, image_size)
            dummy_output = self.stem(dummy_input)
            dummy_output = self.stage_1(dummy_output)
            dummy_output = self.stage_2(dummy_output)
            dummy_output = self.stage_3(dummy_output)
            dummy_output = self.stage_4(dummy_output)
            dummy_output = nn.AdaptiveAvgPool2d((1, 1))(dummy_output)
            flattened_size = dummy_output.numel()
        
        '''Out layer'''
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Sequential(nn.Linear(flattened_size,1),nn.Dropout(p=0.2))

    def forward(self,x):
        B,C,M,N = x.shape

        # Stem layer:
        # print(x.shape)
        x = self.stem(x)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.out(x)
        # x = nn.ReLU(inplace=True)(x)
        return x

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ConvNeXt().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())