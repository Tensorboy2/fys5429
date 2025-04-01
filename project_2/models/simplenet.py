import torch
import torch.nn as nn
torch.manual_seed(0)


class SimpleNet(nn.Module):
    def __init__(self, image_size = 128):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(512,1024,3,1,1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer_7 = nn.Sequential(
            nn.Conv2d(1024,2048,3,1,1,bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.out = nn.Linear(2048,1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = out.flatten(1)
        out = self.out(out)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = SimpleNet().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())