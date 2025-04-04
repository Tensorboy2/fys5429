import torch
import torch.nn as nn
torch.manual_seed(0)


class BestNet(nn.Module):
    def __init__(self, image_size = 128):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,10,5,2,0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(10,20,7,1,0),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(20,40,5,1,0),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        '''Dummy forward pass to compute output size'''
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size, image_size)
            dummy_output = self.layer_1(dummy_input)
            dummy_output = self.layer_2(dummy_output)
            dummy_output = self.layer_3(dummy_output)
            flattened_size = dummy_output.numel()

        self.out = nn.Sequential(
            nn.Linear(flattened_size,flattened_size//2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(flattened_size//2,flattened_size//4),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(flattened_size//4,1),

        )

    def forward(self,x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out.view(x.size(0), -1)
        out = self.out(out)
        return out

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = BestNet().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())