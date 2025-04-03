import torch
import torch.nn as nn
torch.manual_seed(0)


class GraczykNet(nn.Module):
    def __init__(self, image_size = 128):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,10,10,1,0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2,2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(10,20,7,1,0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2,2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(20,40,5,1,0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2,2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(40,80,3,1,0,bias=False),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(80),
            nn.MaxPool2d(2,2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(80,160,2,1,0,bias=False),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(160),
            nn.MaxPool2d(2,2)
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(160,400,2,1,0,bias=False),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        '''Dummy forward pass to compute output size'''
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size, image_size)
            dummy_output = self.layer_1(dummy_input)
            # print(dummy_output.shape)
            dummy_output = self.layer_2(dummy_output)
            # print(dummy_output.shape)
            dummy_output = self.layer_3(dummy_output)
            # print(dummy_output.shape)
            dummy_output = self.layer_4(dummy_output)
            # print(dummy_output.shape)
            dummy_output = self.layer_5(dummy_output)
            # print(dummy_output.shape)
            # dummy_output = self.layer_6(dummy_output)
            # print(dummy_output.shape)
            flattened_size = dummy_output.numel()
        
        self.fc1 = nn.Linear(flattened_size,10)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(10,1)

        

    def forward(self,x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        # out = self.layer_6(out)
        out = out.flatten(1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = GraczykNet().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())