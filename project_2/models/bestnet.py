import torch
import torch.nn as nn
torch.manual_seed(0)


class BestNet(nn.Module):
    def __init__(self, image_size = 128):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,10,9,2,0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10,10,3,1,0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10,10,3,1,0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2,2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(10,20,7,2,0),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.Conv2d(20,20,3,1,0),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.Conv2d(20,20,3,1,0),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2,2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(20,40,3,2,0),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Conv2d(40,40,3,1,0),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Conv2d(40,40,3,1,0),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2,2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(40,80,5,2,0),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(),
            nn.Conv2d(80,80,5,1,0),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(),
            nn.Conv2d(80,80,5,1,0),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2,2)
        )
        
        '''Dummy forward pass to compute output size'''
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size, image_size)
            dummy_output = self.layer_1(dummy_input)
            dummy_output = self.layer_2(dummy_output)
            dummy_output = self.layer_3(dummy_output)
            # dummy_output = self.layer_4(dummy_output)
            flattened_size = dummy_output.numel()

        self.out = nn.Sequential(
            nn.Linear(flattened_size,1),
            nn.Dropout(p=0.2),

        )

    def forward(self,x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        # out = self.layer_4(out)
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