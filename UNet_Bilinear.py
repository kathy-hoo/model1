import torch
from torch import nn
from torch.nn import functional as F
#我要自己写unet,全凭自己呦


class double_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        return x


class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self,x1,x2):
        x1=self.up(x1)
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x=torch.cat((x1,x2),1)
        x=F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

# net=Up(1024,512)
# x1=torch.rand(10,1024,28,28)
# x2=torch.rand(10,512,56,56)
# print(net(x1,x2).shape)

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,feature_scale=1):
        super().__init__()
        self.feature_scale=feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.maxpool=nn.MaxPool2d(2)
        self.block1=double_conv(n_channels,filters[0])
        self.block2=double_conv(filters[0],filters[1])
        self.block3=double_conv(filters[1],filters[2])
        self.block4=double_conv(filters[2],filters[3])
        self.block5=double_conv(filters[3],filters[4])
        self.block6=Up(filters[4],filters[3])
        self.block7=Up(filters[3],filters[2])
        self.block8=Up(filters[2],filters[1])
        self.block9=Up(filters[1],filters[0])
        self.block10=nn.Conv2d(filters[0],n_classes,kernel_size=1)
    def forward(self,x):
        x1=self.block1(x)
        x2 = self.block2(self.maxpool(x1))
        x3 = self.block3(self.maxpool(x2))
        x4 = self.block4(self.maxpool(x3))
        x5 = self.block5(self.maxpool(x4))
        x=self.block6(x5,x4)
        x=self.block7(x,x3)
        x=self.block8(x,x2)
        x=self.block9(x,x1)
        x=self.block10(x)
        return x

# x=torch.rand(2,1,512,512)
# net=UNet(1,5)
# print(net(x).shape)
# summary(net,(1,512,512),device='cpu')
