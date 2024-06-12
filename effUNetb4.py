from effnet import efficientnet_b4
import torch
from torch import nn
from torch.nn import functional as F

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):  #320,112 尺度变大一倍
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        

    def forward(self,x1,x2):
        x1=self.up(x1)
        x=torch.cat((x1,x2),1)
        x=F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  #+x2 +x_res为残差块
        return x

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):  # 320,112 尺度变大一倍
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = self.up(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))  # +x2 +x_res为残差块
#         return x


class effUNetb4(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        base_model=efficientnet_b4(1000)
        weights_dict = torch.load('efficientnetb4.pth')
        load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if base_model.state_dict()[k].numel() == v.numel()}
        print(base_model.load_state_dict(load_weights_dict, strict=False))
        base_layers=list(base_model.children())
        base_layer=list(base_layers[0])
        self.layer1=nn.Sequential(
                    nn.Conv2d(1,3,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
                    base_layer[0],base_layer[1],base_layer[2]
                    )  #size=192
        self.layer2=nn.Sequential(*base_layer[3:7]) #size=96
        self.layer3=nn.Sequential(*base_layer[7:11]) #48
        self.layer4=nn.Sequential(*base_layer[11:23]) #24
        self.layer5=nn.Sequential(*base_layer[23:33]) #12
        self.up4=Up(448,160)
        self.up3=Up(160,56)
        self.up2=Up(56,32)
        self.up1=Up(32,24)
        self.up0=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                               nn.Conv2d(24,num_classes,1,bias=True))
        # self.classify=nn.Conv2d(16,num_classes,1)
        # self.up0=nn.ConvTranspose2d(24,num_classes,kernel_size=3,stride=2,padding=1,output_padding=1)


    def forward(self,x):
        x1=self.layer1(x)
        x2=self.layer2(x1)
        x3=self.layer3(x2)
        x4=self.layer4(x3)
        x5=self.layer5(x4)
        x=self.up4(x5,x4)
        x=self.up3(x,x3)
        x=self.up2(x,x2)
        x=self.up1(x,x1)
        x=self.up0(x)
        # x=self.classify(x)
        return x

# net=effUNetb4(2)
# x=torch.rand(2,1,384,384)
# print(net(x).shape)
# summary(net,(1,512,512),device='cpu')
# base_model=efficientnet_b4(1000)
# base_layers=list(base_model.children())
# base_layer=list(base_layers[0])
# for i in range(len(base_layer)):
#     x=base_layer[i](x)
#     print(x.shape)
