from effnet import efficientnet_b0
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp



class Up(nn.Module):
    def __init__(self,in_channels,out_channels,is_deconv=False):  #320,112 尺度变大一倍
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.deconv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=2,output_padding=1,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2,dilation=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.is_deconv=is_deconv

    def forward(self,x1,x2):
        if self.is_deconv:
            x1=self.deconv(x1)
        else:
            x1=self.up(x1)
            x1 = F.relu(self.bn1(self.conv1(x1))) #silu
        x=torch.cat((x1,x2),1)
        x=F.relu(self.bn2(self.conv2(x)))   #silu
        x = F.relu(self.bn3(self.conv3(x)))  #silu
        return x


class effUNet(nn.Module):
    def __init__(self,num_classes,is_deconv=False):
        super().__init__()
        base_model=efficientnet_b0(1000)
        weights_dict = torch.load('efficientnetb0.pth')
        load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if base_model.state_dict()[k].numel() == v.numel()}
        print(base_model.load_state_dict(load_weights_dict, strict=False))
        base_layers=list(base_model.children())
        base_layer=list(base_layers[0])
        self.layer1=nn.Sequential(
                    nn.Conv2d(1,3,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False),
                    base_layer[0],
                    base_layer[1]
                    )
        self.layer2=nn.Sequential(base_layer[2],base_layer[3])
        self.layer3=nn.Sequential(base_layer[4],base_layer[5])
        self.layer4=nn.Sequential(*base_layer[6:12])
        self.layer5=nn.Sequential(*base_layer[12:17])
        self.up4=Up(320,112,is_deconv)
        self.up3=Up(112,40,is_deconv)
        self.up2=Up(40,24,is_deconv)
        self.up1=Up(24,16,is_deconv)
        self.up0=nn.ConvTranspose2d(16,num_classes,kernel_size=3,stride=2,padding=1,output_padding=1)

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
        return x

# net=smp.Unet(
#         encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
#         in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#         classes=2,  # model output channels (number of classes in your dataset)
#     )
# print(net)
# net=effUNet(5,is_deconv=False)
# x=torch.rand(2,1,384,384)
# print(net(x).shape)
# summary(net,(1,512,512),device='cpu')