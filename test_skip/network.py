import torch 
import torch.nn as nn 
from torch.nn import functional as  F 
from left import LeftModel
from middle import MiddleModule
# from torch.utils
import thop 

class Model(nn.Module):
    def __init__(self, proj_dim = 64, out_dim = 5, resolution = 128):
        '''
        proj_dim : proj dim in middle model
        out_dim : classes num 
        resolution : resolution of origin image
        '''
        super().__init__()
        self.left_model = LeftModel()
        self.middle_model = MiddleModule()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
      
        
        self.cls = nn.Conv2d(64, out_dim, kernel_size=3, padding = 1)
        self.resolution = resolution
    def forward(self, X):
        tf_output, cnn_output = self.left_model(X) # bs 64 w h, bs 64 wh 
        left_output = tf_output
        # [b 64, 32, 32], [b, 128, 16, 16], [b, 256, 8, 8], [b, 512, 4, 4]
        l4, l3, l2, l1 = left_output
        
        x2 = F.upsample(l1, size=l2.size()[-2:], mode="bilinear") # [b, 512, 8, 8]
        x12 = self.conv1(torch.cat((self.deconv1(x2), l2), dim = 1))
        
        x3 = F.upsample(x12, size=l3.size()[-2:], mode="bilinear")
        x23 = self.conv2(torch.cat((self.deconv2(x3), l3), dim = 1))
        
        x4 = F.upsample(x23, size=l4.size()[-2:], mode="bilinear")
        x34 = self.conv3(torch.cat((self.deconv3(x4) ,l4), dim = 1))
        # x12 = self.conv1(x2 + )
        
        return F.upsample(self.cls(x34), scale_factor=4, mode = "bilinear")
        
        # l3, l2, l1 = self.middle_model(left_output)
        # l1 = F.upsample(l1, size=l2.size()[-2:], mode="bilinear")
        # l2_l1 = self.conv1(l1 + l2)
        # l2_l1 = F.upsample(l2_l1, size=l3.size()[-2:],  mode="bilinear")
        # l3_l2_l1 = self.conv2(l2_l1 + l3)
        # l3_l2_l1 = F.upsample(l3_l2_l1, size=(self.resolution, self.resolution),  mode="bilinear")
        # l3_l2_l1 = self.conv3(l3_l2_l1)
        # return l3_l2_l1

if __name__ == "__main__":
    device = torch.device("cpu")
    model = Model(64, 5, 128).to(device)
    print(model)
    X = torch.rand(8, 1, 128, 128).to(device)
    flops, params = thop.profile(model, inputs = (X, ))
    print(flops / (10 ** 9))
    print(params)
    # up = nn.Upsample(scale_factor=2)
    # x1 = up(X)
    # x2 = torch.rand(8, 1, 256, 256)
    # diffY = x2.size()[2] - x1.size()[2]
    # diffX = x2.size()[3] - x1.size()[3]

    # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                 diffY // 2, diffY - diffY // 2])
    
    # print(x1.shape)