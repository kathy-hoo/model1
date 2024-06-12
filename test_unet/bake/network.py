
import torch
from torchvision import models as resnet_model
from torch import nn
from transformers import ViTFeatureExtractor, ViTForImageClassification


class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class FAT_Net(nn.Module):
    def __init__(self, in_channle=3, n_classes=1):
        super(FAT_Net, self).__init__()

        # transformer = torch.hub.load('../model', 'deit_tiny_distilled_patch16_224', pretrained=True)
        model_id = '/home/kathy/projects/project_guo/model/vit'
        label2id = {str(k) : v for (k, v) in enumerate(range(5))}
        id2label = {v : k for (k, v) in label2id.items()}

        transformer = ViTForImageClassification.from_pretrained(
            model_id,  # classification head
            num_labels=5, 
            label2id = label2id,
            id2label = id2label
        )
        
        resnet = resnet_model.resnet34(pretrained=False)
        print("model loaded !")
        self.conv_channle = nn.Sequential(
            nn.Conv2d(in_channle, 3, 3), 
            nn.BatchNorm2d(3), 
            nn.ReLU()
        )
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.upsample = nn.Upsample(scale_factor  = 22 / 14)

        self.patch_embed = transformer.vit.embeddings
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(10, 10), padding = 1), 
            
        )
        self.transformers = nn.ModuleList(
            [transformer.vit.encoder.layer[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_channle(x)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)
        feature_cnn = self.upsample(feature_cnn)
        e3 = self.upsample(e3)
        e2 = self.upsample(e2)
        e1 = self.upsample(e1)

        emb = self.patch_embed(x)
        emb = emb.reshape(emb.shape[0], -1, 768)
        
        for i in range(12):
            emb = self.transformers[i](emb)[0]
        emb = emb[:, :, :]
        
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 768, 22, 22)
        feature_tf = self.conv_seq_img(feature_tf)

        
        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out
    
if __name__ ==  "__main__":
    device = torch.device("cuda:8")
    
    model = FAT_Net(1, 5).to(device)
    
    X = torch.randn(8, 1, 226, 226).to(device)
    
    X =  model(X)
    print(X.shape)