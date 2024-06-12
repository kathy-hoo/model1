# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys 
sys.path.append('/home/kathy/projects/project_guo/test_missformer')

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from model.networks.swin_unet import SwinTransformerSys
import thop 
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swin_unet = SwinTransformerSys(img_size = 128,
                                patch_size=16,
                                in_chans=3,
                                num_classes=5,
                                embed_dim=96,
                                depths=[ 2, 6, 2, 2 ],
                                num_heads=[4, 4, 4, 4],
                                window_size=8,
                                mlp_ratio=2,
                                qkv_bias=True,
                                qk_scale=True,
                                drop_rate=0.1,
                                drop_path_rate=0.1,
                                ape=True,
                                use_checkpoint=False)
        # self.deconv = nn.ConvTranspose2d()
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        
        return  logits # torch.nn.functional.upsample(logits, scale_factor=4)

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
if __name__ == "__main__":
    device = torch.device("cpu")
    model = SwinUnet().to(device)
    X = torch.rand(8, 1, 128, 128).to(device)
    # deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=4, output_padding=1)
    # y  = deconv(X)
    # print(y.shape)
    oup = model(X)
    print(oup.shape)
    flops, params = thop.profile(model, inputs = (X, ))
    print(flops / (10 ** 9))
    print(params)