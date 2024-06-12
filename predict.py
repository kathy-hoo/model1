import numpy as np
import torch
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
from UNet_Bilinear import UNet
import pandas as pd
from os.path import splitext
from tqdm import tqdm
from effUNetb4 import effUNetb4

# 星号部分代码在换模型是需要改
######################################################
net = effUNetb4(5)
######################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

net.to(device=device)
######################################################
net.load_state_dict(torch.load('checkpoints/unet_b4_epoch60.pth', map_location=device))
#######################################################

print('Model loaded!')
test_dir = '../../dataset/CHAOS/Test_Sets/MR/11/T1DUAL/DICOM_anon/InPhase'




def IoU(mask_true, mask_pred):
    assert mask_true.ndim == 2 and mask_pred.ndim == 2
    assert mask_true.shape[0] == mask_pred.shape[0] and mask_true.shape[1] == mask_pred.shape[1]
    mask_true[mask_true != 0] = 1
    mask_pred[mask_pred != 0] = 1
    Full = mask_true + mask_pred
    x1 = np.sum(Full == 1)
    x2 = np.sum(Full == 2)
    return x2 / (x1 + x2), 2 * x2 / (mask_true.sum() + mask_pred.sum())


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    # img= full_img.transpose([2,0,1])
    img = torch.tensor(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)[0]
        probs = probs.argmax(dim=0)
        pp = probs.to('cpu')
        p = pp.numpy()
        return p


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return (mask * 50).astype(np.uint8)
    elif mask.ndim == 3:
        return (np.argmax(mask, axis=0) * 255).astype(np.uint8)


#############################################################
Path('seg_result/11/').mkdir(parents=True, exist_ok=True)
#############################################################
for i, filename in enumerate(os.listdir(test_dir)):
    print(f'\nPredicting image {filename} ...')
    image = cv2.imread(os.path.join(test_dir, filename))[:, :, 0]
    assert image.shape[0] == 256 and image.shape[1] == 256
    image_up = np.zeros((0, 256))
    image_down = np.zeros((0, 256))
    image = np.concatenate((image_up, image), axis=0)
    image = np.concatenate((image, image_down), axis=0)
    if image.shape[0] != 256 or image.shape[1] != 256:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # 双线性
    new_image = (image - np.mean(image)) / np.std(image)
    new_image = np.expand_dims(new_image, 0)
    mask = predict_img(net=net,
                       full_img=new_image,
                       out_threshold=0.5,
                       device=device)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    result = mask_to_image(mask)
    result = result[0:256, :]
    cv2.imwrite('seg_result/11/' + filename, result)
#######################################################

# for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
#     image = cv2.imread(name)
#     mask = predict_img(net=net,
#                         full_img=image,
#                         out_threshold=0.5,
#                         device=device)
