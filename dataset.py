from torch.utils.data import Dataset
from os.path import splitext
from os import listdir
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

trfm = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    A.OneOf([A.RandomContrast(), A.RandomGamma()], p=0.5)
])


def mask_process(mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    palette = np.array([0, 63, 126, 189, 252])
    for i in range(len(palette)):
        equality = np.equal(mask, palette[i])
        # class_map = np.all(equality, axis=-1)
        mask[equality] = i
    return mask


class MRABDataset(Dataset):
    def __init__(self, image_dir, mask_dir, scale=256, augment=True):  # 256*256
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.idx = [splitext(file)[0] for file in listdir(image_dir)]
        self.gen_image_list()
        self.augment = augment

    def gen_image_list(self):
        self.image_list = []
        self.mask_list = []
        for filename in self.idx:
            file = filename + '.png'
            self.mask_list.append(os.path.join(self.mask_dir, file))
            self.image_list.append(os.path.join(self.image_dir, file))

    def __len__(self):
        return len(self.image_list)  # 部分image有多个肿瘤mask

    def __getitem__(self, index):

        # print(self.image_list[index])
        image = cv2.imread(self.image_list[index])[:, :, 0]
        mask = cv2.imread(self.mask_list[index])[:, :, 0]
        # mask=mask/255.
        if self.augment == True:
            augments = trfm(image=image, mask=mask)
            image, mask = augments['image'], augments['mask'][None]
            mask = np.squeeze(mask)
        assert image.shape == mask.shape, 'Image and mask must be equal shape'
        # assert image.shape[0]==256 and image.shape[1]==256,'Image not equal size 256x256'
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (self.scale, self.scale), interpolation=cv2.INTER_LINEAR)  # 双线性
        new_image = (image - np.mean(image)) / np.std(image)
        new_image = np.expand_dims(new_image, 0)
        mask = cv2.resize(mask, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)  # 最近邻
        new_mask = mask_process(mask)
        return new_image, new_mask

# image z-score标准化  mask 0-1二值
#
# image_dir= 'Train_Sets/1/T1/DICOM_anon/InPhase'
# mask_dir='Train_Sets/1/T1/Ground'
#
# img=cv2.imread(os.path.join(image_dir,'IMG-0004-00020.png'))[:,:,0]
# mask=cv2.imread(os.path.join(mask_dir,'IMG-0004-00044.png'))[:,:,0]
# mask=mask_process(mask)
# plt.imshow(mask,cmap='gray')
# plt.show()

# dataset=MRABDataset(image_dir,mask_dir)
# print(dataset.__len__())
# img, msk = dataset.__getitem__(20)

# plt.imshow(img.reshape(msk.shape),cmap='gray')
# plt.figure(),plt.imshow(msk,cmap='gray')
# plt.show()
# image=cv2.imread('bus/original/000008.png')
# for i in range(10):
#     image1=trfm(image=image)['image']
#     plt.figure(figsize=(10,10))
#     plt.imshow(image1,cmap='gray')
#     plt.show()
