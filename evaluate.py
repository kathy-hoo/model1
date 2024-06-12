import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
# import surface_distance as surfdist
# from dice_score import multiclass_dice_coeff, dice_coeff

def substract(x,y):
    """
    求2个2维0-1矩阵相减，用于求FP,FN
    :return:value
    """
    z=x-y
    z=np.maximum(z,0)
    return np.sum(z)



def sen_ppv(mask_pred:torch.tensor,mask_true:torch.tensor,epsion=1e-4):
    #仅适用于二维
    mp,mt=mask_pred.to('cpu').numpy(),mask_true.to('cpu').numpy()
    mp=mp[:, 1:, ...]
    mt=mt[:, 1:, ...]
    dice_res,sen_res,ppv_res=0,0,0
    bs=mp.shape[0]
    for i in range(bs):
        mtt=np.squeeze(mt[i])
        mpp=np.squeeze(mp[i])
        TP=np.sum(mtt*mpp)
        FP=substract(mpp,mtt)
        FN=substract(mtt,mpp)
        dice_res+=2*TP/(2*TP+FP+FN+epsion)
        sen_res+=TP/(TP+FN+epsion)
        ppv_res+=TP/(TP+FP+epsion)
    return dice_res/bs,sen_res/bs,ppv_res/bs



def evaluate(net, dataloader, n_classes,device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    dice=0
    sen=0
    ppv=0

    # iterate over the validation set
    for image,mask_true in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        #image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            dice_res,sen_res,ppv_res=sen_ppv(mask_pred,mask_true)
            dice+=dice_res
            sen+=sen_res
            ppv+=ppv_res

           

    #net.train()

    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    return dice/num_val_batches,sen/num_val_batches,ppv/num_val_batches
