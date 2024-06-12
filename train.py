from dataset import MRABDataset
from torch.utils.data import DataLoader, random_split
import torch
from torch import optim
import torch.nn as nn
from effUNetb4 import effUNetb4
from tqdm import tqdm
from pathlib import Path
import os
from evaluate import evaluate

val_percent: float = 0.1
batch_size: int = 6
epochs = 60

result_path = 'result.txt'
image_dir = 'train/image'
mask_dir = 'train/mask'
checkpoint_dir = 'checkpoints/'
dataset = MRABDataset(image_dir, mask_dir)

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)  # 内存充足的时候，可以设置pin_memory=True
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = effUNetb4(5)
# net=UNet(n_channels=1,n_classes=5)
net.to(device=device)

optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)
criterion = nn.CrossEntropyLoss()
best_score = 0.7  # 保存大于阈值的模型
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

for epoch in range(epochs):
    print("The {}th epoch".format(epoch + 1))
    net.train()
    epoch_loss = 0
    for images, masks in tqdm(train_loader):
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)
        masks_pred = net(images)
        loss = criterion(masks_pred, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print('lr=', optimizer.param_groups[0]['lr'])
    scheduler.step()

    print('epoch_loss={:.4f}'.format(epoch_loss))
    dice, sen, ppv = evaluate(net, val_loader, n_classes=5, device=device)
    print('dice score={:.4f},sen score={:.4f},ppv score={:.4f}'.format(dice, sen, ppv))
    # with open(result_path, 'a') as f:
    #     f.write('\n epoch:{}, train_loss:{:.4f}, val_dice_score=:{:.4f}'.format(epoch + 1, epoch_loss,val_score))

    if dice > best_score:
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'unet_b4_epoch{}.pth'.format(epochs)))
        best_score = dice
        print(f'Checkpoint saved!')

print('Training complete!')
