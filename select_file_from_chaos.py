import os
import shutil

from_dir = '../../dataset/CHAOS/Train_Sets/MR'
image_dir = 'train/image'
mask_dir = 'train/mask'

# filename=os.listdir(from_dir)
# for i in filename:
#     shutil.copy(from_dir+'/'+i,image_dir+'/'+i)

# for fpath,fdir,fname in os.walk(from_dir):  #遍历所有文件与问价夹
'''
    fpath是一个string，代表目录的路径，
    fdir是一个list，包含了dirpath下所有子目录的名字，
    fname是一个list，包含了非目录文件的名字，这些名字不包含路径信息。
    如果需要得到全路径,需要使用 os.path.join(dirpath, name)
'''
folds = os.listdir(from_dir)
for fold in folds:
    IM_dir = from_dir + '/' + fold + '/T1/DICOM_anon/InPhase'
    MASK_dir = from_dir + '/' + fold + '/T1/Ground'
    for i, im in enumerate(os.listdir(IM_dir)):
        shutil.copyfile(IM_dir + '/' + im, (image_dir + '/' + '{}_{}.png').format(fold, i))
    for i, msk in enumerate(os.listdir(MASK_dir)):
        shutil.copy(MASK_dir + '/' + msk, (mask_dir + '/' + '{}_{}.png').format(fold, i))
