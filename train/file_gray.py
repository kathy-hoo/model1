import numpy as np
import cv2
import os

filenames = os.listdir('../../../dataset/CHAOS/Test_Sets/MR/11/T1/DICOM_anon/InPhase')
# filenames = os.listdir('image')

for f in filenames:
    img_path = os.path.join('../../../dataset/CHAOS/Test_Sets/MR/11/T1/DICOM_anon/InPhase', f)
    img = cv2.imread(img_path)
    # print(img_path,    np.unique(img))
    print(img.shape)

