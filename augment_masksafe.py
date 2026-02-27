import cv2
import numpy as np
import albumentations as A

def get_augmenter():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0, cval_mask=0, p=0.8
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ])

def apply_aug(img, mask):
    mask = (mask > 0).astype(np.uint8)     
    out = get_augmenter()(image=img, mask=mask)
    mask_aug = (out["mask"] > 0).astype(np.uint8)
    return out["image"], mask_aug
