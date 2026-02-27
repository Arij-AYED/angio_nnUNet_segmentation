# angio_nnUNet_segmentation
I trained nnU-Net v2 (2D) to segment coronary vessels in angiography images. I built preprocessing to resize and validate masks, implemented mask-safe augmentations (nearest interpolation + binarization), trained fold 0, and evaluated on a held-out test set with Dice and qualitative overlays. Final test performance: mean Dice 0.869 (std 0.151).
