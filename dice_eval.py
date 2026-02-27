import cv2
import numpy as np
from pathlib import Path

def dice(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    return (2 * inter) / (a.sum() + b.sum() + 1e-8)

GT_DIR = Path("labelsTs")
PR_DIR = Path("predictions")

scores = []
for gt_path in GT_DIR.glob("*.png"):
    pr_path = PR_DIR / gt_path.name
    if not pr_path.exists():
        continue
    gt = cv2.imread(str(gt_path), 0)
    pr = cv2.imread(str(pr_path), 0)
    scores.append(dice(gt, pr))

print("Mean Dice:", float(np.mean(scores)))
print("Std Dice:", float(np.std(scores)))
