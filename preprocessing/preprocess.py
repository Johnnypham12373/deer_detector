import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

root = r"mmdetection\data"
data = ["test", "train", "val"]
# Load as grayscale
img = np.array(Image.open(r"C:\Users\Johnny\Desktop\deer_detector\mmdetection\data\train\bw_19691231_190719_frame_33_jpg.rf.d9aaf31d983b2e995dd17a2f5d2ddad5.jpg").convert("L"))

img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img_dn = cv2.medianBlur(img_norm, 3) 

col_mean = np.mean(img_norm, axis=0)
img_destriped = img_norm - col_mean + np.mean(col_mean)
img_destriped = np.clip(img_destriped, 0, 255).astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
img_clahe = clahe.apply(img_destriped)

img_sharp = cv2.addWeighted(img_clahe, 1.2, img_clahe, -0.2, 0)

# Image.fromarray(img_sharp).save("enhanced.png")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

images = [
    (img, "Original (Grayscale)"),
    (img_norm, "Normalized"),
    (img_dn, "Median Blur"),
    (img_clahe, "CLAHE"),
    (img_destriped, "Destriped"),
    (img_sharp, "Sharpened")
]

for ax, (im, title) in zip(axes, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()