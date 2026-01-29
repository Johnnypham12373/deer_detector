import cv2
import numpy as np
from PIL import Image

# Load as grayscale
img = np.array(Image.open("thumb2.jpg").convert("L"))

print(img.max(), img.min())
# 1) Normalize contrast (important)
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 2) Denoise (choose ONE)
img_dn = cv2.medianBlur(img_norm, 3)          # good for thermal speckle
# img_dn = cv2.GaussianBlur(img_norm, (3,3), 0)

# 3) CLAHE (local contrast)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
img_clahe = clahe.apply(img_dn)

# 4) Optional: mild unsharp mask (edge crispness)
blur = cv2.GaussianBlur(img_clahe, (0,0), 1.0)
img_sharp = cv2.addWeighted(img_clahe, 1.2, blur, -0.2, 0)

print(img_sharp.max(), img_sharp.min())
Image.fromarray(img_sharp).save("enhanced.png")
