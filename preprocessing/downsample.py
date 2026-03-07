from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def estimate_stripe_sensor_median(img_sensor_u8: np.ndarray,
                                  smooth_cols: int = 41) -> np.ndarray:
    """
    Estimate stripe as a column-bias at SENSOR resolution using a robust statistic (median).
    img_sensor_u8: (H, W) = (120, 160) uint8
    Returns stripe_sensor_u8: same shape, uint8-like (actually float32 returned)
    """
    I = img_sensor_u8.astype(np.float32)

    # One bias value per column (robust to deer/tree content)
    col_bias = np.median(I, axis=0)              # (W,)
    col_bias -= np.mean(col_bias)                # preserve global brightness

    # Smooth across columns so we only model low-frequency stripe drift
    if smooth_cols is not None and smooth_cols > 1:
        if smooth_cols % 2 == 0:
            smooth_cols += 1
        col_bias = cv2.GaussianBlur(col_bias.reshape(1, -1),
                                    (smooth_cols, 1), 0).ravel()

    stripe = np.tile(col_bias[None, :], (I.shape[0], 1))  # (H, W)
    return stripe.astype(np.float32)


def destripe_match_sensor(img_train_u8: np.ndarray,
                          sensor_size=(160, 120),
                          smooth_cols: int = 41,
                          strength: float = 1.0,
                          up_interp=cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Destripe a training image (256x192) by estimating stripe at native sensor resolution (160x120).

    img_train_u8: uint8, shape (192, 256) or (H, W)
    sensor_size: (W, H) = (160, 120)
    smooth_cols: smoothing for column bias at sensor grid
    strength: 0..1.2 typically. <1 is safer. Try 0.8 if over-subtraction.
    up_interp: interpolation for upsampling stripe map back (LINEAR is usually best)

    Returns:
      clean_u8: destriped at train resolution
      stripe_train_f32: stripe map (train resolution, float32)
      img_sensor_u8: downsampled sensor-grid image used for estimation
    """
    Ht, Wt = img_train_u8.shape[:2]
    Ws, Hs = sensor_size

    # 1) Downsample to sensor grid (AREA is best for downsampling)
    img_sensor = cv2.resize(img_train_u8, (Ws, Hs), interpolation=cv2.INTER_AREA)

    stripe_sensor = estimate_stripe_sensor_median(img_sensor, smooth_cols=smooth_cols) 

    img_norm = cv2.normalize(
    stripe_sensor.astype(np.float32),
    None,
    0.0, 1.0,
    cv2.NORM_MINMAX
    )

    # img_dn = cv2.medianBlur(img_norm, 3) 

    # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    # img_clahe = clahe.apply(img_dn)

    # # 3) Upsample stripe map to train resolution
    stripe_train = cv2.resize(img_norm, (Wt, Ht), interpolation=up_interp).astype(np.float32)

    # # # 4) Subtract stripe map at train resolution
    # clean = img_train_u8.astype(np.float32) - strength * stripe_train
    # clean = np.clip(clean, 0, 255).astype(np.uint8)

    return stripe_train

def preprocess_image(img_bgr_or_gray: np.ndarray,
                     sensor_size=(160, 120),
                     smooth_cols: int = 41,
                     strength: float = 0.9,
                     add_light_blur: bool = True) -> np.ndarray:
    """
    Applies destriping on a grayscale image.
    If input is BGR, converts to grayscale first (keeps pipeline consistent for thermal).
    """
    if img_bgr_or_gray is None:
        raise ValueError("Failed to read image (None returned).")

    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray

    clean = destripe_match_sensor(
        gray,
        sensor_size=sensor_size,
        smooth_cols=smooth_cols,
        strength=strength
    )

    if add_light_blur:
        clean = cv2.GaussianBlur(clean, (0, 0), 0.4)

    return clean

def preprocess_for_yolo(img_train_u8: np.ndarray,
                        mix_prob: float = 0.3,
                        destripe_strength: float = 0.9,
                        smooth_cols: int = 41,
                        add_light_blur: bool = True) -> np.ndarray:
    """
    Recommended training preprocessing:
      - Mostly destriped (estimated at sensor grid)
      - Sometimes raw (mix_prob) to prevent dependency on preprocessing
      - Optional very light blur (helps suppress residual banding without harming deer)

    img_train_u8: uint8 grayscale (H, W) = (192, 256)
    mix_prob: probability to keep raw image instead of destriped
    """
    # Decide whether to keep raw (robustness)
    if np.random.rand() < mix_prob:
        out = img_train_u8.copy()
    else:
        out, _, _ = destripe_match_sensor(
            img_train_u8,
            sensor_size=(160, 120),
            smooth_cols=smooth_cols,
            strength=destripe_strength
        )

    if add_light_blur:
        out = cv2.GaussianBlur(out, (0, 0), 0.4)

    return out


# -------------------------
# Quick visualization helper
# -------------------------
def show_destripe_debug(img_train_u8: np.ndarray,
                        destripe_strength: float = 0.9,
                        smooth_cols: int = 41):
    import matplotlib.pyplot as plt

    clean, stripe_train, img_sensor = destripe_match_sensor(
        img_train_u8,
        sensor_size=(160, 120),
        smooth_cols=smooth_cols,
        strength=destripe_strength
    )

    diff = cv2.absdiff(img_train_u8, clean)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(img_train_u8, cmap="gray"); ax[0].set_title("Input (256×192)"); ax[0].axis("off")
    ax[1].imshow(img_sensor, cmap="gray");   ax[1].set_title("Downsample to sensor (160×120)"); ax[1].axis("off")

    # visualize stripe map (centered)
    stripe_disp = np.clip(stripe_train + 128, 0, 255).astype(np.uint8)
    ax[2].imshow(stripe_disp, cmap="gray");  ax[2].set_title("Stripe map (train res, shifted)"); ax[2].axis("off")

    ax[3].imshow(clean, cmap="gray");        ax[3].set_title("Clean (Input − Stripe)"); ax[3].axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.imshow(diff, cmap="gray")
    plt.title("|Input − Clean| (should be mostly vertical noise)")
    plt.axis("off")
    plt.show()

    return clean

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def preprocess_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    sensor_size=(160, 120),
    smooth_cols: int = 41,
    strength: float = 0.9,
    add_light_blur: bool = True
) -> None:
    """
    Recursively processes all files under src_split_dir.
    - Images -> preprocessed and saved
    - Non-images -> copied as-is
    """
    all_files = [p for p in src_split_dir.rglob("*") if p.is_file()]

    for src_path in tqdm(all_files, desc=f"Processing {src_split_dir.name}", unit="file"):
        rel = src_path.relative_to(src_split_dir)
        dst_path = dst_split_dir / rel

        ensure_parent_dir(dst_path)

        if is_image_file(src_path):
            # read image (keep as loaded; then preprocess converts to grayscale if needed)
            img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)

            try:
                out = preprocess_image(
                    img,
                    sensor_size=sensor_size,
                    smooth_cols=smooth_cols,
                    strength=strength,
                    add_light_blur=add_light_blur
                )
            except Exception as e:
                print(f"[WARN] Failed preprocessing {src_path}: {e}. Copying original.")
                shutil.copy2(src_path, dst_path)
                continue

            # Save output as same extension. For JPEG, set quality.
            ext = src_path.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif ext == ".png":
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                cv2.imwrite(str(dst_path), out)
        else:
            # Copy labels/annotations/anything else unchanged
            shutil.copy2(src_path, dst_path)

def main():
    base = Path(r"C:\Users\Johnny\Desktop\deer_detector\mmdetection\data")
    out_base = base.parent / "data_1_1"  # sibling of `data`

    splits = ["train", "val", "test"]

    # Params (tweak if needed)
    sensor_size = (160, 120)   # (W, H)
    smooth_cols = 41
    strength = 0.9
    add_light_blur = True

    print(f"Input:  {base}")
    print(f"Output: {out_base}")
    out_base.mkdir(parents=True, exist_ok=True)

    for s in splits:
        src = base / s
        dst = out_base / s

        if not src.exists():
            print(f"[SKIP] Split not found: {src}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        preprocess_split(
            src, dst,
            sensor_size=sensor_size,
            smooth_cols=smooth_cols,
            strength=strength,
            add_light_blur=add_light_blur
        )

    print("\nDone. Preprocessed dataset written to:")
    print(out_base)


if __name__ == "__main__":
    main()
# img = np.array(Image.open(r"C:\Users\Johnny\Desktop\deer_detector\image.png").convert("L"))
# # img: uint8 grayscale or color
# h, w = img.shape[:2]

# # img is uint8 grayscale (192,256)
# img_stripe = preprocess_for_yolo(
#     img,
#     mix_prob=0.7,             # 30% raw, 70% destriped
#     destripe_strength=0.9,    # safe default
#     smooth_cols=41,
#     add_light_blur=True
# )
# img = cv2.imread(r"C:\Users\Johnny\Desktop\deer_detector\mmdetection\data\train\bw_20250601_213827_frame_14963_jpg.rf.364d9225c1ce593b49f5879b74b75154.jpg", cv2.IMREAD_GRAYSCALE)
# # img_stripe = show_destripe_debug(img, destripe_strength=0.9, smooth_cols=41)
# img_norm = cv2.normalize(img_stripe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# img_dn = cv2.medianBlur(img_norm, 3) 

# clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
# img_clahe = clahe.apply(img_dn)

# img_sharp = cv2.addWeighted(img_clahe, 1.2, img_clahe, -0.2, 0)

# # Image.fromarray(img_sharp).save("enhanced.png")
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.ravel()

# images = [
#     (img, "Original (Grayscale)"),
#     (img_stripe, "Destriped"),
#     (img_norm, "Normalized"),
#     (img_dn, "Median Blur"),
#     (img_clahe, "CLAHE"),
#     (img_sharp, "Sharpened")
# ]

# for ax, (im, title) in zip(axes, images):
#     ax.imshow(im, cmap="gray")
#     ax.set_title(title)
#     ax.axis("off")

# plt.tight_layout()
# plt.show()
