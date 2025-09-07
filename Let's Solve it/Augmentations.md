### Must-do augmentations (AGAR)

| Augmentation | Why it helps for plates | Quick setting | Source |
|---|---|---|---|
| **Rotation + H/V flips** | Plate orientation is arbitrary; boosts invariance without altering biology. | Any angle (0–360°) + flips | [ScienceDirect §4](https://www.sciencedirect.com/science/article/pii/S277244252400042X#sec4), [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Scale jitter (zoom in/out)** | Mimics magnification/µm-px differences across setups; helps small-object recall. | Random resize ~0.8–1.2× | [ScienceDirect §4](https://www.sciencedirect.com/science/article/pii/S277244252400042X#sec4), [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Brightness/contrast/gamma jitter** | Models exposure/illumination variation common in microscopes. | ±10–20% brightness/contrast; γ 0.8–1.2 | [ScienceDirect §4](https://www.sciencedirect.com/science/article/pii/S277244252400042X#sec4), [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Mild additive noise** | Simulates sensor/shot noise; improves robustness on faint colonies. | Gaussian noise (σ small) | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |


### Augmentations to try (from the papers)

| Augmentation | Why to try | Quick setting | Source |
|---|---|---|---|
| **Translation** | Stage drift / off-center fields. | Random pixel shifts | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Random cropping** | Varies FOV; regularizes context. | Crop 80–100% area | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Blurring (defocus)** | Autofocus variability; tests sensitivity to mild blur. | Gaussian blur σ ≤ 1–1.5 px | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Elastic deformation** | Shape variability; useful if colony edges deform with agar imperfections. | Small α/σ | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **MixUp (mixing-based)** | Regularizes boundaries; can help class imbalance. | α ≈ 0.2–0.4 | [ScienceDirect §4](https://www.sciencedirect.com/science/article/pii/S277244252400042X#sec4), [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |


### Augmentations tied to microscopy quality issues

| Imaging issue                    | Augmentation                      | Why                                                   | Source                                                                                                                                                             |
| -------------------------------- | --------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Out-of-focus frames**          | Defocus/Gaussian blur             | Trains tolerance to mild focus drift.                 | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23),[Science Direct](https://www.sciencedirect.com/science/article/pii/S2001037022001192)         |
| **Sensor/low-light noise**       | Additive noise (Gaussian/Poisson) | Models acquisition noise; stabilizes faint features.  | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23), [Nature](https://www.nature.com/articles/s41587-022-01450-8?utm_source)                      |
| **Exposure/illumination shifts** | Brightness/contrast/gamma jitter  | Covers lamp intensity/white-balance variation.        | [ScienceDirect §4](https://www.sciencedirect.com/science/article/pii/S277244252400042X#sec4), [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23) |
| **Stage drift / misalignment**   | Small translation + rotation      | Matches real capture jitter without changing content. | [PMC §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/#Sec23)                                                                                               |

## Code
```python
#!/usr/bin/env python3
"""
agar_augment.py — Augmentations for AGAR-style plate images.

Presets map directly to the three tables we compiled:
- must: rotation+flips, scale jitter, brightness/contrast/gamma jitter, mild noise
- try:  add translation, random cropping, blur (defocus), elastic deformation (all mild)
- quality: focus/noise/exposure robustness → blur + noise + brightness/contrast/gamma + small translation/rotation

You can also use --preset custom and toggle each aug with flags.

Install:
    pip install albumentations==1.* opencv-python

Usage examples:
    # Must-do preset, 4 augmented samples per image
    python agar_augment.py --input_dir ./images --output_dir ./aug --n_per_image 4 --preset must

    # Try-preset with MixUp enabled (alpha=0.3)
    python agar_augment.py --input_dir ./images --output_dir ./aug_try --n_per_image 4 --preset try --enable_mixup --mixup_alpha 0.3

    # Custom: only rotation+flips and brightness/contrast
    python agar_augment.py --input_dir ./images --output_dir ./aug_custom --preset custom \
        --enable_rotation --enable_hflip --enable_vflip --enable_brightness_contrast

Notes:
- Images are read as 3-channel (BGR) and saved as .jpg by default.
- MixUp blends two *images*; if you need label mixing, handle labels in your training loop accordingly.
"""

import argparse
import os
from pathlib import Path
import random

import cv2
import numpy as np
import albumentations as A


# --------------------------
# Config & preset utilities
# --------------------------

def build_config_from_preset(preset: str) -> dict:
    """
    Returns a dict of booleans/params for the requested preset.
    You can override any key via --enable_* / --disable_* flags (custom only) or param flags.
    """
    # Base defaults (all off; conservative)
    cfg = dict(
        # geometry
        rotation=False, rot_deg=180,
        hflip=False, vflip=False,
        scale_jitter=False, scale_range=(0.8, 1.2),
        translation=False, translate_frac=0.05,

        # photometrics
        brightness_contrast=False, brightness_limit=0.2, contrast_limit=0.2,
        gamma=False, gamma_range=(0.8, 1.2),

        # quality-related
        noise=False, noise_var=(10.0, 50.0),   # variance range for GaussNoise (0..255 scale^2)
        blur=False, blur_limit=(3, 5),         # kernel size range for GaussianBlur
        elastic=False, elastic_alpha=5, elastic_sigma=50, elastic_affine=5,

        # cropping
        random_crop=False, crop_scale=(0.8, 1.0), crop_ratio=(0.9, 1.1),

        # mixup
        mixup=False, mixup_alpha=0.3,
    )

    p = preset.lower()
    if p == "must":
        cfg.update(dict(
            rotation=True, rot_deg=180,
            hflip=True, vflip=True,
            scale_jitter=True, scale_range=(0.8, 1.2),
            brightness_contrast=True, brightness_limit=0.2, contrast_limit=0.2,
            gamma=True, gamma_range=(0.8, 1.2),
            noise=True, noise_var=(10.0, 50.0),
        ))
    elif p == "try":
        cfg.update(dict(
            rotation=True, rot_deg=180,
            hflip=True, vflip=True,
            scale_jitter=True, scale_range=(0.85, 1.15),
            translation=True, translate_frac=0.05,
            brightness_contrast=True, brightness_limit=0.2, contrast_limit=0.2,
            gamma=True, gamma_range=(0.85, 1.15),
            noise=True, noise_var=(10.0, 50.0),
            blur=True, blur_limit=(3, 7),
            elastic=True, elastic_alpha=5, elastic_sigma=50, elastic_affine=5,
            random_crop=True, crop_scale=(0.8, 1.0), crop_ratio=(0.95, 1.05),
        ))
        # MixUp is opt-in via --enable_mixup
    elif p == "quality":
        cfg.update(dict(
            rotation=True, rot_deg=15,     # small orientation jitter
            hflip=True, vflip=True,
            translation=True, translate_frac=0.03,
            brightness_contrast=True, brightness_limit=0.15, contrast_limit=0.15,
            gamma=True, gamma_range=(0.9, 1.1),
            noise=True, noise_var=(10.0, 40.0),
            blur=True, blur_limit=(3, 7),
        ))
    elif p == "custom":
        pass
    else:
        raise ValueError(f"Unknown preset: {preset} (choose: must, try, quality, custom)")
    return cfg


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """For --preset custom, turn individual augs on/off from CLI flags and update params."""
    if args.preset != "custom":
        return cfg

    # toggles (enable_* / disable_*) — only a subset for brevity; extend as needed
    toggles = [
        ("rotation", args.enable_rotation, args.disable_rotation),
        ("hflip", args.enable_hflip, args.disable_hflip),
        ("vflip", args.enable_vflip, args.disable_vflip),
        ("scale_jitter", args.enable_scale_jitter, args.disable_scale_jitter),
        ("translation", args.enable_translation, args.disable_translation),
        ("brightness_contrast", args.enable_brightness_contrast, args.disable_brightness_contrast),
        ("gamma", args.enable_gamma, args.disable_gamma),
        ("noise", args.enable_noise, args.disable_noise),
        ("blur", args.enable_blur, args.disable_blur),
        ("elastic", args.enable_elastic, args.disable_elastic),
        ("random_crop", args.enable_random_crop, args.disable_random_crop),
        ("mixup", args.enable_mixup, args.disable_mixup),
    ]
    for key, on, off in toggles:
        if on:
            cfg[key] = True
        if off:
            cfg[key] = False

    # numeric params (if provided)
    if args.rot_deg is not None: cfg["rot_deg"] = args.rot_deg
    if args.scale_min is not None and args.scale_max is not None:
        cfg["scale_range"] = (args.scale_min, args.scale_max)
    if args.translate_frac is not None: cfg["translate_frac"] = args.translate_frac
    if args.brightness_limit is not None: cfg["brightness_limit"] = args.brightness_limit
    if args.contrast_limit is not None: cfg["contrast_limit"] = args.contrast_limit
    if args.gamma_min is not None and args.gamma_max is not None:
        cfg["gamma_range"] = (args.gamma_min, args.gamma_max)
    if args.noise_var_min is not None and args.noise_var_max is not None:
        cfg["noise_var"] = (args.noise_var_min, args.noise_var_max)
    if args.blur_min is not None and args.blur_max is not None:
        cfg["blur_limit"] = (int(args.blur_min), int(args.blur_max))
    if args.elastic_alpha is not None: cfg["elastic_alpha"] = args.elastic_alpha
    if args.elastic_sigma is not None: cfg["elastic_sigma"] = args.elastic_sigma
    if args.elastic_affine is not None: cfg["elastic_affine"] = args.elastic_affine
    if args.crop_scale_min is not None and args.crop_scale_max is not None:
        cfg["crop_scale"] = (args.crop_scale_min, args.crop_scale_max)
    if args.crop_ratio_min is not None and args.crop_ratio_max is not None:
        cfg["crop_ratio"] = (args.crop_ratio_min, args.crop_ratio_max)
    if args.mixup_alpha is not None: cfg["mixup_alpha"] = args.mixup_alpha

    return cfg


# --------------------------
# Builder for Albumentations
# --------------------------

def build_augmentations(cfg: dict, out_size=None) -> A.Compose:
    """
    out_size: (H, W) to enforce final size when using RandomResizedCrop; if None, we keep original size.
    """
    tfs = []

    # Geometry: combine rotation/scale/translation using ShiftScaleRotate for stability
    want_ssr = cfg["rotation"] or cfg["scale_jitter"] or cfg["translation"]
    if want_ssr:
        rotate_limit = cfg["rot_deg"] if cfg["rotation"] else 0
        scale_limits = (
            cfg["scale_range"][0] - 1.0,
            cfg["scale_range"][1] - 1.0
        ) if cfg["scale_jitter"] else (0.0, 0.0)
        shift_limit = cfg["translate_frac"] if cfg["translation"] else 0.0
        tfs.append(
            A.ShiftScaleRotate(
                shift_limit=shift_limit, scale_limit=scale_limits, rotate_limit=rotate_limit,
                border_mode=cv2.BORDER_REFLECT_101, p=0.9
            )
        )

    if cfg["hflip"]:
        tfs.append(A.HorizontalFlip(p=0.5))
    if cfg["vflip"]:
        tfs.append(A.VerticalFlip(p=0.5))

    # Random cropping (keeps final size)
    if cfg["random_crop"] and out_size is not None:
        H, W = out_size
        tfs.append(
            A.RandomResizedCrop(
                height=H, width=W,
                scale=cfg["crop_scale"], ratio=cfg["crop_ratio"],
                interpolation=cv2.INTER_AREA, p=0.6
            )
        )

    # Photometric
    if cfg["brightness_contrast"]:
        tfs.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg["brightness_limit"],
                contrast_limit=cfg["contrast_limit"],
                p=0.6
            )
        )
    if cfg["gamma"]:
        # albumentations expects gamma_limit as ints in [80, 120] → map from (0.8,1.2)
        gl = (int(cfg["gamma_range"][0] * 100), int(cfg["gamma_range"][1] * 100))
        tfs.append(A.RandomGamma(gamma_limit=gl, p=0.4))

    # Quality-related
    if cfg["noise"]:
        tfs.append(A.GaussNoise(var_limit=cfg["noise_var"], per_channel=False, p=0.4))
    if cfg["blur"]:
        # GaussianBlur uses kernel size; keep small to mimic mild defocus
        tfs.append(A.GaussianBlur(blur_limit=cfg["blur_limit"], p=0.3))
    if cfg["elastic"]:
        tfs.append(
            A.ElasticTransform(
                alpha=cfg["elastic_alpha"], sigma=cfg["elastic_sigma"], alpha_affine=cfg["elastic_affine"],
                border_mode=cv2.BORDER_REFLECT_101, p=0.2
            )
        )

    return A.Compose(tfs)


def mixup(img_a: np.ndarray, img_b: np.ndarray, alpha: float = 0.3) -> (np.ndarray, float):
    """
    Simple MixUp: returns mixed image and lambda.
    (Caller is responsible for mixing labels/targets appropriately.)
    Images must share the same HxW; we resize img_b to match img_a.
    """
    if alpha <= 0:
        lam = 0.5
    else:
        lam = np.random.beta(alpha, alpha)
    h, w = img_a.shape[:2]
    img_b_rs = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    mixed = (lam * img_a.astype(np.float32) + (1.0 - lam) * img_b_rs.astype(np.float32))
    return np.clip(mixed, 0, 255).astype(np.uint8), lam


# --------------------------
# I/O + Runner
# --------------------------

def list_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description="AGAR augmentations (must/try/quality/custom)")
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--n_per_image", type=int, default=2, help="Augmented samples to generate per input")
    ap.add_argument("--preset", choices=["must", "try", "quality", "custom"], default="must")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_height", type=int, default=None, help="Final H for RandomResizedCrop (optional)")
    ap.add_argument("--out_width", type=int, default=None, help="Final W for RandomResizedCrop (optional)")

    # Overrides for custom preset (enable/disable)
    for key in ["rotation","hflip","vflip","scale_jitter","translation",
                "brightness_contrast","gamma","noise","blur","elastic","random_crop","mixup"]:
        ap.add_argument(f"--enable_{key}", action="store_true")
        ap.add_argument(f"--disable_{key}", action="store_true")

    # Param overrides (custom)
    ap.add_argument("--rot_deg", type=float)
    ap.add_argument("--scale_min", type=float)
    ap.add_argument("--scale_max", type=float)
    ap.add_argument("--translate_frac", type=float)
    ap.add_argument("--brightness_limit", type=float)
    ap.add_argument("--contrast_limit", type=float)
    ap.add_argument("--gamma_min", type=float)
    ap.add_argument("--gamma_max", type=float)
    ap.add_argument("--noise_var_min", type=float)
    ap.add_argument("--noise_var_max", type=float)
    ap.add_argument("--blur_min", type=int)
    ap.add_argument("--blur_max", type=int)
    ap.add_argument("--elastic_alpha", type=float)
    ap.add_argument("--elastic_sigma", type=float)
    ap.add_argument("--elastic_affine", type=float)
    ap.add_argument("--crop_scale_min", type=float)
    ap.add_argument("--crop_scale_max", type=float)
    ap.add_argument("--crop_ratio_min", type=float)
    ap.add_argument("--crop_ratio_max", type=float)
    ap.add_argument("--mixup_alpha", type=float)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config_from_preset(args.preset)
    cfg = apply_overrides(cfg, args)

    # Determine final size for crop-based ops (optional)
    out_size = None
    if args.out_height and args.out_width:
        out_size = (int(args.out_height), int(args.out_width))

    aug = build_augmentations(cfg, out_size=out_size)

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    # For MixUp partner sampling
    def sample_partner(exclude_idx):
        idx = exclude_idx
        while idx == exclude_idx:
            idx = random.randrange(len(images))
        return images[idx]

    for idx, ipath in enumerate(images):
        img = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read: {ipath}")
            continue

        stem = ipath.stem
        for k in range(args.n_per_image):
            augmented = aug(image=img)["image"]

            if cfg["mixup"]:
                # pick a partner image
                jpath = sample_partner(idx)
                img_b = cv2.imread(str(jpath), cv2.IMREAD_COLOR)
                if img_b is None:
                    continue
                augmented, lam = mixup(augmented, img_b, alpha=cfg["mixup_alpha"])
                out_name = f"{stem}_aug{k:02d}_mix_{Path(jpath).stem}_lam{lam:.2f}.jpg"
            else:
                out_name = f"{stem}_aug{k:02d}.jpg"

            out_path = args.output_dir / out_name
            cv2.imwrite(str(out_path), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Done. Wrote augmented images to: {args.output_dir}")


if __name__ == "__main__":
    main()

```

## Message
Based on https://www.sciencedirect.com/science/article/pii/S277244252400042X and https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/, here are different image augmentations we could try:
1. must do: rotation/flips/shearing (effectiveness backed by research)
2. to try: translation (not as reliable but we can try) + zoom + random cropping + color shifting (based on research only work on specific dataset) + brightness/contrast (not sure either)
3. Microscope based augmentations: out of focus frames (blur) + noise (gaussian + poisson)
4. If we need more data, we can try GAN methods. but based on studies they are quite unreliable due to vanishing gradient and mode collapsing problems