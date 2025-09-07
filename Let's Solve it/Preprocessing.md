
# Preprocessing

| Step | Why it’s non-negotiable for **AGAR** | How (one-liner) | Key source (links) |
|---|---|---|---|
| **Petri-dish masking (rim/glare removal)** | AGAR plate rims, labels, and edge glare confuse detectors; standard ROI is the dish interior. | Detect plate with Hough circle → keep interior, drop rim. | https://thesai.org/Downloads/Volume13No2/Paper_74-Image_based_Automatic_Counting_of_Bacillus_cereus_Colonies.pdf |
| **Illumination / shading correction** | Cross-device images with uneven lighting hurt small-object recall; flat-field correction stabilizes contrast. | Run BaSiC/CIDRE (retrospective flat-field) before any scaling. | https://www.nature.com/articles/ncomms14836 , https://pmc.ncbi.nlm.nih.gov/articles/PMC4359755/ |
| **Contrast normalization (CLAHE)** | Local contrast boosts colony/background separation without over-amplifying noise like global HE can. | Apply CLAHE after shading correction. | https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html |
| **Size / resolution standardization** | AGAR was captured with **two cameras** under **diverse lighting**; normalizing size/scale reduces domain shift. | Resize to fixed long side (e.g., 1280–1536 px) with aspect preserved. | https://agar.neurosys.com/ , https://neurosys.com/blog/annotated-germs-automated-recognition-agar |

```python
# preprocess.py

# Pipeline:
# 1) Illumination correction (flat-field)
# 2) (Optional) Physical normalization to target µm/px
# 3) Dish masking
# 4) NORMALIZATION (choose one via --norm): none | percentile | autocontrast | hist_eq | clahe
# 5) Resize to standard long side
#
# Suggested for AGAR: --norm percentile --p_low 1 --p_high 99

import argparse
from pathlib import Path
import cv2
import numpy as np

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def illumination_correct(img_bgr, ksize=201):
    """Approximate flat-field via large-kernel median blur per channel."""
    img = img_bgr.astype(np.float32)
    bg = cv2.medianBlur(img, ksize)
    bg[bg < 1.0] = 1.0
    corrected = img / bg
    corrected = corrected / corrected.max(axis=(0,1), keepdims=True)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return corrected

def resample_to_um_per_px(img, current_um_per_px=None, target_um_per_px=None):
    """Optional physical normalization: resample so that physical pixel size == target_um_per_px."""
    if current_um_per_px is None or target_um_per_px is None:
        return img
    scale = float(current_um_per_px) / float(target_um_per_px)  # >1 => upsample, <1 => downsample
    if abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def find_dish_mask(gray, min_radius_frac=0.35, max_radius_frac=0.49):
    """Detect the Petri dish with HoughCircles; fallback to centered circle; erode to drop rim."""
    h, w = gray.shape[:2]
    minR = int(min(h, w) * min_radius_frac)
    maxR = int(min(h, w) * max_radius_frac)
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h,w)//2,
        param1=100, param2=30, minRadius=minR, maxRadius=maxR
    )
    mask = np.zeros_like(gray, dtype=np.uint8)
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        cx, cy, r = max(circles[0, :], key=lambda x: x[2])
    else:
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * 0.45)
    r = int(r * 0.96)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.erode(mask, erode_k, iterations=1)
    return mask

# --- Normalization options (operate on GRAYSCALE after masking) ---

def normalize_hist_eq(gray):
    return cv2.equalizeHist(gray)

def normalize_clahe(gray, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)

def normalize_percentile(gray, low=1.0, high=99.0):
    vals = gray[gray > 0]
    if vals.size == 0:  # nothing but background
        return gray
    p_low = np.percentile(vals, low)
    p_high = np.percentile(vals, high)
    if p_high <= p_low:
        return gray
    g = gray.astype(np.float32)
    g = (g - p_low) / (p_high - p_low)
    g = np.clip(g, 0.0, 1.0) * 255.0
    return g.astype(np.uint8)

def normalize_autocontrast(gray, cutoff=0.5):
    """
    PIL-like autocontrast: drop `cutoff`% pixels on each tail automatically, then rescale.
    E.g., cutoff=0.5 means 0.5% low and 0.5% high are saturated.
    """
    return normalize_percentile(gray, low=cutoff, high=100.0 - cutoff)

def resize_long_side(img, long_side=1536):
    h, w = img.shape[:2]
    if max(h, w) == long_side:
        return img
    scale = long_side / float(max(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def preprocess(
    img_bgr,
    long_side=1536,
    current_um_per_px=None,
    target_um_per_px=None,
    norm_mode="percentile",
    p_low=1.0,
    p_high=99.0,
    autocontrast_cutoff=0.5,
    clahe_clip=2.0,
    clahe_tile=8,
    save_intermediates=False,
    outdir=None,
    stem="image",
):
    # 1) Illumination correction
    img_corr = illumination_correct(img_bgr)

    # 2) (Optional) Physical normalization to target µm/px
    img_phys = resample_to_um_per_px(img_corr, current_um_per_px, target_um_per_px)

    # 3) Dish masking
    gray_corr = cv2.cvtColor(img_phys, cv2.COLOR_BGR2GRAY)
    mask = find_dish_mask(gray_corr)
    img_masked = cv2.bitwise_and(img_phys, img_phys, mask=mask)
    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # 4) NORMALIZATION (choose one)
    if norm_mode == "none":
        gray_norm = gray
    elif norm_mode == "percentile":
        gray_norm = normalize_percentile(gray, low=p_low, high=p_high)
    elif norm_mode == "autocontrast":
        gray_norm = normalize_autocontrast(gray, cutoff=autocontrast_cutoff)
    elif norm_mode == "hist_eq":
        gray_norm = normalize_hist_eq(gray)
    elif norm_mode == "clahe":
        gray_norm = normalize_clahe(gray, clip=clahe_clip, tile=clahe_tile)
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    # 5) Resize / standardize resolution
    out_resized = resize_long_side(gray_norm, long_side=long_side)

    if save_intermediates and outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(outdir) / f"{stem}_00_raw.jpg"), img_bgr)
        cv2.imwrite(str(Path(outdir) / f"{stem}_10_corr.jpg"), img_corr)
        cv2.imwrite(str(Path(outdir) / f"{stem}_15_phys.jpg"), img_phys)
        cv2.imwrite(str(Path(outdir) / f"{stem}_20_mask.png"), mask)
        cv2.imwrite(str(Path(outdir) / f"{stem}_30_gray.jpg"), gray)
        cv2.imwrite(str(Path(outdir) / f"{stem}_40_norm_{norm_mode}.jpg"), gray_norm)
        cv2.imwrite(str(Path(outdir) / f"{stem}_50_final.jpg"), out_resized)

    return out_resized

def main():
    ap = argparse.ArgumentParser(description="AGAR preprocessing with selectable normalization")
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="preprocessed_out", help="Directory to save outputs")
    ap.add_argument("--long_side", type=int, default=1536, help="Target long side in pixels")
    ap.add_argument("--current_um_per_px", type=float, default=None, help="Current physical pixel size (µm/px)")
    ap.add_argument("--target_um_per_px", type=float, default=None, help="Desired physical pixel size (µm/px)")
    ap.add_argument("--norm", default="percentile",
                    choices=["none","percentile","autocontrast","hist_eq","clahe"],
                    help="Normalization mode")
    ap.add_argument("--p_low", type=float, default=1.0, help="Lower percentile (percentile mode)")
    ap.add_argument("--p_high", type=float, default=99.0, help="Upper percentile (percentile mode)")
    ap.add_argument("--autocontrast_cutoff", type=float, default=0.5, help="Tail cutoff %% per side (autocontrast)")
    ap.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit (clahe mode)")
    ap.add_argument("--clahe_tile", type=int, default=8, help="CLAHE tile grid size (clahe mode)")
    ap.add_argument("--save_intermediates", action="store_true", help="Save intermediate steps")
    args = ap.parse_args()

    img = read_image(args.input)
    stem = Path(args.input).stem
    final_img = preprocess(
        img,
        long_side=args.long_side,
        current_um_per_px=args.current_um_per_px,
        target_um_per_px=args.target_um_per_px,
        norm_mode=args.norm,
        p_low=args.p_low,
        p_high=args.p_high,
        autocontrast_cutoff=args.autocontrast_cutoff,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        save_intermediates=True,
        outdir=args.outdir,
        stem=stem,
    )

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    final_path = Path(args.outdir) / f"{stem}_final.jpg"
    cv2.imwrite(str(final_path), final_img)
    print(f"Saved: {final_path}")

if __name__ == "__main__":
    main()

```

## Pipeline Suggestion
These are preprocessing pipeline mainly for the AGRO dataset we are using. (tgt with sources when applicable)
- Petri-dish masking (using Hough circle): https://arxiv.org/abs/2505.20365
- Illumination correction: (CIDRE) https://pubmed.ncbi.nlm.nih.gov/25775044/ + (BASIC) https://www.nature.com/articles/ncomms14836
- Contrast Normalization: (CLAHE) https://www.nature.com/articles/s41598-025-88451-0
- Resolution Standardization (resizing long side to sth like 1536px): https://arxiv.org/html/2502.03674v1 (this is for satellite images, but detecting small objects is still similar)
- µm/px (physical) normalization: https://qupath.readthedocs.io/en/stable/docs/deep/stardist.html (they need pixel in their model)
- percentile-based min-max normalization (alternative to CLAHE). we can see if it helps.