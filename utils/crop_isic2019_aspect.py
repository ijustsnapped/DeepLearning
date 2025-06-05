#!/usr/bin/env python3
"""
crop_isic2019_rectangular.py

“Paper‐style” rectangular crop for ISIC2019 uncropped dermoscopy images, with a progress bar.
1) Convert to grayscale.
2) Binarize with a LOW threshold so that everything inside the circular FOV → 1.
3) Compute image moments; from them, compute an ellipse that has the same second central moments.
4) Build a bounding rectangle around that ellipse.
5) Crop if the ellipse‐area / image‐area ∈ [min_ratio, max_ratio].
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def crop_paper_style(src_path, dst_path, thresh_val, min_ratio, max_ratio):
    # 1) Read
    img = cv2.imread(src_path)
    if img is None:
        print(f"[WARN] Could not load {src_path}, skipping.")
        return

    h, w = img.shape[:2]
    image_area = float(w * h)

    # 2) Grayscale → threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use a low fixed threshold so that the entire FOV becomes “foreground.”
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 3) Compute moments on the binary mask
    M = cv2.moments(binary)
    if abs(M["m00"]) < 1e-8:
        # no white pixels at all, just save original
        cv2.imwrite(dst_path, img)
        return

    # 4) From moments, extract center (cx, cy) and ellipse axes/angle.
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Calculate second‐order central moments:
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]

    # Construct covariance matrix for the ellipse fitting
    cov = np.array([[mu20, mu11],
                    [mu11, mu02]], dtype=np.float32)

    # Eigen‐decompose to get major/minor axes
    evals, evecs = np.linalg.eigh(cov)
    # sort eigenvalues so that λ1 ≥ λ2
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # The lengths of the ellipse axes (up to a scale factor) are proportional to sqrt(eigenvalues)
    # We want to draw an ellipse that covers “approximately” the same region of white pixels.
    # A common choice is: 2 * sqrt(2 * λ_i), which is the width/height of the 1‐std ellipse in Gaussian analogy.
    # You can tweak the factor; here we use 2*sqrt(eigenvalue) to cover ~95% of data if it were Gaussian-ish.
    major_axis_length = 2.0 * np.sqrt(evals[0])
    minor_axis_length = 2.0 * np.sqrt(evals[1])

    # But if the mask is nearly circular, evals will be ~equal → easy.
    # If one is zero or negative due to quantization, clamp to a small positive
    if major_axis_length <= 0:
        major_axis_length = 1.0
    if minor_axis_length <= 0:
        minor_axis_length = 1.0

    # Convert to integer px‐lengths
    a = int(np.ceil(major_axis_length))
    b = int(np.ceil(minor_axis_length))

    # 5) Use those as half‐width/half‐height to build a bounding rect around center (cx, cy)
    #    (Note: ellipse orientation does not actually affect the rectangle size; we just take the axis lengths.)
    x1 = max(cx - a, 0)
    y1 = max(cy - b, 0)
    x2 = min(cx + a, w - 1)
    y2 = min(cy + b, h - 1)

    ellipse_area = np.pi * (a / 2.0) * (b / 2.0)  # area of ellipse with radii (a/2, b/2)
    ratio = ellipse_area / image_area

    # 6) If ratio outside [min_ratio, max_ratio], skip cropping
    if ratio < min_ratio or ratio > max_ratio:
        cv2.imwrite(dst_path, img)
        return

    # 7) Finally crop the rectangle
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(dst_path, cropped)


def process_folder(input_dir, output_dir, thresh_val, min_ratio, max_ratio):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input '{input_dir}' does not exist.")
    os.makedirs(output_dir, exist_ok=True)

    VALID = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_files = os.listdir(input_dir)
    # Filter only valid image files
    to_process = [
        fname for fname in all_files
        if os.path.splitext(fname)[1].lower() in VALID
    ]

    # Wrap with tqdm to show a progress bar
    for fname in tqdm(to_process, desc="Cropping images", ncols=80):
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        crop_paper_style(src, dst, thresh_val, min_ratio, max_ratio)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="“Paper‐style” rectangular crop for ISIC2019 (with progress bar).")
    p.add_argument("--input_dir",  type=str, required=True, help="Path to ISIC2019_uncropped")
    p.add_argument("--output_dir", type=str, required=True, help="Where to place cropped images")
    p.add_argument("--threshold",  type=int, default=50,
                   help="Fixed gray threshold (0–255). Try 10–50 for derm images.")
    p.add_argument("--min_ratio",  type=float, default=0.01,
                   help="Min ellipse‐area/image‐area to crop (0.01 is usually fine).")
    p.add_argument("--max_ratio",  type=float, default=0.9,
                   help="Max ellipse‐area/image‐area to crop (0.9 is usually fine).")
    args = p.parse_args()

    process_folder(
        input_dir  = args.input_dir,
        output_dir = args.output_dir,
        thresh_val = args.threshold,
        min_ratio  = args.min_ratio,
        max_ratio  = args.max_ratio
    )
