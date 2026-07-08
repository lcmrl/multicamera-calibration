"""Diagnostic: trace CCT detection pipeline rejection reasons per image."""
from __future__ import annotations
import sys
import math
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cct_detect.detector import CCTDetector, canonical_code


def diagnose(image_path: Path, min_circ: float = 0.75):
    detector = CCTDetector(n_bits=14)
    image_bgr = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image: {image_path.name}  ({W}×{H})")
    print(f"{'='*60}")

    # ── replicate _find_candidates with instrumentation ──────────
    min_area = max(30, int(H * W * 1e-6))
    max_area = int(H * W * 0.005)
    print(f"  contour area range: [{min_area}, {max_area}]")

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaries = []
    for off in (-20, -10, 0, 10, 20):
        tv = int(np.clip(otsu_val + off, 30, 230))
        _, bw = cv2.threshold(blurred, tv, 255, cv2.THRESH_BINARY)
        binaries.append(bw)
    for bs in (31, 61):
        for C in (10, 20):
            bw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, -C)
            binaries.append(bw)

    rej = {"area": 0, "circularity": 0, "ellipse_size": 0, "aspect": 0, "contour_pts": 0}
    n_total = 0
    scored: list[tuple[float, tuple]] = []
    for bw in binaries:
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            n_total += 1
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                rej["area"] += 1
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < min_circ:
                rej["circularity"] += 1
                continue
            if len(cnt) < 10:
                rej["contour_pts"] += 1
                continue
            ell = cv2.fitEllipse(cnt)
            (cx, cy), (aw, ah), ang = ell
            major = max(aw, ah) / 2.0
            minor = min(aw, ah) / 2.0
            if major < 5 or minor < 4 or major > 80:
                rej["ellipse_size"] += 1
                continue
            if minor / (major + 1e-9) < 0.3:
                rej["aspect"] += 1
                continue
            scored.append((circ, ell))

    # Dedup
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates, centers = [], []
    for _, ell in scored:
        c = np.array([ell[0][0], ell[0][1]])
        r = max(ell[1]) / 2.0
        if any(np.linalg.norm(c - fc) < max(4.0, r * 0.4) for fc in centers):
            continue
        candidates.append(ell)
        centers.append(c)

    print(f"  raw contours: {n_total:,}  rejections: {rej}")
    print(f"  candidates after dedup: {len(candidates)}")

    # ── trace each candidate ─────────────────────────────────────
    n_border = n_bad_ring = n_no_decode = n_ok = 0
    for ell in candidates:
        patch = detector._rectify_patch(gray, ell, out_size=200)
        if patch is None:
            n_border += 1
            continue
        if not detector._validate_rectified(patch):
            n_bad_ring += 1
            continue
        result = detector._decode_patch(patch)
        if result is None:
            n_no_decode += 1
            continue
        n_ok += 1

    print(f"  border-clipped:       {n_border}")
    print(f"  ring-validation fail: {n_bad_ring}")
    print(f"  decode fail:          {n_no_decode}")
    print(f"  decoded OK:           {n_ok}  (before dedup)")

    # ── also check: how many ellipses are near image border ──────
    near_border = sum(
        1 for c in centers
        if c[0] < 50 or c[0] > W - 50 or c[1] < 50 or c[1] > H - 50
    )
    print(f"  candidates near border (<50px): {near_border}")

    # ── circularity histogram of all passing-area contours ───────
    all_circ: list[float] = []
    for bw in binaries:
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            all_circ.append(circ)
    arr = np.array(all_circ)
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 0.85), (0.85, 1.01)]:
        count = int(np.sum((arr >= lo) & (arr < hi)))
        print(f"  circularity [{lo:.2f},{hi:.2f}): {count}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "sample"
    for img in sorted(base.glob("*.JPG")):
        if "_det" not in img.stem:
            diagnose(img)
