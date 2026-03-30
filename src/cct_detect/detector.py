"""
CCT (Concentric Circular coded Target) detector.

Target geometry (in units of the inner white circle semi-axis R):
  0 .. R     White filled centre
  R .. 2R    Black guard ring
  2R .. 3R   Code ring (14 sectors, white=1, black=0)

Pipeline:
  1. Otsu + adaptive binarisation  ->  contour extraction
  2. Ellipse fit on circular-enough contours
  3. Affine rectification to a canonical circle
  4. Strict ring validation on the binarised rectified patch
  5. Angular profile sampling on the grayscale rectified patch
  6. Phase-search decode, canonical code via minimum cyclic rotation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Code utilities
# ---------------------------------------------------------------------------

def _rotate_left(v: int, k: int, n: int) -> int:
    mask = (1 << n) - 1
    return ((v << k) & mask) | ((v & mask) >> (n - k))


def canonical_code(value: int, n_bits: int) -> int:
    best = value
    for k in range(1, n_bits):
        best = min(best, _rotate_left(value, k, n_bits))
    return best


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    target_id: int
    center: tuple[float, float]
    ellipse: tuple[tuple[float, float], tuple[float, float], float]
    confidence: float
    code_bits: list[int]
    raw_code: int
    pattern: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CCTDetector:
    def __init__(self, n_bits: int = 14):
        self.n_bits = n_bits

    # ---- 1. candidate ellipses ------------------------------------------

    def _find_candidates(self, gray: np.ndarray, min_circ: float = 0.75):
        """
        Find ellipse candidates from multiple binarisations.
        Higher circularity threshold than before - real CCT inner circles
        are very circular even under moderate perspective.
        """
        H, W = gray.shape[:2]
        min_area = max(30, int(H * W * 1e-6))
        max_area = int(H * W * 0.005)

        scored: list[tuple[float, tuple]] = []

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Otsu
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaries = []
        for off in (-20, -10, 0, 10, 20):
            tv = int(np.clip(otsu_val + off, 30, 230))
            _, bw = cv2.threshold(blurred, tv, 255, cv2.THRESH_BINARY)
            binaries.append(bw)

        # Adaptive
        for bs in (31, 61):
            for C in (10, 20):
                bw = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, bs, -C,
                )
                binaries.append(bw)

        for bw in binaries:
            contours, _ = cv2.findContours(
                bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                perim = cv2.arcLength(cnt, True)
                if perim < 1:
                    continue
                circ = 4.0 * math.pi * area / (perim * perim)
                if circ < min_circ:
                    continue
                if len(cnt) < 10:
                    continue
                ell = cv2.fitEllipse(cnt)
                (cx, cy), (aw, ah), ang = ell
                major = max(aw, ah) / 2.0
                minor = min(aw, ah) / 2.0
                if major < 5 or minor < 4 or major > 80:
                    continue
                if minor / (major + 1e-9) < 0.3:
                    continue
                scored.append((circ, ell))

        # Dedup
        scored.sort(key=lambda x: x[0], reverse=True)
        result, centers = [], []
        for _, ell in scored:
            c = np.array([ell[0][0], ell[0][1]])
            r = max(ell[1]) / 2.0
            if any(np.linalg.norm(c - fc) < max(4.0, r * 0.4) for fc in centers):
                continue
            result.append(ell)
            centers.append(c)
        return result

    # ---- 2. affine rectification ---------------------------------------

    @staticmethod
    def _rectify_patch(gray: np.ndarray, ell, out_size: int = 200):
        """
        Warp the neighbourhood of *ell* so the ellipse becomes a circle
        centred in a (out_size x out_size) patch.  Returns the warped
        grayscale patch or None if the source region is out of bounds.
        """
        (cx, cy), (aw, ah), ang = ell
        H, W = gray.shape[:2]

        # The 3x outer ellipse
        major3 = max(aw, ah) * 1.6  # a bit more than 3x radius
        half = int(math.ceil(major3))

        # Source ROI (may be clipped)
        r_min = int(round(cy - half))
        r_max = int(round(cy + half))
        c_min = int(round(cx - half))
        c_max = int(round(cx + half))
        if r_min < 0 or c_min < 0 or r_max > H or c_max > W:
            return None

        # Build affine: map 3x-ellipse bounding box corners to square
        theta = math.radians(ang)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        a3 = aw / 2.0 * 3.0
        b3 = ah / 2.0 * 3.0

        # 3 source points on the 3x-ellipse (0deg, 90deg, and center)
        # Point at 0 deg on ellipse
        src_pts = np.float32([
            [cx + cos_t * a3,           cy + sin_t * a3],
            [cx - sin_t * b3,           cy + cos_t * b3],
            [cx,                         cy],
        ])
        # Corresponding destination: circle of radius out_size/2
        r_out = out_size / 2.0
        dst_pts = np.float32([
            [out_size / 2.0 + r_out,  out_size / 2.0],
            [out_size / 2.0,          out_size / 2.0 + r_out],
            [out_size / 2.0,          out_size / 2.0],
        ])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(gray, M, (out_size, out_size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=128)
        return warped

    # ---- 3. strict ring validation on binarised rectified patch ---------

    @staticmethod
    def _validate_rectified(patch_gray: np.ndarray, sample_n: int = 36):
        """
        On the rectified 200x200 patch, binarise and check:
          - centre ring (0.5 r1) is ALL white
          - guard ring  (1.5 r1) is ALL black
          - code ring   (2.5 r1) has at least 2 white AND 2 black samples
        where r1 = patch_size / 6  (= radius of inner white circle).
        
        Returns True if the pattern matches a valid CCT.
        """
        sz = patch_gray.shape[0]
        X0 = Y0 = sz / 2.0
        r1 = sz / 6.0   # inner white circle radius

        # Binarise the rectified patch
        _, bw = cv2.threshold(patch_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        n_white_center = 0
        n_black_guard = 0
        n_white_code = 0
        n_valid = 0

        for j in range(sample_n):
            angle = 2.0 * math.pi * j / sample_n
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Centre sample at 0.5 r1
            x = int(round(X0 + 0.5 * r1 * cos_a))
            y = int(round(Y0 + 0.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz:
                if bw[y, x] > 0:
                    n_white_center += 1
                n_valid += 1

            # Guard ring at 1.5 r1
            x = int(round(X0 + 1.5 * r1 * cos_a))
            y = int(round(Y0 + 1.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz:
                if bw[y, x] == 0:
                    n_black_guard += 1

            # Code ring at 2.5 r1
            x = int(round(X0 + 2.5 * r1 * cos_a))
            y = int(round(Y0 + 2.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz:
                if bw[y, x] > 0:
                    n_white_code += 1

        if n_valid == 0:
            return False

        # Strict checks (allow small tolerance for perspective residuals)
        if n_white_center < sample_n * 0.85:
            return False
        if n_black_guard < sample_n * 0.85:
            return False
        n_black_code = sample_n - n_white_code
        if n_white_code < 2 or n_black_code < 2:
            return False

        return True

    # ---- 4. decode angular profile from rectified grayscale patch -------

    def _decode_patch(self, patch_gray: np.ndarray):
        """
        Sample the code ring on the rectified grayscale patch and decode.
        Returns (canon, raw, bits_list, confidence, pattern_str) or None.
        """
        sz = patch_gray.shape[0]
        X0 = Y0 = sz / 2.0
        r1 = sz / 6.0

        n_ang = max(360, 60 * self.n_bits)  # 840 for 14 bits
        inner_r = 2.4 * r1
        outer_r = 3.2 * r1
        radii = np.linspace(inner_r, outer_r, 7)

        profile = np.zeros(n_ang, dtype=np.float64)
        for i in range(n_ang):
            angle = 2.0 * math.pi * i / n_ang
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            vals = []
            for rad in radii:
                x = int(round(X0 + rad * cos_a))
                y = int(round(Y0 + rad * sin_a))
                if 0 <= x < sz and 0 <= y < sz:
                    vals.append(float(patch_gray[y, x]))
            if vals:
                profile[i] = np.mean(vals)
            else:
                return None

        # Smooth
        ks = max(3, n_ang // 80)
        if ks % 2 == 0:
            ks += 1
        pad = ks // 2
        padded = np.concatenate([profile[-pad:], profile, profile[:pad]])
        kernel = np.ones(ks) / ks
        smooth = np.convolve(padded, kernel, mode='valid')[:n_ang]

        # Decode
        step = n_ang / self.n_bits
        p_lo = np.percentile(smooth, 15)
        p_hi = np.percentile(smooth, 85)
        if p_hi - p_lo < 15:
            return None
        thresh = (p_lo + p_hi) / 2.0

        best_score = float('inf')
        best = None
        n_phases = max(1, int(round(step)))

        for ph in range(n_phases):
            means = []
            for s in range(self.n_bits):
                lo_i = int(round(ph + s * step))
                hi_i = int(round(ph + (s + 1) * step))
                idx = np.arange(lo_i, hi_i) % n_ang
                if len(idx) == 0:
                    break
                means.append(float(np.mean(smooth[idx])))
            if len(means) != self.n_bits:
                continue

            arr = np.array(means)
            bits = (arr >= thresh).astype(int)

            transitions = int(np.sum(bits != np.roll(bits, 1)))
            if transitions < 2 or transitions > 13:
                continue
            n1 = int(np.sum(bits))
            if n1 < 2 or n1 > self.n_bits - 2:
                continue

            # MSE
            recon = np.zeros(n_ang)
            for s in range(self.n_bits):
                lo_i = int(round(ph + s * step))
                hi_i = int(round(ph + (s + 1) * step))
                idx = np.arange(lo_i, hi_i) % n_ang
                recon[idx] = p_hi if bits[s] else p_lo
            mse = float(np.mean((smooth - recon) ** 2))
            margin = float(np.mean(np.abs(arr - thresh)))
            score = mse - 0.15 * margin

            if score < best_score:
                best_score = score
                raw = sum((1 << k) for k, b in enumerate(bits) if b)
                canon = canonical_code(raw, self.n_bits)
                pat = ''.join(str(b) for b in bits)
                best = (canon, raw, bits.tolist(), margin, pat, mse)

        if best is None:
            return None

        canon, raw, bits, margin, pat, mse = best
        range_sq = (p_hi - p_lo) ** 2
        if range_sq > 0 and mse / range_sq > 0.020:
            return None

        # Reject trivially simple codes (e.g. ID 1, 3, 7) which are
        # common false positives from noise.
        n1 = sum(bits)
        transitions = sum(1 for i in range(self.n_bits)
                         if bits[i] != bits[(i + 1) % self.n_bits])
        if n1 < 4 and transitions < 4:
            return None

        return canon, raw, bits, margin, pat

    # ---- 5. main pipeline -----------------------------------------------

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        candidates = self._find_candidates(gray)

        # Process largest candidates first (better decode quality)
        candidates.sort(key=lambda e: max(e[1]), reverse=True)

        detections: list[Detection] = []
        used: list[np.ndarray] = []

        for ell in candidates:
            (cx, cy), (aw, ah), ang = ell
            r = max(aw, ah) / 2.0

            # Rectify
            patch = self._rectify_patch(gray, ell, out_size=200)
            if patch is None:
                continue

            # Validate ring structure on binarised rectified patch
            if not self._validate_rectified(patch):
                continue

            # Decode
            result = self._decode_patch(patch)
            if result is None:
                continue

            canon, raw, bits, margin, pat = result

            # Dedup: use a radius that covers the full target extent
            # (target outer edge ≈ 3× the fitted inner-circle radius)
            c_arr = np.array([cx, cy])
            dup = False
            for i, uc in enumerate(used):
                det_r = max(detections[i].ellipse[1]) / 2.0
                dup_r = max(15.0, max(r, det_r) * 3.5)
                if np.linalg.norm(c_arr - uc) < dup_r:
                    if margin > detections[i].confidence:
                        detections[i] = Detection(
                            target_id=canon, center=(cx, cy), ellipse=ell,
                            confidence=margin, code_bits=bits, raw_code=raw,
                            pattern=pat,
                        )
                        used[i] = c_arr
                    dup = True
                    break
            if not dup:
                detections.append(Detection(
                    target_id=canon, center=(cx, cy), ellipse=ell,
                    confidence=margin, code_bits=bits, raw_code=raw,
                    pattern=pat,
                ))
                used.append(c_arr)

        detections.sort(key=lambda d: (d.center[1], d.center[0]))
        return detections

    # ---- 6. annotation ---------------------------------------------------

    def annotate(self, image_bgr: np.ndarray,
                 detections: Sequence[Detection]) -> np.ndarray:
        out = image_bgr.copy()
        for det in detections:
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            cv2.ellipse(out, det.ellipse, (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
            label = str(det.target_id)
            cv2.putText(out, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return out
