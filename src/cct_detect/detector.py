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

from .refinement import (
    CenterEstimate,
    prepare_refinement_image,
    refine_projected_center,
)

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
    # ``center`` is the best available estimate of the projected physical
    # target centre.  ``ellipse_center`` is retained because the ellipse fit
    # is only a provisional image measurement under perspective/distortion.
    ellipse_center: tuple[float, float] | None = None
    center_correction_px: tuple[float, float] | None = None
    center_method: str = "ellipse"
    center_covariance_px2: tuple[float, float, float, float] | None = None


@dataclass
class DetectionDiagnostics:
    """Stage counts and rejection locations from the production detector."""

    counts: dict[str, int]
    rejected: list[dict[str, object]]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CCTDetector:
    def __init__(self, n_bits: int = 14,
                 valid_id_range: tuple[int, int] | None = None,
                 min_target_radius: float = 5.0,
                 max_target_radius: float | None = 150.0):
        self.n_bits = n_bits
        self.valid_id_range = valid_id_range  # (min_id, max_id) inclusive, or None
        self.min_target_radius = float(min_target_radius)
        # Keep the conservative historical default.  ``None`` is treated as
        # the same fixed 150 px cap; an image-relative cap is deliberately not
        # used because it makes arbitrary large scene contours eligible.
        self.max_target_radius = 150.0 if max_target_radius is None else float(max_target_radius)

    # ---- 1. candidate ellipses ------------------------------------------

    def _find_candidates(self, gray: np.ndarray, min_circ: float = 0.60):
        """
        Find ellipse candidates from multiple binarisations.
        """
        H, W = gray.shape[:2]
        max_radius = self.max_target_radius
        min_area = max(30, int(H * W * 1e-6))
        max_area = int(H * W * 0.008)

        scored: list[tuple[float, tuple]] = []
        rejection_counts = {
            "area": 0, "perimeter": 0, "circularity": 0,
            "contour_points": 0, "size": 0, "aspect": 0,
        }

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Otsu with broader offset sweep
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaries = []
        for off in (-30, -20, -10, 0, 10, 20, 30):
            tv = int(np.clip(otsu_val + off, 30, 230))
            _, bw = cv2.threshold(blurred, tv, 255, cv2.THRESH_BINARY)
            binaries.append(bw)

        # Adaptive
        for bs in (31, 61, 91):
            for C in (5, 10, 20):
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
                    rejection_counts["area"] += 1
                    continue
                perim = cv2.arcLength(cnt, True)
                if perim < 1:
                    rejection_counts["perimeter"] += 1
                    continue
                circ = 4.0 * math.pi * area / (perim * perim)
                # This is deliberately a hard gate: arbitrary scene contours
                # must not reach the decoder merely because they can be fit by
                # an ellipse.
                if circ < min_circ:
                    rejection_counts["circularity"] += 1
                    continue
                if len(cnt) < 10:
                    rejection_counts["contour_points"] += 1
                    continue
                ell = cv2.fitEllipse(cnt)
                (cx, cy), (aw, ah), ang = ell
                major = max(aw, ah) / 2.0
                minor = min(aw, ah) / 2.0
                if (major < self.min_target_radius or minor < 4.0
                        or major > max_radius):
                    rejection_counts["size"] += 1
                    continue
                if minor / (major + 1e-9) < 0.30:
                    rejection_counts["aspect"] += 1
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
        self._last_candidate_rejections = rejection_counts
        return result

    # ---- 2. affine rectification ---------------------------------------

    @staticmethod
    def _rectify_patch(
        gray: np.ndarray,
        ell,
        out_size: int = 200,
        return_mask: bool = False,
    ):
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

        # Source ROI — pad image if near border rather than rejecting
        r_min = int(round(cy - half))
        r_max = int(round(cy + half))
        c_min = int(round(cx - half))
        c_max = int(round(cx + half))
        source_valid = np.ones(gray.shape[:2], dtype=np.uint8)
        if r_min < 0 or c_min < 0 or r_max > H or c_max > W:
            pad = half + 2
            gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                      cv2.BORDER_REFLECT_101)
            source_valid = cv2.copyMakeBorder(
                source_valid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0,
            )
            cx += pad
            cy += pad
            H, W = gray.shape[:2]
            r_min = int(round(cy - half))
            r_max = int(round(cy + half))
            c_min = int(round(cx - half))
            c_max = int(round(cx + half))

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
        if not return_mask:
            return warped
        valid = cv2.warpAffine(
            source_valid, M, (out_size, out_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        return warped, valid

    # ---- 3. strict ring validation on binarised rectified patch ---------

    @staticmethod
    def _validate_rectified(
        patch_gray: np.ndarray,
        valid_mask: np.ndarray | None = None,
        sample_n: int = 36,
    ):
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

        n_white_center = n_black_guard = n_white_code = 0
        n_center = n_guard = n_code = 0

        for j in range(sample_n):
            angle = 2.0 * math.pi * j / sample_n
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Centre sample at 0.5 r1
            x = int(round(X0 + 0.5 * r1 * cos_a))
            y = int(round(Y0 + 0.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] > 0:
                    n_white_center += 1
                n_center += 1

            # Guard ring at 1.5 r1
            x = int(round(X0 + 1.5 * r1 * cos_a))
            y = int(round(Y0 + 1.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] == 0:
                    n_black_guard += 1
                n_guard += 1

            # Code ring at 2.5 r1
            x = int(round(X0 + 2.5 * r1 * cos_a))
            y = int(round(Y0 + 2.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] > 0:
                    n_white_code += 1
                n_code += 1

        # A padded/border patch must never pass by normalising its evidence to
        # only the visible samples.  The historical detector required every
        # angular sample to be present; retain that invariant when a validity
        # mask is available.
        if min(n_center, n_guard, n_code) < sample_n:
            return False

        # Allow tolerance for perspective residuals and uneven illumination
        if n_white_center < sample_n * 0.70:
            return False
        if n_black_guard < sample_n * 0.70:
            return False
        n_black_code = sample_n - n_white_code
        if n_white_code < 2 or n_black_code < 2:
            return False

        return True

    @staticmethod
    def _validate_radial_consistency(
        patch_gray: np.ndarray,
        valid_mask: np.ndarray | None = None,
        sample_n: int = 48,
    ) -> bool:
        """Require the three rings to persist over radial samples.

        The original single-radius checks are retained as the first gate.  A
        second, independent check rejects a crescent, text glyph, or other
        arbitrary ellipse that happens to match at one radius: the white
        centre and black guard must remain stable over neighbouring radii, and
        each code sector must keep the same binary value through the code
        annulus.
        """
        sz = patch_gray.shape[0]
        center = sz / 2.0
        r1 = sz / 6.0
        _, bw = cv2.threshold(
            patch_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        def ring_samples(radius: float) -> np.ndarray | None:
            values: list[int] = []
            for j in range(sample_n):
                angle = 2.0 * math.pi * j / sample_n
                x = int(round(center + radius * r1 * math.cos(angle)))
                y = int(round(center + radius * r1 * math.sin(angle)))
                if not (0 <= x < sz and 0 <= y < sz):
                    return None
                if valid_mask is not None and not bool(valid_mask[y, x]):
                    return None
                values.append(int(bw[y, x] > 0))
            return np.asarray(values, dtype=np.int8)

        centre_profiles = [ring_samples(radius) for radius in (0.30, 0.50, 0.70)]
        guard_profiles = [ring_samples(radius) for radius in (1.25, 1.50, 1.75)]
        code_profiles = [ring_samples(radius) for radius in (2.25, 2.50, 2.75)]
        if any(profile is None for profile in (*centre_profiles, *guard_profiles, *code_profiles)):
            return False

        centre = np.stack([profile for profile in centre_profiles if profile is not None])
        guard = np.stack([profile for profile in guard_profiles if profile is not None])
        code = np.stack([profile for profile in code_profiles if profile is not None])
        if any(float(np.mean(profile)) < 0.70 for profile in centre):
            return False
        if any(float(np.mean(profile == 0)) < 0.70 for profile in guard):
            return False

        # Radial agreement is evaluated per angle, so a valid sector may be
        # white or black but cannot change value merely with radius.
        code_vote = np.mean(code, axis=0) >= 0.5
        agreement = np.mean(code == code_vote[None, :])
        if float(agreement) < 0.80:
            return False
        if int(np.count_nonzero(code_vote)) < 2 or int(np.count_nonzero(~code_vote)) < 2:
            return False
        return True

    # ---- 4. decode angular profile from rectified grayscale patch -------

    def _decode_patch(self, patch_gray: np.ndarray, valid_mask: np.ndarray | None = None):
        """
        Sample the code ring on the rectified grayscale patch and decode.
        Returns (canon, raw, bits_list, confidence, pattern_str) or None.
        """
        sz = patch_gray.shape[0]
        X0 = Y0 = sz / 2.0
        r1 = sz / 6.0

        n_ang = max(360, 60 * self.n_bits)  # 840 for 14 bits
        # The decoder intentionally samples the established code annulus,
        # including its outer edge; the strict ring checks below and the
        # residual threshold are what reject scene patterns that only happen
        # to look circular.
        inner_r = 2.4 * r1
        outer_r = 3.2 * r1
        radii = np.linspace(inner_r, outer_r, 7)
        # Use the established integer samples and never interpolate missing
        # rays.  Interpolation can fabricate a plausible code from a clipped
        # border or unrelated scene contour.
        profile = np.zeros(n_ang, dtype=np.float64)
        for i in range(n_ang):
            angle = 2.0 * math.pi * i / n_ang
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            vals = []
            for rad in radii:
                x = int(round(X0 + rad * cos_a))
                y = int(round(Y0 + rad * sin_a))
                if (0 <= x < sz and 0 <= y < sz
                        and (valid_mask is None or valid_mask[y, x])):
                    vals.append(float(patch_gray[y, x]))
            if not vals:
                return None
            profile[i] = np.mean(vals)

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
        if p_hi - p_lo < 8:
            return None
        thresh = (p_lo + p_hi) / 2.0

        best_score = float('inf')
        best = None
        best_by_id: dict[int, tuple[float, tuple]] = {}
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
            raw = sum((1 << k) for k, b in enumerate(bits) if b)
            canon = canonical_code(raw, self.n_bits)
            pat = ''.join(str(b) for b in bits)
            candidate = (score, (raw, bits.tolist(), margin, pat, mse))
            prior = best_by_id.get(canon)
            if prior is None or score < prior[0]:
                best_by_id[canon] = candidate
            if score < best_score:
                best_score = score
                best = (canon, raw, bits.tolist(), margin, pat, mse)

        if best is None:
            return None

        canon, raw, bits, margin, pat, mse = best
        range_sq = (p_hi - p_lo) ** 2
        if range_sq > 0 and mse / range_sq > 0.030:
            return None

        # Reject block codes (e.g. 0000001111111) — real CCT codes have >= 4
        # transitions between sectors; simple all-ones blocks are noise FPs.
        n1 = sum(bits)
        transitions = sum(1 for i in range(self.n_bits)
                         if bits[i] != bits[(i + 1) % self.n_bits])
        if transitions < 4:
            return None

        # A close competing ID is an ambiguity, not a confident detection.
        ranked_ids = sorted(best_by_id.values(), key=lambda item: item[0])
        if len(ranked_ids) > 1 and ranked_ids[1][0] - ranked_ids[0][0] < 0.006:
            return None
        return canon, raw, bits, margin, pat

    # ---- 5. main pipeline -----------------------------------------------

    def detect_with_diagnostics(
        self, image_bgr: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> tuple[list[Detection], DetectionDiagnostics]:
        raw_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw_gray)
        candidates = self._find_candidates(gray)
        candidates.sort(key=lambda e: max(e[1]), reverse=True)

        counts = {
            "candidates": len(candidates), "validated": 0, "decoded": 0,
            "refined_centers": 0, "reject_validate": 0, "reject_decode": 0,
            "spatial_duplicates": 0, "id_duplicates": 0,
        }
        counts.update({f"candidate_reject_{key}": int(value) for key, value in getattr(self, "_last_candidate_rejections", {}).items()})
        rejected: list[dict[str, object]] = []
        detections: list[Detection] = []
        used: list[np.ndarray] = []

        for ell in candidates:
            (cx, cy), (aw, ah), ang = ell
            r = max(aw, ah) / 2.0

            # Rectify
            patch, valid_mask = self._rectify_patch(gray, ell, out_size=200, return_mask=True)
            if patch is None:
                continue

            # Validate ring structure on binarised rectified patch
            if not self._validate_rectified(patch, valid_mask):
                counts["reject_validate"] += 1
                rejected.append({"x": float(cx), "y": float(cy), "stage": "validate"})
                continue
            if not self._validate_radial_consistency(patch, valid_mask):
                counts["reject_validate"] += 1
                rejected.append({"x": float(cx), "y": float(cy), "stage": "radial_validate"})
                continue
            counts["validated"] += 1

            # Decode
            result = self._decode_patch(patch, valid_mask)
            if result is None:
                counts["reject_decode"] += 1
                rejected.append({"x": float(cx), "y": float(cy), "stage": "decode"})
                continue
            counts["decoded"] += 1

            canon, raw, bits, margin, pat = result

            # Keep acceptance and deduplication tied to the conservative
            # ellipse measurement.  Calibrated conic refinement is applied
            # only after the candidate has survived every detector gate.
            center = np.array([cx, cy], dtype=float)

            # Dedup: full target outer edge ≈ 3× inner-circle radius;
            # use 4× to also catch candidates from outer rings of the same
            # target while not eating adjacent distinct targets.
            c_arr = np.array([cx, cy])
            for i, uc in enumerate(used):
                det_r = max(detections[i].ellipse[1]) / 2.0
                # Keep the original broad spatial suppression.  A single
                # printed/painted shape often yields several contour scales;
                # retaining them is a direct path to different hallucinated
                # IDs surviving the later ID-level deduplication.
                dup_r = max(20.0, max(r, det_r) * 4.0)
                if np.linalg.norm(c_arr - uc) < dup_r:
                    if margin > detections[i].confidence:
                        detections[i] = Detection(
                            target_id=canon, center=(float(center[0]), float(center[1])), ellipse=ell,
                            confidence=margin, code_bits=bits, raw_code=raw,
                            pattern=pat,
                            ellipse_center=(float(cx), float(cy)),
                            center_correction_px=(0.0, 0.0),
                            center_method="ellipse",
                        )
                        used[i] = c_arr
                    counts["spatial_duplicates"] += 1
                    break
            else:
                detections.append(Detection(
                    target_id=canon, center=(float(center[0]), float(center[1])), ellipse=ell,
                    confidence=margin, code_bits=bits, raw_code=raw,
                    pattern=pat,
                    ellipse_center=(float(cx), float(cy)),
                    center_correction_px=(0.0, 0.0),
                    center_method="ellipse",
                ))
                used.append(c_arr)

        detections.sort(key=lambda d: (d.center[1], d.center[0]))

        # Final ID-level dedup: if the same target_id appears more than once,
        # keep only the highest-confidence instance.
        seen: dict[int, int] = {}  # target_id -> index in best_detections
        best_detections: list[Detection] = []
        for det in detections:
            if det.target_id in seen:
                idx = seen[det.target_id]
                if det.confidence > best_detections[idx].confidence:
                    best_detections[idx] = det
                counts["id_duplicates"] += 1
            else:
                seen[det.target_id] = len(best_detections)
                best_detections.append(det)
        best_detections.sort(key=lambda d: (d.center[1], d.center[0]))

        # Filter by valid ID range if specified
        if self.valid_id_range is not None:
            lo, hi = self.valid_id_range
            best_detections = [d for d in best_detections if lo <= d.target_id <= hi]

        # Refine only finalized detections.  This makes the calibrated center
        # estimate an accuracy improvement, never a source of extra accepted
        # candidates or altered spatial/ID deduplication decisions.
        refined_detections: list[Detection] = []
        refinement_gray = (
            prepare_refinement_image(raw_gray) if intrinsics is not None else raw_gray
        )
        for det in best_detections:
            ellipse_center = det.ellipse_center or (
                float(det.ellipse[0][0]), float(det.ellipse[0][1])
            )
            estimate: CenterEstimate = refine_projected_center(
                refinement_gray, det.ellipse, intrinsics,
                image_is_prepared=intrinsics is not None,
            )
            if estimate.valid:
                counts["refined_centers"] += 1
                center = estimate.projected_center_px
                correction = (
                    float(center[0] - ellipse_center[0]),
                    float(center[1] - ellipse_center[1]),
                )
                method = estimate.method
                covariance = (
                    tuple(float(v) for v in estimate.conditional_covariance_px2.reshape(-1))
                    if estimate.conditional_covariance_px2 is not None else None
                )
            else:
                center = np.asarray(ellipse_center, dtype=float)
                correction = det.center_correction_px or (0.0, 0.0)
                method = "ellipse"
                covariance = None
            refined_detections.append(Detection(
                target_id=det.target_id,
                center=(float(center[0]), float(center[1])),
                ellipse=det.ellipse,
                confidence=det.confidence,
                code_bits=det.code_bits,
                raw_code=det.raw_code,
                pattern=det.pattern,
                ellipse_center=(float(ellipse_center[0]), float(ellipse_center[1])),
                center_correction_px=correction,
                center_method=method,
                center_covariance_px2=covariance,
            ))
        best_detections = sorted(
            refined_detections,
            key=lambda d: (d.center[1], d.center[0]),
        )

        return best_detections, DetectionDiagnostics(counts=counts, rejected=rejected)

    def detect(self, image_bgr: np.ndarray, intrinsics: np.ndarray | None = None) -> list[Detection]:
        detections, _ = self.detect_with_diagnostics(image_bgr, intrinsics=intrinsics)
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
