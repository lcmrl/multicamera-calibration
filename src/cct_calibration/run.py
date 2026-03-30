from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyceres
import rerun as rr

from cct_detect.detector import CCTDetector as _CCTDetector


@dataclass
class ImageDetections:
    image_path: Path
    image_index: int
    width: int
    height: int
    detections: Dict[int, np.ndarray]


@dataclass
class CalibrationState:
    intrinsics: np.ndarray
    poses: Dict[int, np.ndarray]
    points: Dict[int, np.ndarray]
    seed_image_indices: List[int]
    inlier_track_ids: List[int]


@dataclass
class CameraCalibrationResult:
    camera_name: str
    image_size: tuple[int, int]
    used_images: int
    registered_images: int
    tracks: int
    observations: int
    summary: str
    intrinsics: np.ndarray
    mean_reprojection_error: float
    rms_reprojection_error: float
    iterations: int
    initial_cost: float
    final_cost: float
    convergence_plot_path: Path
    rerun_recording_path: Path
    scene_data_path: Path
    targets_3d_path: Path
    cameras_path: Path
    colmap_dir_path: Path
    detections_dir_path: Path
    target_detections_path: Path


@dataclass
class SolverDiagnostics:
    summary: str
    iterations: int
    initial_cost: float
    final_cost: float
    cost_history: List[float]


@dataclass
class DetectionOnlyResult:
    camera_name: str
    total_images: int
    used_images: int
    detections_dir_path: Path
    target_detections_path: Path


def parse_valid_ids_file(path: Path) -> Set[int]:
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    valid_ids = {int(token) for token in tokens}
    if not valid_ids:
        raise ValueError(f"No valid IDs found in {path}")
    return valid_ids


def resolve_valid_ids(valid_ids_file: Path | None, valid_id_max: int | None) -> Set[int] | None:
    if valid_ids_file is not None:
        return parse_valid_ids_file(valid_ids_file)
    if valid_id_max is not None:
        return set(range(valid_id_max + 1))
    return None


def normalize_target_id(raw_id: int, valid_ids: Set[int] | None, max_hamming_distance: int) -> tuple[int | None, bool]:
    if valid_ids is None:
        return raw_id, False
    if raw_id in valid_ids:
        return raw_id, False

    best_id: int | None = None
    best_distance: int | None = None
    is_ambiguous = False
    for candidate in valid_ids:
        distance = (raw_id ^ candidate).bit_count()
        if best_distance is None or distance < best_distance:
            best_id = candidate
            best_distance = distance
            is_ambiguous = False
        elif distance == best_distance:
            is_ambiguous = True

    if best_id is None or best_distance is None or is_ambiguous or best_distance > max_hamming_distance:
        return None, False
    return best_id, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a single camera from CCTDecode detections and pyceres bundle adjustment."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains the photos_* folders.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera name to calibrate, for example camera_0. Repeat to run multiple cameras.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=10,
        help="Keep only images with more than this many unique detections.",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=8,
        help="Minimum number of shared tracks required to seed or register a view.",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=12.0,
        help="Drop initialized observations above this pixel error before bundle adjustment.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Safety cap on pyceres iterations. Early termination is driven by loss convergence.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=3.0,
        help="Huber loss delta in pixels.",
    )
    parser.add_argument(
        "--loss-relative-tolerance",
        type=float,
        default=1e-5,
        help="Relative loss-change threshold for convergence-based stopping.",
    )
    parser.add_argument(
        "--loss-patience",
        type=int,
        default=5,
        help="Terminate after this many consecutive successful iterations below the relative loss threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_output"),
        help="Directory for convergence plots, pose visualizations, and JSON summaries.",
    )
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Open a Rerun viewer with oriented camera poses and target-center points after calibration.",
    )
    parser.add_argument(
        "--valid-ids-file",
        type=Path,
        default=None,
        help="Optional text file with whitespace- or comma-separated valid target IDs.",
    )
    parser.add_argument(
        "--valid-id-max",
        type=int,
        default=None,
        help="Optional maximum valid target ID. If set, valid IDs are assumed to be 0..N.",
    )
    parser.add_argument(
        "--max-id-hamming-distance",
        type=int,
        default=0,
        help="If valid IDs are provided, snap decoded IDs to the nearest valid ID only when the Hamming distance is at most this value.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Run detection only, skip pose initialization and bundle adjustment, and export detections.",
    )
    return parser.parse_args()


def iter_camera_images(data_root: Path, camera_name: str) -> List[Path]:
    image_paths: List[Path] = []
    for folder in sorted(data_root.glob("photos_*")):
        if not folder.is_dir():
            continue
        image_paths.extend(sorted(folder.glob(f"{camera_name}_*.jpg")))
    return image_paths


def merge_duplicate_detections(points: Sequence[np.ndarray], merge_radius: float) -> np.ndarray | None:
    if len(points) == 1:
        return points[0]
    stacked = np.vstack(points)
    center = np.mean(stacked, axis=0)
    if np.max(np.linalg.norm(stacked - center, axis=1)) <= merge_radius:
        return center
    return None


def save_annotated_detection_image(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save annotated detection image to {output_path}")


def save_target_detections(
    images: Sequence[ImageDetections],
    camera_name: str,
    output_dir: Path,
) -> Path:
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = camera_output_dir / "target_detections.txt"
    lines = ["image_name\ttarget_id\tx_image\ty_image"]
    for image in images:
        for target_id, point in sorted(image.detections.items()):
            lines.append(
                f"{image.image_path.name}\t{int(target_id)}\t{float(point[0]):.6f}\t{float(point[1]):.6f}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def filter_images_for_bundle(
    images: Sequence[ImageDetections],
    min_detections: int,
) -> List[ImageDetections]:
    return [image for image in images if len(image.detections) > min_detections]


def build_combined_detection_lines(
    images: Sequence[ImageDetections],
    camera_name: str,
) -> List[str]:
    lines: List[str] = []
    for image in images:
        for target_id, point in sorted(image.detections.items()):
            lines.append(
                f"{camera_name}\t{image.image_path.name}\t{int(target_id)}\t{float(point[0]):.6f}\t{float(point[1]):.6f}"
            )
    return lines


def save_combined_target_detections(
    detections_by_camera: Sequence[tuple[str, Sequence[ImageDetections]]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_detections.txt"
    lines = ["camera_name\timage_name\ttarget_id\tx_image\ty_image"]
    for camera_name, images in detections_by_camera:
        lines.extend(build_combined_detection_lines(images, camera_name))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _load_detections_cct_detect(
    image_paths: Sequence[Path],
    detections_output_dir: Path | None,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
) -> List[ImageDetections]:
    """Detection path using the modern cct_detect detector with affine rectification."""
    detector = _CCTDetector(n_bits=14)
    kept: List[ImageDetections] = []
    expected_size: tuple[int, int] | None = None
    for image_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if expected_size is None:
            expected_size = (width, height)
        elif expected_size != (width, height):
            raise ValueError(
                f"Image size mismatch for {image_path}: {(width, height)} != {expected_size}"
            )

        detections = detector.detect(image)

        grouped: Dict[int, List[np.ndarray]] = {}
        for det in detections:
            raw_id = det.target_id
            target_id, corrected = normalize_target_id(raw_id, valid_ids, max_id_hamming_distance)
            if target_id is None:
                continue
            grouped.setdefault(target_id, []).append(
                np.array([det.center[0], det.center[1]], dtype=np.float64)
            )

        merge_radius = max(width, height) * 0.01
        unique_detections: Dict[int, np.ndarray] = {}
        for target_id, points in grouped.items():
            merged = merge_duplicate_detections(points, merge_radius)
            if merged is not None:
                unique_detections[target_id] = merged

        if detections_output_dir is not None:
            annotated = detector.annotate(image, detections)
            save_annotated_detection_image(detections_output_dir / image_path.name, annotated)

        if unique_detections:
            kept.append(
                ImageDetections(
                    image_path=image_path,
                    image_index=image_index,
                    width=width,
                    height=height,
                    detections=unique_detections,
                )
            )

        if (image_index + 1) % 20 == 0 or image_index + 1 == len(image_paths):
            print(
                f"processed {image_index + 1}/{len(image_paths)} images, kept {len(kept)}",
                flush=True,
            )

    return kept


def load_image_detections(
    image_paths: Sequence[Path],
    detections_output_dir: Path | None = None,
    valid_ids: Set[int] | None = None,
    max_id_hamming_distance: int = 0,
) -> List[ImageDetections]:
    return _load_detections_cct_detect(
        image_paths,
        detections_output_dir,
        valid_ids,
        max_id_hamming_distance,
    )


def build_track_index(images: Sequence[ImageDetections]) -> Dict[int, Dict[int, np.ndarray]]:
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    for image in images:
        for target_id, point in image.detections.items():
            tracks.setdefault(target_id, {})[image.image_index] = point
    return tracks


def intrinsics_matrix(intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_vector(intrinsics: np.ndarray) -> np.ndarray:
    return np.array([intrinsics[4], intrinsics[5], intrinsics[6], intrinsics[7]], dtype=np.float64)


def rotation_matrix_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose[:3])
    return rotation


def project_point(intrinsics: np.ndarray, pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    point_camera = rotation @ point + pose[3:]
    z_coord = point_camera[2]
    if z_coord <= 1e-8:
        return np.array([1e6, 1e6], dtype=np.float64)

    x_coord = point_camera[0] / z_coord
    y_coord = point_camera[1] / z_coord
    radius_sq = x_coord * x_coord + y_coord * y_coord
    radial = 1.0 + intrinsics[4] * radius_sq + intrinsics[5] * radius_sq * radius_sq
    x_tangential = 2.0 * intrinsics[6] * x_coord * y_coord + intrinsics[7] * (radius_sq + 2.0 * x_coord * x_coord)
    y_tangential = intrinsics[6] * (radius_sq + 2.0 * y_coord * y_coord) + 2.0 * intrinsics[7] * x_coord * y_coord
    x_distorted = x_coord * radial + x_tangential
    y_distorted = y_coord * radial + y_tangential
    return np.array(
        [
            intrinsics[0] * x_distorted + intrinsics[2],
            intrinsics[1] * y_distorted + intrinsics[3],
        ],
        dtype=np.float64,
    )


def pose_to_projection(intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    translation = pose[3:].reshape(3, 1)
    return intrinsics_matrix(intrinsics) @ np.hstack([rotation, translation])


def triangulate_two_views(
    intrinsics: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray | None:
    projection_a = pose_to_projection(intrinsics, pose_a)
    projection_b = pose_to_projection(intrinsics, pose_b)
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        point_a.reshape(2, 1),
        point_b.reshape(2, 1),
    )
    if abs(homogeneous[3, 0]) < 1e-12:
        return None
    point = (homogeneous[:3, 0] / homogeneous[3, 0]).astype(np.float64)
    if not np.all(np.isfinite(point)):
        return None
    for pose in (pose_a, pose_b):
        if (rotation_matrix_from_pose(pose) @ point + pose[3:])[2] <= 1e-6:
            return None
    return point


def choose_seed_state(images: Sequence[ImageDetections], min_shared: int) -> CalibrationState:
    if not images:
        raise RuntimeError("No images passed the detection count filter.")

    width = images[0].width
    height = images[0].height
    intrinsics = np.array(
        [
            float(max(width, height)),
            float(max(width, height)),
            width / 2.0,
            height / 2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    camera_matrix = intrinsics_matrix(intrinsics)

    best_state: CalibrationState | None = None
    best_inlier_count = 0
    for first_index in range(len(images)):
        first_ids = set(images[first_index].detections)
        for second_index in range(first_index + 1, len(images)):
            shared_ids = sorted(first_ids & set(images[second_index].detections))
            if len(shared_ids) < min_shared:
                continue

            first_points = np.array([images[first_index].detections[target_id] for target_id in shared_ids], dtype=np.float64)
            second_points = np.array([images[second_index].detections[target_id] for target_id in shared_ids], dtype=np.float64)

            essential, essential_mask = cv2.findEssentialMat(
                first_points,
                second_points,
                camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
            if essential is None or essential_mask is None:
                continue

            recovered, rotation, translation, pose_mask = cv2.recoverPose(
                essential,
                first_points,
                second_points,
                camera_matrix,
            )
            if recovered < min_shared or pose_mask is None:
                continue

            inlier_ids = [shared_ids[index] for index in range(len(shared_ids)) if pose_mask[index, 0] != 0]
            if len(inlier_ids) < min_shared or len(inlier_ids) <= best_inlier_count:
                continue

            pose_a = np.zeros(6, dtype=np.float64)
            pose_b = np.zeros(6, dtype=np.float64)
            pose_b[:3, ...] = cv2.Rodrigues(rotation)[0].reshape(3)
            pose_b[3:] = translation.reshape(3)

            points: Dict[int, np.ndarray] = {}
            for target_id in inlier_ids:
                point = triangulate_two_views(
                    intrinsics,
                    pose_a,
                    pose_b,
                    images[first_index].detections[target_id],
                    images[second_index].detections[target_id],
                )
                if point is not None:
                    points[target_id] = point

            if len(points) < min_shared:
                continue

            best_inlier_count = len(points)
            best_state = CalibrationState(
                intrinsics=intrinsics.copy(),
                poses={
                    images[first_index].image_index: pose_a,
                    images[second_index].image_index: pose_b,
                },
                points=points,
                seed_image_indices=[images[first_index].image_index, images[second_index].image_index],
                inlier_track_ids=sorted(points),
            )

    if best_state is None:
        raise RuntimeError("Failed to initialize a seed image pair with enough shared detections.")
    return best_state


def register_remaining_images(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    min_shared: int,
) -> None:
    images_by_index = {image.image_index: image for image in images}
    while True:
        progress_made = False
        for image in images:
            if image.image_index in state.poses:
                continue

            shared_ids = sorted(set(image.detections) & set(state.points))
            if len(shared_ids) < min_shared:
                continue

            object_points = np.array([state.points[target_id] for target_id in shared_ids], dtype=np.float64)
            image_points = np.array([image.detections[target_id] for target_id in shared_ids], dtype=np.float64)
            solved, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics_matrix(state.intrinsics),
                distortion_vector(state.intrinsics),
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=200,
            )
            if not solved or inliers is None or len(inliers) < min_shared:
                continue

            inlier_indices = inliers.reshape(-1)
            inlier_object_points = object_points[inlier_indices]
            inlier_image_points = image_points[inlier_indices]
            refined, rotation_vec, translation_vec = cv2.solvePnP(
                inlier_object_points,
                inlier_image_points,
                intrinsics_matrix(state.intrinsics),
                distortion_vector(state.intrinsics),
                rvec=rotation_vec,
                tvec=translation_vec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not refined:
                continue

            pose = np.concatenate([rotation_vec.reshape(3), translation_vec.reshape(3)]).astype(np.float64)
            state.poses[image.image_index] = pose
            progress_made = True

            for target_id in image.detections:
                if target_id in state.points:
                    continue
                candidate_pairs: List[tuple[float, int]] = []
                for other_image_index in state.poses:
                    if other_image_index == image.image_index:
                        continue
                    other_image = images_by_index[other_image_index]
                    if target_id not in other_image.detections:
                        continue
                    baseline = float(
                        np.linalg.norm(state.poses[other_image_index][3:] - pose[3:])
                    )
                    candidate_pairs.append((baseline, other_image_index))

                for _, other_image_index in sorted(candidate_pairs, reverse=True):
                    other_image = images_by_index[other_image_index]
                    point = triangulate_two_views(
                        state.intrinsics,
                        state.poses[other_image_index],
                        pose,
                        other_image.detections[target_id],
                        image.detections[target_id],
                    )
                    if point is not None:
                        state.points[target_id] = point
                        break

        if not progress_made:
            break


def collect_observations(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    max_reprojection_error: float,
) -> List[tuple[int, int, np.ndarray]]:
    observations: List[tuple[int, int, np.ndarray]] = []
    for image in images:
        pose = state.poses.get(image.image_index)
        if pose is None:
            continue
        for target_id, point_2d in image.detections.items():
            point_3d = state.points.get(target_id)
            if point_3d is None:
                continue
            residual = project_point(state.intrinsics, pose, point_3d) - point_2d
            if not np.all(np.isfinite(residual)):
                continue
            if np.linalg.norm(residual) <= max_reprojection_error:
                observations.append((image.image_index, target_id, point_2d))
    return observations


def compute_reprojection_errors(
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for image_index, target_id, point_2d in observations:
        pose = state.poses.get(image_index)
        point_3d = state.points.get(target_id)
        if pose is None or point_3d is None:
            continue
        residual = project_point(state.intrinsics, pose, point_3d) - point_2d
        if np.all(np.isfinite(residual)):
            errors.append(float(np.linalg.norm(residual)))
    return np.asarray(errors, dtype=np.float64)


class LossConvergenceCallback(pyceres.IterationCallback):
    def __init__(self, relative_tolerance: float, patience: int):
        super().__init__()
        self.relative_tolerance = relative_tolerance
        self.patience = patience
        self.cost_history: List[float] = []
        self._stable_steps = 0

    def __call__(self, summary) -> pyceres.CallbackReturnType:
        cost = float(summary.cost)
        self.cost_history.append(cost)

        if len(self.cost_history) == 1:
            return pyceres.CallbackReturnType.SOLVER_CONTINUE

        if not getattr(summary, "step_is_successful", True):
            self._stable_steps = 0
            return pyceres.CallbackReturnType.SOLVER_CONTINUE

        previous_cost = self.cost_history[-2]
        relative_change = abs(previous_cost - cost) / max(abs(previous_cost), 1.0)
        if relative_change <= self.relative_tolerance:
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        if self._stable_steps >= self.patience:
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def camera_center_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    translation = pose[3:]
    return (-rotation.T @ translation).astype(np.float64)


def camera_axes_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    return rotation.T.astype(np.float64)


def build_alignment_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if len(points) < 3:
        return centroid, np.eye(3, dtype=np.float64)

    _, _, basis_t = np.linalg.svd(centered, full_matrices=False)
    basis = basis_t.T
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return centroid, basis


def align_scene(state: CalibrationState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered_pose_indices = np.array(sorted(state.poses), dtype=np.int32)
    camera_centers = np.array([camera_center_from_pose(state.poses[index]) for index in ordered_pose_indices], dtype=np.float64)
    camera_axes = np.array([camera_axes_from_pose(state.poses[index]) for index in ordered_pose_indices], dtype=np.float64)

    track_ids = np.array(sorted(state.points), dtype=np.int32)
    target_points = np.array([state.points[target_id] for target_id in track_ids], dtype=np.float64) if len(track_ids) else np.empty((0, 3), dtype=np.float64)

    combined = np.vstack([camera_centers, target_points]) if len(target_points) else camera_centers
    centroid, basis = build_alignment_basis(combined)

    aligned_centers = (camera_centers - centroid) @ basis
    aligned_points = (target_points - centroid) @ basis if len(target_points) else target_points
    aligned_axes = np.empty_like(camera_axes)
    for camera_index in range(len(camera_axes)):
        aligned_axes[camera_index] = basis.T @ camera_axes[camera_index]

    return ordered_pose_indices, aligned_centers, aligned_axes, track_ids, aligned_points


def save_scene_data(
    state: CalibrationState,
    camera_name: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera_name}_scene.npz"
    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)
    np.savez_compressed(
        output_path,
        pose_indices=pose_indices,
        camera_centers=camera_centers,
        camera_axes=camera_axes,
        track_ids=track_ids,
        target_points=target_points,
    )
    return output_path


def save_text_exports(
    state: CalibrationState,
    camera_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    targets_3d_path = camera_output_dir / "targets_3d.txt"
    cameras_path = camera_output_dir / "cameras.txt"

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)

    target_lines = [
        f"{int(track_id)} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}"
        for track_id, point in zip(track_ids, target_points, strict=False)
    ]
    targets_3d_path.write_text("\n".join(target_lines) + ("\n" if target_lines else ""), encoding="utf-8")

    camera_lines = []
    for pose_index, center, axes in zip(pose_indices, camera_centers, camera_axes, strict=False):
        quaternion = rotation_matrix_to_quaternion(axes)
        camera_lines.append(
            f"{int(pose_index)} {center[0]:.9f} {center[1]:.9f} {center[2]:.9f} {quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}"
        )
    cameras_path.write_text("\n".join(camera_lines) + ("\n" if camera_lines else ""), encoding="utf-8")

    return targets_3d_path, cameras_path


def save_colmap_output(
    state: CalibrationState,
    bundle_images: Sequence[ImageDetections],
    camera_name: str,
    output_dir: Path,
) -> Path:
    """Save calibration results in COLMAP text format.

    Produces cameras.txt, images.txt, and points3D.txt inside
    ``output_dir / camera_name / colmap``.
    """
    colmap_dir = output_dir / camera_name / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    images_by_index = {img.image_index: img for img in bundle_images}
    image_width = bundle_images[0].width if bundle_images else 0
    image_height = bundle_images[0].height if bundle_images else 0

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)

    # --- cameras.txt (single shared camera, OPENCV model) ---
    intr = state.intrinsics  # [fx, fy, cx, cy, k1, k2, p1, p2]
    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy,k1,k2,p1,p2]",
        f"1 OPENCV {image_width} {image_height} "
        f"{intr[0]:.9f} {intr[1]:.9f} {intr[2]:.9f} {intr[3]:.9f} "
        f"{intr[4]:.9f} {intr[5]:.9f} {intr[6]:.9f} {intr[7]:.9f}",
    ]
    (colmap_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")

    # Build a mapping: track_id → COLMAP point3D_id (1-based)
    track_id_to_colmap_id = {int(tid): idx + 1 for idx, tid in enumerate(track_ids)}

    # --- images.txt ---
    # COLMAP stores world-to-camera transforms (R, t) such that X_cam = R * X_world + t
    # Our poses are already in that convention (rvec, tvec).
    # But align_scene applies a rotation/translation, so we need to recompute
    # the aligned poses from aligned_centers and aligned_axes.
    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for colmap_image_id, (pose_idx, center, axes) in enumerate(
        zip(pose_indices, camera_centers, camera_axes, strict=False), start=1
    ):
        # axes is the camera rotation matrix (world-to-camera is axes.T)
        R_w2c = axes.T  # axes = R.T, so axes.T = R (world-to-camera rotation)
        t_w2c = -R_w2c @ center  # t = -R * C
        quat = rotation_matrix_to_quaternion(R_w2c)  # [x, y, z, w]
        qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])

        image_name = images_by_index[int(pose_idx)].image_path.name if int(pose_idx) in images_by_index else f"image_{int(pose_idx):04d}.jpg"

        images_lines.append(
            f"{colmap_image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {image_name}"
        )

        # 2D points line: list detected targets that correspond to 3D points
        points2d_parts = []
        img_det = images_by_index.get(int(pose_idx))
        if img_det is not None:
            for target_id, pt2d in sorted(img_det.detections.items()):
                colmap_pt_id = track_id_to_colmap_id.get(target_id, -1)
                points2d_parts.append(f"{pt2d[0]:.4f} {pt2d[1]:.4f} {colmap_pt_id}")
        images_lines.append(" ".join(points2d_parts) if points2d_parts else "")

    (colmap_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")

    # --- points3D.txt ---
    # Build track info (which images observe each point)
    points3d_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    # Build per-point observation track
    point_tracks: Dict[int, List[tuple[int, int]]] = {int(tid): [] for tid in track_ids}
    colmap_image_id_map = {int(pi): ci for ci, pi in enumerate(pose_indices, start=1)}
    for pose_idx_int, img_det in images_by_index.items():
        if pose_idx_int not in state.poses:
            continue
        colmap_img_id = colmap_image_id_map.get(pose_idx_int)
        if colmap_img_id is None:
            continue
        sorted_dets = sorted(img_det.detections.items())
        for pt2d_idx, (target_id, _) in enumerate(sorted_dets):
            if target_id in point_tracks:
                point_tracks[target_id].append((colmap_img_id, pt2d_idx))

    for idx, (tid, pt3d) in enumerate(zip(track_ids, target_points, strict=False)):
        colmap_pt_id = idx + 1
        track_entries = point_tracks.get(int(tid), [])
        track_str = " ".join(f"{img_id} {pt_idx}" for img_id, pt_idx in track_entries)
        points3d_lines.append(
            f"{colmap_pt_id} {pt3d[0]:.9f} {pt3d[1]:.9f} {pt3d[2]:.9f} 200 200 200 0.0 {track_str}"
        )

    (colmap_dir / "points3D.txt").write_text("\n".join(points3d_lines) + "\n", encoding="utf-8")

    return colmap_dir


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * scale
        x_value = (rotation[2, 1] - rotation[1, 2]) / scale
        y_value = (rotation[0, 2] - rotation[2, 0]) / scale
        z_value = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w_value = (rotation[2, 1] - rotation[1, 2]) / scale
        x_value = 0.25 * scale
        y_value = (rotation[0, 1] + rotation[1, 0]) / scale
        z_value = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w_value = (rotation[0, 2] - rotation[2, 0]) / scale
        x_value = (rotation[0, 1] + rotation[1, 0]) / scale
        y_value = 0.25 * scale
        z_value = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w_value = (rotation[1, 0] - rotation[0, 1]) / scale
        x_value = (rotation[0, 2] + rotation[2, 0]) / scale
        y_value = (rotation[1, 2] + rotation[2, 1]) / scale
        z_value = 0.25 * scale
    quaternion = np.array([x_value, y_value, z_value, w_value], dtype=np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return quaternion


def write_rerun_visualization(
    state: CalibrationState,
    camera_name: str,
    image_size: tuple[int, int],
    output_dir: Path,
    spawn_viewer: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = output_dir / f"{camera_name}.rrd"

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)
    recording = rr.init(f"calibration_{camera_name}", spawn=False)
    if spawn_viewer:
        rr.spawn(recording=recording)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)

    if len(target_points):
        rr.log(
            f"world/{camera_name}/targets",
            rr.Points3D(
                target_points,
                radii=0.02,
                colors=[180, 180, 180],
                labels=[str(int(track_id)) for track_id in track_ids],
            ),
            recording=recording,
        )

    image_width, image_height = image_size
    for pose_index, center, axes in zip(pose_indices, camera_centers, camera_axes, strict=False):
        camera_path = f"world/{camera_name}/cameras/{int(pose_index):04d}"
        rr.log(
            camera_path,
            rr.Transform3D(
                translation=center,
                quaternion=rr.Quaternion(xyzw=rotation_matrix_to_quaternion(axes)),
            ),
            rr.TransformAxes3D(axis_length=0.15),
            recording=recording,
        )
        rr.log(
            camera_path,
            rr.Pinhole(
                image_from_camera=intrinsics_matrix(state.intrinsics),
                resolution=[image_width, image_height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
                line_width=0.004,
                color=[255, 128, 0],
            ),
            recording=recording,
        )

    rr.save(recording_path, recording=recording)
    return recording_path


def save_convergence_plot(
    camera_name: str,
    cost_history: Sequence[float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera_name}_convergence.png"

    fig, axis = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(cost_history), dtype=np.int32)
    axis.plot(iterations, cost_history, color="#1f77b4", linewidth=2)
    axis.set_title(f"Loss convergence for {camera_name}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cost")
    axis.grid(True, linestyle="--", alpha=0.4)
    if len(cost_history) > 1 and min(cost_history) > 0:
        axis.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


class ReprojectionCost(pyceres.CostFunction):
    def __init__(self, observation: np.ndarray):
        super().__init__()
        self.observation = observation.astype(np.float64)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([8, 6, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        parameter_blocks = [np.array(parameters[index], dtype=np.float64, copy=True) for index in range(3)]
        prediction = project_point(parameter_blocks[0], parameter_blocks[1], parameter_blocks[2])
        residual_vector = prediction - self.observation
        residuals[0] = float(residual_vector[0])
        residuals[1] = float(residual_vector[1])

        if jacobians is not None:
            for block_index, block in enumerate(parameter_blocks):
                if jacobians[block_index] is None:
                    continue
                jacobian = np.zeros((2, block.shape[0]), dtype=np.float64)
                for column in range(block.shape[0]):
                    step = 1e-6 * max(1.0, abs(block[column]))
                    plus_blocks = [value.copy() for value in parameter_blocks]
                    minus_blocks = [value.copy() for value in parameter_blocks]
                    plus_blocks[block_index][column] += step
                    minus_blocks[block_index][column] -= step
                    residual_plus = project_point(plus_blocks[0], plus_blocks[1], plus_blocks[2]) - self.observation
                    residual_minus = project_point(minus_blocks[0], minus_blocks[1], minus_blocks[2]) - self.observation
                    jacobian[:, column] = (residual_plus - residual_minus) / (2.0 * step)

                for row in range(2):
                    for column in range(block.shape[0]):
                        jacobians[block_index][row * block.shape[0] + column] = jacobian[row, column]
        return True


def _filter_inlier_observations(
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
    max_error: float,
) -> List[tuple[int, int, np.ndarray]]:
    """Keep only observations whose reprojection error is within *max_error* pixels."""
    inliers: List[tuple[int, int, np.ndarray]] = []
    for image_index, target_id, point_2d in observations:
        pose = state.poses.get(image_index)
        point_3d = state.points.get(target_id)
        if pose is None or point_3d is None:
            continue
        residual = project_point(state.intrinsics, pose, point_3d) - point_2d
        if np.all(np.isfinite(residual)) and np.linalg.norm(residual) <= max_error:
            inliers.append((image_index, target_id, point_2d))
    return inliers


def solve_bundle_adjustment(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
    max_iterations: int,
    huber_delta: float,
    loss_relative_tolerance: float,
    loss_patience: int,
    robust_loss: str = "huber",
    cauchy_scale: float = 3.0,
) -> SolverDiagnostics:
    problem = pyceres.Problem()
    if robust_loss == "cauchy":
        loss = pyceres.CauchyLoss(cauchy_scale)
    else:
        loss = pyceres.HuberLoss(huber_delta)
    manifolds: List[pyceres.Manifold] = []

    for image_index, target_id, point_2d in observations:
        cost = ReprojectionCost(point_2d)
        problem.add_residual_block(
            cost,
            loss,
            [state.intrinsics, state.poses[image_index], state.points[target_id]],
        )

    if state.seed_image_indices:
        problem.set_parameter_block_constant(state.poses[state.seed_image_indices[0]])
    if len(state.seed_image_indices) > 1:
        second_seed_pose = state.poses[state.seed_image_indices[1]]
        translation_fixed_manifold = pyceres.SubsetManifold(6, [3, 4, 5])
        manifolds.append(translation_fixed_manifold)
        problem.set_manifold(second_seed_pose, translation_fixed_manifold)

    width = images[0].width
    height = images[0].height
    max_dimension = float(max(width, height))
    problem.set_parameter_lower_bound(state.intrinsics, 0, 0.2 * max_dimension)
    problem.set_parameter_lower_bound(state.intrinsics, 1, 0.2 * max_dimension)
    problem.set_parameter_upper_bound(state.intrinsics, 0, 3.0 * max_dimension)
    problem.set_parameter_upper_bound(state.intrinsics, 1, 3.0 * max_dimension)
    problem.set_parameter_lower_bound(state.intrinsics, 2, 0.25 * width)
    problem.set_parameter_upper_bound(state.intrinsics, 2, 0.75 * width)
    problem.set_parameter_lower_bound(state.intrinsics, 3, 0.25 * height)
    problem.set_parameter_upper_bound(state.intrinsics, 3, 0.75 * height)
    for distortion_index in range(4, 8):
        problem.set_parameter_lower_bound(state.intrinsics, distortion_index, -1.0)
        problem.set_parameter_upper_bound(state.intrinsics, distortion_index, 1.0)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = True
    options.num_threads = 1
    options.update_state_every_iteration = True
    options.function_tolerance = loss_relative_tolerance

    callback = LossConvergenceCallback(loss_relative_tolerance, loss_patience)
    options.callbacks = [callback]

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    cost_history = callback.cost_history or [float(summary.initial_cost), float(summary.final_cost)]
    return SolverDiagnostics(
        summary=summary.BriefReport(),
        iterations=int(summary.num_successful_steps + summary.num_unsuccessful_steps + 1),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        cost_history=[float(cost) for cost in cost_history],
    )


def calibrate_camera(
    camera_name: str,
    data_root: Path,
    min_detections: int,
    min_shared: int,
    max_reprojection_error: float,
    max_iterations: int,
    huber_delta: float,
    loss_relative_tolerance: float,
    loss_patience: int,
    output_dir: Path,
    show_3d: bool,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
) -> CameraCalibrationResult:
    image_paths = iter_camera_images(data_root, camera_name)
    if not image_paths:
        raise RuntimeError(f"No images found for {camera_name} under {data_root}")

    print(f"collecting detections for {camera_name} from {len(image_paths)} images", flush=True)
    detections_dir_path = output_dir / camera_name / "annotated_detections"
    images = load_image_detections(
        image_paths,
        detections_output_dir=detections_dir_path,
        valid_ids=valid_ids,
        max_id_hamming_distance=max_id_hamming_distance,
    )
    target_detections_path = save_target_detections(images, camera_name, output_dir)
    bundle_images = filter_images_for_bundle(images, min_detections)
    if len(bundle_images) < 2:
        raise RuntimeError(f"Not enough images with more than {min_detections} detections for {camera_name}")

    state = choose_seed_state(bundle_images, min_shared)
    register_remaining_images(bundle_images, state, min_shared)

    observations = collect_observations(bundle_images, state, max_reprojection_error)
    if len(observations) < min_shared * 2:
        raise RuntimeError(f"Not enough geometrically consistent observations for {camera_name}")

    registered_before = len(state.poses)
    diagnostics = solve_bundle_adjustment(
        bundle_images,
        state,
        observations,
        max_iterations,
        huber_delta,
        loss_relative_tolerance,
        loss_patience,
    )

    # Second pass: try to register images that failed on noisy initial geometry.
    # After BA the 3D points and intrinsics are refined, so PnP may now succeed.
    register_remaining_images(bundle_images, state, min_shared)
    observations = collect_observations(bundle_images, state, max_reprojection_error)
    registered_after = len(state.poses)
    if registered_after > registered_before:
        print(
            f"second pass registered {registered_after - registered_before} additional images "
            f"({registered_after} total)",
            flush=True,
        )
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
        )
    # --- Robust refinement pass ---
    # Re-collect all possible observations (generous threshold) and run BA with
    # Cauchy loss to downweight outlier detections / triangulations.
    all_observations = collect_observations(bundle_images, state, max_reprojection_error * 3.0)
    if len(all_observations) >= len(observations):
        print(
            f"robust refinement: {len(all_observations)} observations "
            f"(was {len(observations)})",
            flush=True,
        )
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            all_observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
            robust_loss="cauchy",
            cauchy_scale=3.0,
        )
        # Discard observations that remain outliers after robust optimisation
        observations = _filter_inlier_observations(state, all_observations, max_reprojection_error)
        print(
            f"robust refinement kept {len(observations)}/{len(all_observations)} inlier observations",
            flush=True,
        )
        # Final clean BA on inliers only (standard Huber)
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
        )

    reprojection_errors = compute_reprojection_errors(state, observations)
    if reprojection_errors.size == 0:
        raise RuntimeError(f"No reprojection errors available after optimization for {camera_name}")

    convergence_plot_path = save_convergence_plot(camera_name, diagnostics.cost_history, output_dir)
    scene_data_path = save_scene_data(state, camera_name, output_dir)
    rerun_recording_path = write_rerun_visualization(state, camera_name, (bundle_images[0].width, bundle_images[0].height), output_dir, show_3d)
    targets_3d_path, cameras_path = save_text_exports(state, camera_name, output_dir)
    colmap_dir_path = save_colmap_output(state, bundle_images, camera_name, output_dir)
    track_ids = {target_id for _, target_id, _ in observations}
    return CameraCalibrationResult(
        camera_name=camera_name,
        image_size=(bundle_images[0].width, bundle_images[0].height),
        used_images=len(bundle_images),
        registered_images=len(state.poses),
        tracks=len(track_ids),
        observations=len(observations),
        summary=diagnostics.summary,
        intrinsics=state.intrinsics.copy(),
        mean_reprojection_error=float(np.mean(reprojection_errors)),
        rms_reprojection_error=float(np.sqrt(np.mean(np.square(reprojection_errors)))),
        iterations=diagnostics.iterations,
        initial_cost=diagnostics.initial_cost,
        final_cost=diagnostics.final_cost,
        convergence_plot_path=convergence_plot_path,
        rerun_recording_path=rerun_recording_path,
        scene_data_path=scene_data_path,
        targets_3d_path=targets_3d_path,
        cameras_path=cameras_path,
        colmap_dir_path=colmap_dir_path,
        detections_dir_path=detections_dir_path,
        target_detections_path=target_detections_path,
    )


def detect_camera(
    camera_name: str,
    data_root: Path,
    min_detections: int,
    output_dir: Path,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
) -> tuple[DetectionOnlyResult, List[ImageDetections]]:
    image_paths = iter_camera_images(data_root, camera_name)
    if not image_paths:
        raise RuntimeError(f"No images found for {camera_name} under {data_root}")

    print(f"collecting detections for {camera_name} from {len(image_paths)} images", flush=True)
    detections_dir_path = output_dir / camera_name / "annotated_detections"
    images = load_image_detections(
        image_paths,
        detections_output_dir=detections_dir_path,
        valid_ids=valid_ids,
        max_id_hamming_distance=max_id_hamming_distance,
    )
    target_detections_path = save_target_detections(images, camera_name, output_dir)
    return DetectionOnlyResult(
        camera_name=camera_name,
        total_images=len(image_paths),
        used_images=len(images),
        detections_dir_path=detections_dir_path,
        target_detections_path=target_detections_path,
    ), images


def format_result(result: CameraCalibrationResult) -> dict:
    intrinsics = result.intrinsics
    return {
        "camera": result.camera_name,
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "used_images": result.used_images,
        "registered_images": result.registered_images,
        "tracks": result.tracks,
        "observations": result.observations,
        "iterations": result.iterations,
        "summary": result.summary,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "avg_reprojection_error_px": result.mean_reprojection_error,
        "rms_reprojection_error_px": result.rms_reprojection_error,
        "convergence_plot": str(result.convergence_plot_path),
        "rerun_recording": str(result.rerun_recording_path),
        "scene_data": str(result.scene_data_path),
        "targets_3d": str(result.targets_3d_path),
        "cameras_tum": str(result.cameras_path),
        "colmap_dir": str(result.colmap_dir_path),
        "annotated_detections": str(result.detections_dir_path),
        "target_detections": str(result.target_detections_path),
        "camera_parameters": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3]),
            "k1": float(intrinsics[4]),
            "k2": float(intrinsics[5]),
            "p1": float(intrinsics[6]),
            "p2": float(intrinsics[7]),
        },
    }


def format_detection_result(result: DetectionOnlyResult) -> dict:
    return {
        "camera": result.camera_name,
        "total_images": result.total_images,
        "used_images": result.used_images,
        "annotated_detections": str(result.detections_dir_path),
        "target_detections": str(result.target_detections_path),
    }


def main() -> int:
    args = parse_args()
    valid_ids = resolve_valid_ids(args.valid_ids_file, args.valid_id_max)
    cameras = args.cameras or ["camera_0", "camera_1"]

    if args.detect_only:
        detection_results: List[dict] = []
        detections_by_camera: List[tuple[str, List[ImageDetections]]] = []
        for camera_name in cameras:
            result, images = detect_camera(
                camera_name=camera_name,
                data_root=args.data_root,
                min_detections=args.min_detections,
                output_dir=args.output_dir,
                valid_ids=valid_ids,
                max_id_hamming_distance=args.max_id_hamming_distance,
            )
            result_dict = format_detection_result(result)
            detection_results.append(result_dict)
            detections_by_camera.append((camera_name, images))
            print(json.dumps(result_dict, indent=2), flush=True)

        combined_detections_path = save_combined_target_detections(detections_by_camera, args.output_dir)
        combined_result = {
            "detect_only": True,
            "cameras": detection_results,
            "combined_target_detections": str(combined_detections_path),
        }
        summary_path = args.output_dir / "detection_summary.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(combined_result, indent=2), encoding="utf-8")
        print(json.dumps(combined_result, indent=2), flush=True)
        return 0

    results: List[dict] = []
    for camera_name in cameras:
        result = calibrate_camera(
            camera_name=camera_name,
            data_root=args.data_root,
            min_detections=args.min_detections,
            min_shared=args.min_shared,
            max_reprojection_error=args.max_reprojection_error,
            max_iterations=args.max_iterations,
            huber_delta=args.huber_delta,
            loss_relative_tolerance=args.loss_relative_tolerance,
            loss_patience=args.loss_patience,
            output_dir=args.output_dir,
            show_3d=args.show_3d,
            valid_ids=valid_ids,
            max_id_hamming_distance=args.max_id_hamming_distance,
        )
        result_dict = format_result(result)
        results.append(result_dict)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{camera_name}_summary.json"
        summary_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        print(json.dumps(result_dict, indent=2), flush=True)

    if len(results) > 1:
        print(json.dumps({"results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())