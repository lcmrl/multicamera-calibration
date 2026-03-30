"""Combined multi-camera CCT calibration pipeline.

Pipeline:
  1. Run SfM via multicamera-calibration to obtain initial camera poses.
  2. Run CCT target detection on all images.
  3. Initialize a multi-camera CalibrationState from SfM poses + CCT detections.
  4. Joint bundle adjustment of all cameras with per-camera intrinsics and
     a fixed relative-pose constraint within each rig frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyceres
import pycolmap
import yaml

from cct_calibration.run import (
    CalibrationState,
    ImageDetections,
    LossConvergenceCallback,
    ReprojectionCost,
    SolverDiagnostics,
    _load_detections_cct_detect,
    collect_observations,
    compute_reprojection_errors,
    distortion_vector,
    filter_images_for_bundle,
    intrinsics_matrix,
    normalize_target_id,
    project_point,
    rotation_matrix_from_pose,
    rotation_matrix_to_quaternion,
    save_convergence_plot,
    save_target_detections,
    triangulate_two_views,
    _filter_inlier_observations,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiCameraState:
    """State for joint multi-camera calibration.

    Each camera has its own intrinsics vector (8-element OPENCV model).
    ``rig_poses`` stores the 6-DOF pose (Rodrigues + translation) of camera-0
    for each rig frame.  The pose of camera *k* in rig frame *f* is computed as:

        pose_cam_k = compose_poses(rig_poses[f], relative_poses[k])

    ``relative_poses[0]`` is always the identity (cam-0 is the reference).
    """
    camera_intrinsics: Dict[str, np.ndarray]          # cam_name -> [fx,fy,cx,cy,k1,k2,p1,p2]
    rig_poses: Dict[str, np.ndarray]                   # frame_id (timestamp) -> 6-vector (cam-0 pose)
    relative_poses: Dict[str, np.ndarray]              # cam_name -> 6-vector relative to cam-0
    points: Dict[int, np.ndarray]                      # target_id -> 3D point
    camera_names: List[str]                            # ordered camera names


@dataclass
class MultiCameraResult:
    camera_results: Dict[str, dict]   # per-camera intrinsics + errors
    total_observations: int
    total_tracks: int
    total_registered_frames: int
    mean_reprojection_error: float
    rms_reprojection_error: float
    summary: str
    iterations: int
    initial_cost: float
    final_cost: float


# ---------------------------------------------------------------------------
# Pose composition utilities
# ---------------------------------------------------------------------------

def compose_poses(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Compose two world-to-camera poses:  result = B ∘ A.

    If A maps world→cam0 and B maps cam0→camK, the result maps world→camK.
    """
    R_a = rotation_matrix_from_pose(pose_a)
    t_a = pose_a[3:]
    R_b = rotation_matrix_from_pose(pose_b)
    t_b = pose_b[3:]
    R = R_b @ R_a
    t = R_b @ t_a + t_b
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t]).astype(np.float64)


def identity_pose() -> np.ndarray:
    return np.zeros(6, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 1 — SfM via pycolmap
# ---------------------------------------------------------------------------

def run_sfm(
    image_root: Path,
    camera_names: List[str],
    sfm_output_dir: Path,
    initial_focal: float | None = None,
    known_baseline: float | None = None,
    force: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Run pycolmap SfM and return per-camera intrinsics and per-image poses.

    Returns
    -------
    intrinsics_by_camera : dict  cam_name -> 8-vector [fx,fy,cx,cy,k1,k2,p1,p2]
    poses_by_camera : dict  cam_name -> { image_stem: 6-vector pose }
    """
    db_path = sfm_output_dir / "sfm.db"
    colmap_image_root = sfm_output_dir / "images"
    recon_output = sfm_output_dir / "reconstruction"

    # Determine image dimensions from first image
    first_cam_dir = image_root / camera_names[0]
    first_img_path = sorted(first_cam_dir.glob("*.jpg"))[0]
    first_img = cv2.imread(str(first_img_path))
    img_h, img_w = first_img.shape[:2]

    # Camera ID mapping (deterministic)
    camera_id_map: Dict[str, int] = {cn: idx + 1 for idx, cn in enumerate(camera_names)}

    # If no initial_focal provided, use a common heuristic: max(w,h) * 0.5
    if initial_focal is None:
        initial_focal = max(img_w, img_h) * 0.5

    # Check for cached reconstruction
    images_txt = recon_output / "images.txt"
    cameras_txt = recon_output / "cameras.txt"
    use_cache = (not force) and images_txt.exists() and cameras_txt.exists()

    if use_cache:
        print("  reloading existing SfM reconstruction...", flush=True)
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_text(str(recon_output))
    else:
        # Copy images into COLMAP directory structure
        rig_dir = colmap_image_root / "rig1"
        for cam_name in camera_names:
            dst = rig_dir / cam_name
            dst.mkdir(parents=True, exist_ok=True)
            src_dir = image_root / cam_name
            for img in sorted(src_dir.glob("*.jpg")):
                target = dst / img.name
                if not target.exists():
                    shutil.copy2(str(img), str(target))

        # Clean stale DB
        if db_path.exists():
            try:
                os.remove(str(db_path))
            except PermissionError:
                import gc; gc.collect()
                os.remove(str(db_path))

        db = pycolmap.Database.open(str(db_path))

        # Create cameras
        for cam_name in camera_names:
            cam_config = {
                "model": "OPENCV",
                "width": img_w,
                "height": img_h,
                "params": [initial_focal, initial_focal, img_w / 2.0, img_h / 2.0,
                            0.0, 0.0, 0.0, 0.0],
            }
            camera = pycolmap.Camera(cam_config)
            db.write_camera(camera)

        # Create rig (cam0 is reference)
        colmap_rig = pycolmap.Rig({"rig_id": 1})
        ref_sensor = pycolmap.sensor_t({
            "type": pycolmap.SensorType.CAMERA,
            "id": camera_id_map[camera_names[0]],
        })
        colmap_rig.add_ref_sensor(ref_sensor)

        for cam_name in camera_names[1:]:
            sensor = pycolmap.sensor_t({
                "type": pycolmap.SensorType.CAMERA,
                "id": camera_id_map[cam_name],
            })
            baseline_guess = known_baseline if known_baseline is not None else 0.3
            transform = pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d([0.0, 0.0, 0.0, 1.0]),
                translation=[-baseline_guess, 0.0, 0.0],
            )
            colmap_rig.add_sensor(sensor, transform)
        db.write_rig(colmap_rig)

        # Create images and frames
        frames_dir = image_root / camera_names[0]
        frame_stems = sorted([f.stem for f in frames_dir.glob("*.jpg")])
        image_id = 1
        frame_id = 1
        for stem in frame_stems:
            colmap_frame = pycolmap.Frame({"frame_id": frame_id, "rig_id": 1})
            frame_id += 1
            for cam_name in camera_names:
                img_name = f"rig1/{cam_name}/{stem}.jpg"
                image = pycolmap.Image(
                    name=img_name,
                    points2D=np.empty((0, 2), dtype=np.float64),
                    camera_id=camera_id_map[cam_name],
                    image_id=image_id,
                )
                db.write_image(image, use_image_id=False)
                colmap_frame.add_data_id(image.data_id)
                image_id += 1
            db.write_frame(colmap_frame)

        try:
            db.close()
        except AttributeError:
            pass
        del db

        # Clean old reconstruction
        if recon_output.exists():
            shutil.rmtree(str(recon_output))

        # Use fast extraction settings: fewer features, skip upsampled octave
        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.sift.max_num_features = 4096
        extraction_opts.sift.first_octave = 0

        pycolmap.extract_features(
            db_path, colmap_image_root,
            extraction_options=extraction_opts,
        )

        # Sequential matching is much faster than exhaustive for ordered rig data
        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.overlap = 10
        seq_opts.quadratic_overlap = False
        seq_opts.loop_detection = False
        pycolmap.match_sequential(db_path, pairing_options=seq_opts)

        maps = pycolmap.incremental_mapping(db_path, colmap_image_root, recon_output)

        if not maps:
            raise RuntimeError("SfM reconstruction failed — no maps produced.")

        reconstruction = maps[0]
        reconstruction.write_text(recon_output)

    # Extract camera intrinsics (use SfM values directly as initialisation)
    intrinsics_by_camera: Dict[str, np.ndarray] = {}
    for cam_name in camera_names:
        cam_id = camera_id_map[cam_name]
        colmap_cam = reconstruction.cameras[cam_id]
        params = np.array(colmap_cam.params, dtype=np.float64)
        intrinsics_by_camera[cam_name] = params

    # Extract per-image poses
    poses_by_camera: Dict[str, Dict[str, np.ndarray]] = {cn: {} for cn in camera_names}
    for img_id, colmap_img in reconstruction.images.items():
        name = colmap_img.name  # e.g. "rig1/cam0/1769095475851869445.jpg"
        parts = Path(name).parts  # ('rig1', 'cam0', 'xxx.jpg')
        if len(parts) < 3:
            continue
        cam_name = parts[1]
        stem = Path(parts[2]).stem

        # COLMAP stores world-to-camera as quaternion [qw, qx, qy, qz] + translation
        rigid = colmap_img.cam_from_world()
        R = rigid.rotation.matrix()
        t = np.array(rigid.translation, dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)
        pose = np.concatenate([rvec.reshape(3), t]).astype(np.float64)
        poses_by_camera[cam_name][stem] = pose

    # Scale
    if known_baseline is not None:
        # Read rig output to determine scale
        rigs_path = recon_output / "rigs.txt"
        if rigs_path.exists():
            for line in rigs_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) >= 11:
                    rig_translation = np.array([float(tokens[8]), float(tokens[9]), float(tokens[10])], dtype=np.float64)
                    sfm_baseline = float(np.linalg.norm(rig_translation))
                    if sfm_baseline > 1e-6:
                        scale = known_baseline / sfm_baseline
                        # Scale all translations
                        for cam_name in camera_names:
                            for stem in poses_by_camera[cam_name]:
                                poses_by_camera[cam_name][stem][3:] *= scale
                    break

    print(f"SfM: reconstructed {sum(len(v) for v in poses_by_camera.values())} images "
          f"across {len(camera_names)} cameras", flush=True)
    return intrinsics_by_camera, poses_by_camera


# ---------------------------------------------------------------------------
# Step 2 — CCT detection
# ---------------------------------------------------------------------------

def _load_cached_detections(
    cache_path: Path,
    image_root_cam: Path,
) -> List[ImageDetections] | None:
    """Try to load cached detections from a target_detections.txt file.

    Returns None if the cache file doesn't exist or is unreadable.
    """
    if not cache_path.exists():
        return None

    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    if not lines or not lines[0].startswith("image_name"):
        return None

    # Parse into per-image groups
    from collections import OrderedDict
    grouped: Dict[str, Dict[int, np.ndarray]] = OrderedDict()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        img_name, tid_str, x_str, y_str = parts[0], parts[1], parts[2], parts[3]
        grouped.setdefault(img_name, {})[int(tid_str)] = np.array(
            [float(x_str), float(y_str)], dtype=np.float64
        )

    # Build ImageDetections, reading dimensions from the first image
    result: List[ImageDetections] = []
    width, height = 0, 0
    for idx, (img_name, dets) in enumerate(grouped.items()):
        img_path = image_root_cam / img_name
        if width == 0 and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]
        result.append(ImageDetections(
            image_path=img_path,
            image_index=idx,
            width=width,
            height=height,
            detections=dets,
        ))

    print(f"  loaded {len(result)} images from cache: {cache_path}", flush=True)
    return result


def load_targets3d(path: Path) -> Dict[int, np.ndarray]:
    """Load known 3D target positions from a TSV file.

    Expected format (header optional)::

        target_id  x  y  z
    """
    points: Dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("target_id"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        tid = int(parts[0])
        points[tid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    print(f"loaded {len(points)} known 3D target positions from {path}", flush=True)
    return points


def detect_all_cameras(
    image_root: Path,
    camera_names: List[str],
    output_dir: Path,
    valid_ids: Set[int] | None = None,
    max_id_hamming_distance: int = 0,
    min_detections: int = 5,
    force: bool = False,
) -> Dict[str, List[ImageDetections]]:
    """Run CCT detection on all cameras.

    Returns dict: cam_name -> list of ImageDetections.
    Image indices are globally unique across cameras.
    """
    all_detections: Dict[str, List[ImageDetections]] = {}
    global_index = 0

    for cam_name in camera_names:
        cam_dir = image_root / cam_name
        image_paths = sorted(cam_dir.glob("*.jpg"))
        if not image_paths:
            print(f"  warning: no images found for {cam_name}", flush=True)
            continue

        # Try loading from cache first
        cache_path = output_dir / cam_name / "target_detections.txt"
        cached = None if force else _load_cached_detections(cache_path, cam_dir)

        if cached is not None:
            raw = cached
        else:
            print(f"detecting CCT targets in {cam_name} ({len(image_paths)} images)...", flush=True)
            detections_dir = output_dir / cam_name / "annotated_detections"
            raw = _load_detections_cct_detect(
                image_paths,
                detections_dir,
                valid_ids,
                max_id_hamming_distance,
            )
            # Save detections to cache
            save_target_detections(raw, cam_name, output_dir)

        # Reassign globally unique image indices and keep only images with enough detections
        cam_detections: List[ImageDetections] = []
        for det in raw:
            if len(det.detections) >= min_detections:
                cam_detections.append(ImageDetections(
                    image_path=det.image_path,
                    image_index=global_index,
                    width=det.width,
                    height=det.height,
                    detections=det.detections,
                ))
            global_index += 1

        all_detections[cam_name] = cam_detections
        print(f"  {cam_name}: {len(cam_detections)} images with >= {min_detections} detections", flush=True)

    return all_detections


# ---------------------------------------------------------------------------
# Step 3 — Initialize multi-camera state from SfM poses
# ---------------------------------------------------------------------------

def _compute_relative_pose(
    poses_cam0: Dict[str, np.ndarray],
    poses_camk: Dict[str, np.ndarray],
) -> np.ndarray:
    """Estimate relative pose cam0→camK from overlapping frames.

    Returns the median relative transformation as a 6-vector.
    """
    shared = sorted(set(poses_cam0) & set(poses_camk))
    if not shared:
        raise RuntimeError("No shared frames between cameras for relative pose computation.")

    relative_translations: List[np.ndarray] = []
    relative_rotations: List[np.ndarray] = []
    for stem in shared:
        R0 = rotation_matrix_from_pose(poses_cam0[stem])
        t0 = poses_cam0[stem][3:]
        Rk = rotation_matrix_from_pose(poses_camk[stem])
        tk = poses_camk[stem][3:]
        # Relative: R_rel = Rk @ R0^T,  t_rel = tk - R_rel @ t0
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        relative_rotations.append(R_rel)
        relative_translations.append(t_rel)

    # Take the median translation
    median_t = np.median(np.array(relative_translations), axis=0)

    # For rotation, take the one closest to the median Rodrigues vector
    rvecs = [cv2.Rodrigues(R)[0].reshape(3) for R in relative_rotations]
    median_rvec = np.median(np.array(rvecs), axis=0)
    best_idx = int(np.argmin([np.linalg.norm(rv - median_rvec) for rv in rvecs]))
    best_rvec = rvecs[best_idx]

    return np.concatenate([best_rvec, median_t]).astype(np.float64)


def initialize_multi_camera_state(
    sfm_intrinsics: Dict[str, np.ndarray],
    sfm_poses: Dict[str, Dict[str, np.ndarray]],
    detections_by_camera: Dict[str, List[ImageDetections]],
    camera_names: List[str],
    max_reprojection_error: float = 12.0,
    init_reproj_tolerance: float = 50.0,
    known_targets3d: Dict[int, np.ndarray] | None = None,
    known_baseline: float | None = None,
) -> MultiCameraState:
    """Build a MultiCameraState from SfM outputs and CCT detections.

    For each rig frame, we take cam-0's SfM pose as the rig pose.
    Camera intrinsics come from SfM as initial values.
    3D target positions are triangulated from multi-view CCT observations,
    or taken from ``known_targets3d`` when available.
    """
    ref_cam = camera_names[0]

    # Compute relative poses
    relative_poses: Dict[str, np.ndarray] = {ref_cam: identity_pose()}
    scale_factor = 1.0
    for cam_name in camera_names[1:]:
        relative_poses[cam_name] = _compute_relative_pose(
            sfm_poses[ref_cam], sfm_poses[cam_name],
        )
        # If known baseline is specified, compute a uniform scale factor
        if known_baseline is not None:
            rp = relative_poses[cam_name]
            t_norm = float(np.linalg.norm(rp[3:]))
            if t_norm > 1e-9:
                scale_factor = known_baseline / t_norm

    # Apply scale factor to ALL translations (rig poses + relative poses)
    if known_baseline is not None and abs(scale_factor - 1.0) > 1e-12:
        for cam_name in camera_names[1:]:
            relative_poses[cam_name][3:] *= scale_factor

    for cam_name in camera_names[1:]:
        print(f"relative pose {ref_cam}→{cam_name}: "
              f"t=[{relative_poses[cam_name][3]:.4f}, {relative_poses[cam_name][4]:.4f}, {relative_poses[cam_name][5]:.4f}]"
              f" (|t|={float(np.linalg.norm(relative_poses[cam_name][3:])):.4f} m)",
              flush=True)

    # Build rig poses from cam-0's SfM poses, keyed by image stem
    # Scale translations to match the metric baseline if specified
    rig_poses: Dict[str, np.ndarray] = {}
    for stem, pose in sfm_poses[ref_cam].items():
        p = pose.copy()
        if known_baseline is not None and abs(scale_factor - 1.0) > 1e-12:
            p[3:] *= scale_factor
        rig_poses[stem] = p

    # Build image_index → (cam_name, stem) mapping
    index_to_cam_stem: Dict[int, Tuple[str, str]] = {}
    for cam_name, dets in detections_by_camera.items():
        for det in dets:
            stem = det.image_path.stem
            index_to_cam_stem[det.image_index] = (cam_name, stem)

    # Collect all target observations with their absolute poses
    # observation = (target_id, cam_name, stem, 2D point)
    target_observations: Dict[int, List[Tuple[str, str, np.ndarray]]] = {}
    for cam_name, dets in detections_by_camera.items():
        cam_intrinsics = sfm_intrinsics[cam_name]
        for det in dets:
            stem = det.image_path.stem
            if stem not in rig_poses:
                continue
            for target_id, pt2d in det.detections.items():
                target_observations.setdefault(target_id, []).append(
                    (cam_name, stem, pt2d)
                )

    # Use known 3D points where available, triangulate the rest
    points: Dict[int, np.ndarray] = {}
    fixed_point_ids: Set[int] = set()
    if known_targets3d:
        for tid, pt3d in known_targets3d.items():
            if tid in target_observations:
                points[tid] = pt3d.copy()
                fixed_point_ids.add(tid)
        print(f"  {len(fixed_point_ids)} targets matched from known targets3D file", flush=True)

    reject_reasons: Dict[str, int] = {"too_few": 0, "homogeneous": 0, "non_finite": 0,
                                      "depth": 0, "reprojection": 0}
    all_median_errors: List[float] = []  # for diagnostics
    for target_id, obs_list in target_observations.items():
        if target_id in points:  # already known
            continue
        if len(obs_list) < 2:
            reject_reasons["too_few"] += 1
            continue

        # Build projection matrices for DLT triangulation
        proj_matrices: List[np.ndarray] = []
        pts_2d: List[np.ndarray] = []
        for cam_name, stem, pt2d in obs_list:
            rig_pose = rig_poses[stem]
            abs_pose = compose_poses(rig_pose, relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            t = abs_pose[3:].reshape(3, 1)
            K = intrinsics_matrix(sfm_intrinsics[cam_name])
            P = K @ np.hstack([R, t])
            proj_matrices.append(P)
            pts_2d.append(pt2d)

        # Multi-view DLT triangulation (uses all views)
        A = np.zeros((2 * len(proj_matrices), 4), dtype=np.float64)
        for i, (P, pt) in enumerate(zip(proj_matrices, pts_2d)):
            A[2 * i] = pt[0] * P[2] - P[0]
            A[2 * i + 1] = pt[1] * P[2] - P[1]
        _, _, Vt = np.linalg.svd(A)
        pt4d = Vt[-1]

        if abs(pt4d[3]) < 1e-12:
            reject_reasons["homogeneous"] += 1
            continue
        pt3d = (pt4d[:3] / pt4d[3]).astype(np.float64)
        if not np.all(np.isfinite(pt3d)):
            reject_reasons["non_finite"] += 1
            continue

        # Depth check + reprojection check.
        # SfM poses are approximate, so use a generous tolerance here;
        # bundle adjustment will tighten things up later.
        n_depth_ok = 0
        reproj_errors: List[float] = []
        for cam_name, stem, pt2d in obs_list:
            rig_pose = rig_poses[stem]
            abs_pose = compose_poses(rig_pose, relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            depth = (R @ pt3d + abs_pose[3:])[2]
            if depth > 1e-6:
                n_depth_ok += 1
            projected = project_point(sfm_intrinsics[cam_name], abs_pose, pt3d)
            err = float(np.linalg.norm(projected - pt2d))
            reproj_errors.append(err)

        # Reject only if majority of views have negative depth
        if n_depth_ok < max(1, len(obs_list) // 2):
            reject_reasons["depth"] += 1
            continue
        # Reject if median reprojection error is beyond the generous init tolerance
        median_err = float(np.median(reproj_errors))
        all_median_errors.append(median_err)
        if median_err > init_reproj_tolerance:
            reject_reasons["reprojection"] += 1
            continue

        points[target_id] = pt3d

    print(f"initialized {len(points)} 3D target points from {len(target_observations)} target tracks "
          f"({len(fixed_point_ids)} known, {len(points) - len(fixed_point_ids)} triangulated)", flush=True)
    print(f"  rejection reasons: {reject_reasons}", flush=True)
    if all_median_errors:
        me = np.array(all_median_errors)
        print(f"  median reprojection errors across targets: "
              f"min={me.min():.1f} p25={np.percentile(me,25):.1f} "
              f"median={np.median(me):.1f} p75={np.percentile(me,75):.1f} "
              f"max={me.max():.1f} px", flush=True)
    print(f"initialized {len(rig_poses)} rig frames", flush=True)

    state = MultiCameraState(
        camera_intrinsics={cn: sfm_intrinsics[cn].copy() for cn in camera_names},
        rig_poses=rig_poses,
        relative_poses=relative_poses,
        points=points,
        camera_names=camera_names,
    )
    state._fixed_point_ids = fixed_point_ids  # type: ignore[attr-defined]
    return state


# ---------------------------------------------------------------------------
# Step 4 — Multi-camera bundle adjustment
# ---------------------------------------------------------------------------

class MultiCamReprojectionCost(pyceres.CostFunction):
    """Reprojection residual for a single observation.

    Parameter blocks: [intrinsics(8), rig_pose(6), relative_pose(6), point(3)]
    The absolute camera pose is: compose(rig_pose, relative_pose).
    """

    def __init__(self, observation: np.ndarray):
        super().__init__()
        self.observation = observation.astype(np.float64)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([8, 6, 6, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        intrinsics = np.array(parameters[0], dtype=np.float64, copy=True)
        rig_pose = np.array(parameters[1], dtype=np.float64, copy=True)
        rel_pose = np.array(parameters[2], dtype=np.float64, copy=True)
        point = np.array(parameters[3], dtype=np.float64, copy=True)

        abs_pose = compose_poses(rig_pose, rel_pose)
        prediction = project_point(intrinsics, abs_pose, point)
        residual_vec = prediction - self.observation
        residuals[0] = float(residual_vec[0])
        residuals[1] = float(residual_vec[1])

        if jacobians is not None:
            param_blocks = [intrinsics, rig_pose, rel_pose, point]
            for block_idx, block in enumerate(param_blocks):
                if jacobians[block_idx] is None:
                    continue
                jac = np.zeros((2, block.shape[0]), dtype=np.float64)
                for col in range(block.shape[0]):
                    step = 1e-6 * max(1.0, abs(block[col]))
                    plus_blocks = [b.copy() for b in param_blocks]
                    minus_blocks = [b.copy() for b in param_blocks]
                    plus_blocks[block_idx][col] += step
                    minus_blocks[block_idx][col] -= step

                    abs_plus = compose_poses(plus_blocks[1], plus_blocks[2])
                    abs_minus = compose_poses(minus_blocks[1], minus_blocks[2])
                    res_plus = project_point(plus_blocks[0], abs_plus, plus_blocks[3]) - self.observation
                    res_minus = project_point(minus_blocks[0], abs_minus, minus_blocks[3]) - self.observation
                    jac[:, col] = (res_plus - res_minus) / (2.0 * step)

                for row in range(2):
                    for col in range(block.shape[0]):
                        jacobians[block_idx][row * block.shape[0] + col] = jac[row, col]
        return True


def multi_cam_collect_observations(
    detections_by_camera: Dict[str, List[ImageDetections]],
    state: MultiCameraState,
    max_reprojection_error: float,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Collect valid observations across all cameras.

    Returns list of (cam_name, frame_stem, target_id, 2D_point).
    """
    observations: List[Tuple[str, str, int, np.ndarray]] = []
    for cam_name, dets in detections_by_camera.items():
        intrinsics = state.camera_intrinsics[cam_name]
        rel_pose = state.relative_poses[cam_name]
        for det in dets:
            stem = det.image_path.stem
            rig_pose = state.rig_poses.get(stem)
            if rig_pose is None:
                continue
            abs_pose = compose_poses(rig_pose, rel_pose)
            for target_id, pt2d in det.detections.items():
                pt3d = state.points.get(target_id)
                if pt3d is None:
                    continue
                residual = project_point(intrinsics, abs_pose, pt3d) - pt2d
                if not np.all(np.isfinite(residual)):
                    continue
                if np.linalg.norm(residual) <= max_reprojection_error:
                    observations.append((cam_name, stem, target_id, pt2d))
    return observations


def multi_cam_filter_inlier_observations(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    max_error: float,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Keep only observations below the reprojection error threshold."""
    inliers: List[Tuple[str, str, int, np.ndarray]] = []
    for cam_name, stem, target_id, pt2d in observations:
        intrinsics = state.camera_intrinsics[cam_name]
        rig_pose = state.rig_poses.get(stem)
        pt3d = state.points.get(target_id)
        if rig_pose is None or pt3d is None:
            continue
        abs_pose = compose_poses(rig_pose, state.relative_poses[cam_name])
        residual = project_point(intrinsics, abs_pose, pt3d) - pt2d
        if np.all(np.isfinite(residual)) and np.linalg.norm(residual) <= max_error:
            inliers.append((cam_name, stem, target_id, pt2d))
    return inliers


def multi_cam_compute_reprojection_errors(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for cam_name, stem, target_id, pt2d in observations:
        intrinsics = state.camera_intrinsics[cam_name]
        rig_pose = state.rig_poses.get(stem)
        pt3d = state.points.get(target_id)
        if rig_pose is None or pt3d is None:
            continue
        abs_pose = compose_poses(rig_pose, state.relative_poses[cam_name])
        residual = project_point(intrinsics, abs_pose, pt3d) - pt2d
        if np.all(np.isfinite(residual)):
            errors.append(float(np.linalg.norm(residual)))
    return np.array(errors, dtype=np.float64)


def solve_multi_cam_bundle_adjustment(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    max_iterations: int = 500,
    huber_delta: float = 3.0,
    loss_relative_tolerance: float = 1e-5,
    loss_patience: int = 5,
    robust_loss: str = "huber",
    cauchy_scale: float = 3.0,
    fix_relative_poses: bool = True,
    fixed_point_ids: Set[int] | None = None,
) -> SolverDiagnostics:
    """Joint bundle adjustment over all cameras.

    Parameter blocks per observation:
      - camera intrinsics (8)  — shared per camera
      - rig pose (6)          — shared per frame
      - relative pose (6)     — fixed per non-reference camera
      - 3D point (3)

    The first rig frame is held constant (gauge freedom).
    Relative poses are held constant to enforce the rig constraint.
    """
    problem = pyceres.Problem()
    if robust_loss == "cauchy":
        loss = pyceres.CauchyLoss(cauchy_scale)
    else:
        loss = pyceres.HuberLoss(huber_delta)

    used_rig_poses: set = set()
    used_points: set = set()

    for cam_name, stem, target_id, pt2d in observations:
        cost = MultiCamReprojectionCost(pt2d)
        problem.add_residual_block(
            cost,
            loss,
            [
                state.camera_intrinsics[cam_name],
                state.rig_poses[stem],
                state.relative_poses[cam_name],
                state.points[target_id],
            ],
        )
        used_rig_poses.add(stem)
        used_points.add(target_id)

    # Fix the first rig frame (gauge freedom) — unless known 3D
    # target points already anchor the coordinate frame.
    if not fixed_point_ids:
        first_frame = sorted(used_rig_poses)[0]
        problem.set_parameter_block_constant(state.rig_poses[first_frame])

    # Fix relative poses (rig constraint)
    if fix_relative_poses:
        for cam_name in state.camera_names:
            problem.set_parameter_block_constant(state.relative_poses[cam_name])

    # Fix known 3D points (metric ground truth)
    if fixed_point_ids:
        for tid in fixed_point_ids:
            if tid in used_points:
                problem.set_parameter_block_constant(state.points[tid])

    # Intrinsics bounds (for each camera)
    for cam_name in state.camera_names:
        intr = state.camera_intrinsics[cam_name]
        # We need at least one observation for this camera in the problem
        if not any(cn == cam_name for cn, _, _, _ in observations):
            continue

        # Determine image size from first observation of this camera
        # Use a reasonable max_dimension from existing intrinsics
        max_dim = max(intr[0], intr[1]) * 2.0  # rough estimate
        cx_est = intr[2]
        cy_est = intr[3]
        w_est = cx_est * 2.0
        h_est = cy_est * 2.0

        problem.set_parameter_lower_bound(intr, 0, 0.2 * max_dim)
        problem.set_parameter_lower_bound(intr, 1, 0.2 * max_dim)
        problem.set_parameter_upper_bound(intr, 0, 3.0 * max_dim)
        problem.set_parameter_upper_bound(intr, 1, 3.0 * max_dim)
        problem.set_parameter_lower_bound(intr, 2, 0.25 * w_est)
        problem.set_parameter_upper_bound(intr, 2, 0.75 * w_est)
        problem.set_parameter_lower_bound(intr, 3, 0.25 * h_est)
        problem.set_parameter_upper_bound(intr, 3, 0.75 * h_est)
        for di in range(4, 8):
            problem.set_parameter_lower_bound(intr, di, -1.0)
            problem.set_parameter_upper_bound(intr, di, 1.0)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
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
        cost_history=[float(c) for c in cost_history],
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_multi_cam_colmap_output(
    state: MultiCameraState,
    detections_by_camera: Dict[str, List[ImageDetections]],
    output_dir: Path,
) -> Path:
    """Save joint calibration results in COLMAP text format."""
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt — one camera per sensor
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy,k1,k2,p1,p2]",
    ]
    cam_id_map: Dict[str, int] = {}
    for idx, cam_name in enumerate(state.camera_names):
        cam_id = idx + 1
        cam_id_map[cam_name] = cam_id
        intr = state.camera_intrinsics[cam_name]
        # Get image size from detections
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height
        cam_lines.append(
            f"{cam_id} OPENCV {w} {h} "
            f"{intr[0]:.9f} {intr[1]:.9f} {intr[2]:.9f} {intr[3]:.9f} "
            f"{intr[4]:.9f} {intr[5]:.9f} {intr[6]:.9f} {intr[7]:.9f}"
        )
    (colmap_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    # images.txt — all registered images
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    # Build point3D ID mapping
    track_ids = sorted(state.points.keys())
    track_to_colmap_id = {tid: i + 1 for i, tid in enumerate(track_ids)}

    colmap_img_id = 1
    for stem in sorted(state.rig_poses.keys()):
        for cam_name in state.camera_names:
            abs_pose = compose_poses(state.rig_poses[stem], state.relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            t = abs_pose[3:]
            quat = rotation_matrix_to_quaternion(R)
            qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])

            img_name = f"{cam_name}/{stem}.jpg"
            img_lines.append(
                f"{colmap_img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam_id_map[cam_name]} {img_name}"
            )

            # 2D points line
            pts2d_parts: List[str] = []
            # Find detections for this camera+stem
            for det in detections_by_camera.get(cam_name, []):
                if det.image_path.stem == stem:
                    for target_id, pt2d in sorted(det.detections.items()):
                        colmap_pt_id = track_to_colmap_id.get(target_id, -1)
                        pts2d_parts.append(f"{pt2d[0]:.4f} {pt2d[1]:.4f} {colmap_pt_id}")
                    break
            img_lines.append(" ".join(pts2d_parts) if pts2d_parts else "")
            colmap_img_id += 1

    (colmap_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    # points3D.txt
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for tid in track_ids:
        pt = state.points[tid]
        colmap_pt_id = track_to_colmap_id[tid]
        pts_lines.append(f"{colmap_pt_id} {pt[0]:.9f} {pt[1]:.9f} {pt[2]:.9f} 200 200 200 0.0")
    (colmap_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n", encoding="utf-8")

    return colmap_dir


def save_report(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    detections_by_camera: Dict[str, List[ImageDetections]],
    diagnostics: SolverDiagnostics,
    output_dir: Path,
) -> Path:
    """Write a human-readable calibration report to report.txt."""
    reproj_errors = multi_cam_compute_reprojection_errors(state, observations)
    track_ids = {tid for _, _, tid, _ in observations}
    frame_stems = {stem for _, stem, _, _ in observations}

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("MULTI-CAMERA CCT CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Cameras:           {', '.join(state.camera_names)}")
    lines.append(f"Registered frames: {len(frame_stems)}")
    lines.append(f"3D targets:        {len(track_ids)}")
    lines.append(f"Observations:      {len(observations)}")
    lines.append("")

    lines.append("--- Reprojection errors ---")
    if reproj_errors.size:
        lines.append(f"  Mean: {float(np.mean(reproj_errors)):.4f} px")
        lines.append(f"  RMS:  {float(np.sqrt(np.mean(np.square(reproj_errors)))):.4f} px")
        lines.append(f"  Max:  {float(np.max(reproj_errors)):.4f} px")
        lines.append(f"  Median: {float(np.median(reproj_errors)):.4f} px")
    lines.append("")

    lines.append("--- Solver ---")
    lines.append(f"  Iterations:   {diagnostics.iterations}")
    lines.append(f"  Initial cost: {diagnostics.initial_cost:.6f}")
    lines.append(f"  Final cost:   {diagnostics.final_cost:.6f}")
    lines.append(f"  {diagnostics.summary}")
    lines.append("")

    for cam_name in state.camera_names:
        intr = state.camera_intrinsics[cam_name]
        cam_obs = [(cn, s, t, p) for cn, s, t, p in observations if cn == cam_name]
        cam_errors = multi_cam_compute_reprojection_errors(state, cam_obs)

        # Image resolution from detections
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height

        lines.append(f"--- {cam_name} ---")
        lines.append(f"  Resolution: {w} x {h}")
        lines.append(f"  Observations: {len(cam_obs)}")
        if cam_errors.size:
            lines.append(f"  Mean error: {float(np.mean(cam_errors)):.4f} px")
            lines.append(f"  RMS error:  {float(np.sqrt(np.mean(np.square(cam_errors)))):.4f} px")
        lines.append(f"  Intrinsics (OPENCV):")
        lines.append(f"    fx = {intr[0]:.6f}")
        lines.append(f"    fy = {intr[1]:.6f}")
        lines.append(f"    cx = {intr[2]:.6f}")
        lines.append(f"    cy = {intr[3]:.6f}")
        lines.append(f"    k1 = {intr[4]:.6f}")
        lines.append(f"    k2 = {intr[5]:.6f}")
        lines.append(f"    p1 = {intr[6]:.6f}")
        lines.append(f"    p2 = {intr[7]:.6f}")

        rp = state.relative_poses[cam_name]
        R_rel = rotation_matrix_from_pose(rp)
        t_rel = rp[3:]
        lines.append(f"  Relative pose (w.r.t. {state.camera_names[0]}):")
        lines.append(f"    rvec = [{rp[0]:.6f}, {rp[1]:.6f}, {rp[2]:.6f}]")
        lines.append(f"    tvec = [{t_rel[0]:.6f}, {t_rel[1]:.6f}, {t_rel[2]:.6f}]")
        baseline = float(np.linalg.norm(t_rel))
        lines.append(f"    baseline = {baseline:.6f} m")
        lines.append("")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_kalibr_camchain(
    state: MultiCameraState,
    detections_by_camera: Dict[str, List[ImageDetections]],
    output_dir: Path,
) -> Path:
    """Save camera chain in Kalibr camchain.yaml format.

    Kalibr format uses a 4x4 T_cn_cnm1 matrix expressing the transformation
    from camera n-1 to camera n.  Camera 0 has no such field.
    Distortion model: radtan [k1, k2, p1, p2].
    """
    camchain: dict = {}

    for idx, cam_name in enumerate(state.camera_names):
        intr = state.camera_intrinsics[cam_name]
        rp = state.relative_poses[cam_name]

        # Image resolution
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height

        cam_key = f"cam{idx}"
        cam_entry: dict = {
            "camera_model": "pinhole",
            "intrinsics": [float(intr[0]), float(intr[1]),
                           float(intr[2]), float(intr[3])],
            "distortion_model": "radtan",
            "distortion_coeffs": [float(intr[4]), float(intr[5]),
                                  float(intr[6]), float(intr[7])],
            "resolution": [int(w), int(h)],
            "rostopic": f"/{cam_name}/image_raw",
        }

        # T_cn_cnm1: 4x4 transformation from camera (n-1) to camera n
        if idx > 0:
            R = rotation_matrix_from_pose(rp)
            t = rp[3:]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            cam_entry["T_cn_cnm1"] = T.tolist()

        camchain[cam_key] = cam_entry

    yaml_path = output_dir / "camchain.yaml"
    yaml_path.write_text(yaml.dump(camchain, default_flow_style=None, sort_keys=False),
                         encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_combined_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined multi-camera CCT calibration: SfM initialisation → CCT detection → joint BA."
    )
    parser.add_argument(
        "--image-root", type=Path, required=True,
        help="Root directory containing cam0/, cam1/, ... subdirectories with images.",
    )
    parser.add_argument(
        "--camera", action="append", dest="cameras",
        help="Camera subdirectory name (e.g. cam0). Repeat for each camera. "
             "If omitted, auto-detected from image-root.",
    )
    parser.add_argument(
        "--known-baseline", type=float, default=None,
        help="Known distance (metres) between cam0 and cam1 for SfM scale.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("combined_output"))
    parser.add_argument("--force-sfm", action="store_true",
                        help="Re-run SfM even if a cached reconstruction exists.")
    parser.add_argument("--force-detections", action="store_true",
                        help="Re-run CCT detection even if cached detections exist.")
    parser.add_argument("--targets3d", type=Path, default=None,
                        help="TSV file with known 3D target positions: target_id x y z")
    parser.add_argument("--min-detections", type=int, default=5)
    parser.add_argument("--min-shared", type=int, default=6)
    parser.add_argument("--max-reprojection-error", type=float, default=12.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--loss-tolerance", type=float, default=1e-5)
    parser.add_argument("--loss-patience", type=int, default=5)
    parser.add_argument("--valid-ids-file", type=Path, default=None)
    parser.add_argument("--valid-id-max", type=int, default=None)
    parser.add_argument("--max-id-hamming-distance", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_combined_args()
    image_root: Path = args.image_root.resolve()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect cameras
    if args.cameras:
        camera_names = args.cameras
    else:
        camera_names = sorted(
            d.name for d in image_root.iterdir()
            if d.is_dir() and any(d.glob("*.jpg"))
        )
    if len(camera_names) < 2:
        print(f"ERROR: need at least 2 cameras, found {camera_names}", flush=True)
        return 1
    print(f"cameras: {camera_names}", flush=True)

    # Resolve valid IDs
    valid_ids: Set[int] | None = None
    if args.valid_ids_file:
        from cct_calibration.run import parse_valid_ids_file
        valid_ids = parse_valid_ids_file(args.valid_ids_file)
    elif args.valid_id_max is not None:
        valid_ids = set(range(args.valid_id_max + 1))

    # ── Step 1: SfM ──────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("STEP 1: Structure-from-Motion (pycolmap)", flush=True)
    print("=" * 60, flush=True)
    sfm_dir = output_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    sfm_intrinsics, sfm_poses = run_sfm(
        image_root, camera_names, sfm_dir,
        known_baseline=args.known_baseline,
        force=args.force_sfm,
    )
    for cam_name in camera_names:
        intr = sfm_intrinsics[cam_name]
        n_poses = len(sfm_poses[cam_name])
        print(f"  {cam_name}: {n_poses} poses, "
              f"fx={intr[0]:.1f} fy={intr[1]:.1f} cx={intr[2]:.1f} cy={intr[3]:.1f}",
              flush=True)

    # ── Step 2: CCT Detection ────────────────────────────────────
    print("=" * 60, flush=True)
    print("STEP 2: CCT target detection", flush=True)
    print("=" * 60, flush=True)

    detections_by_camera = detect_all_cameras(
        image_root, camera_names, output_dir,
        valid_ids=valid_ids,
        max_id_hamming_distance=args.max_id_hamming_distance,
        min_detections=args.min_detections,
        force=args.force_detections,
    )

    # ── Step 3: Initialize multi-camera state ────────────────────
    print("=" * 60, flush=True)
    print("STEP 3: Initialize multi-camera state", flush=True)
    print("=" * 60, flush=True)

    # Load known 3D targets if provided
    known_targets3d = load_targets3d(args.targets3d) if args.targets3d else None

    state = initialize_multi_camera_state(
        sfm_intrinsics, sfm_poses, detections_by_camera, camera_names,
        max_reprojection_error=args.max_reprojection_error,
        known_targets3d=known_targets3d,
        known_baseline=args.known_baseline,
    )
    fixed_point_ids: Set[int] = getattr(state, '_fixed_point_ids', set())

    # ── Step 4: Joint bundle adjustment ──────────────────────────
    print("=" * 60, flush=True)
    print("STEP 4: Joint multi-camera bundle adjustment", flush=True)
    print("=" * 60, flush=True)

    observations = multi_cam_collect_observations(
        detections_by_camera, state, args.max_reprojection_error,
    )
    print(f"initial observations: {len(observations)}", flush=True)
    if len(observations) < 10:
        print("ERROR: not enough observations for bundle adjustment", flush=True)
        return 1

    # First BA (Huber)
    diagnostics = solve_multi_cam_bundle_adjustment(
        state, observations,
        max_iterations=args.max_iterations,
        huber_delta=args.huber_delta,
        loss_relative_tolerance=args.loss_tolerance,
        loss_patience=args.loss_patience,
        fixed_point_ids=fixed_point_ids,
    )
    print(f"  first BA: {diagnostics.summary}", flush=True)

    # Second pass: re-collect with tighter data
    observations = multi_cam_collect_observations(
        detections_by_camera, state, args.max_reprojection_error,
    )
    diagnostics = solve_multi_cam_bundle_adjustment(
        state, observations,
        max_iterations=args.max_iterations,
        huber_delta=args.huber_delta,
        loss_relative_tolerance=args.loss_tolerance,
        loss_patience=args.loss_patience,
        fixed_point_ids=fixed_point_ids,
    )
    print(f"  second BA: {diagnostics.summary}", flush=True)

    # Robust refinement (Cauchy)
    wide_observations = multi_cam_collect_observations(
        detections_by_camera, state, args.max_reprojection_error * 3.0,
    )
    if len(wide_observations) >= len(observations):
        print(f"  robust refinement: {len(wide_observations)} observations "
              f"(was {len(observations)})", flush=True)
        diagnostics = solve_multi_cam_bundle_adjustment(
            state, wide_observations,
            max_iterations=args.max_iterations,
            huber_delta=args.huber_delta,
            loss_relative_tolerance=args.loss_tolerance,
            loss_patience=args.loss_patience,
            robust_loss="cauchy",
            cauchy_scale=3.0,
            fixed_point_ids=fixed_point_ids,
        )
        observations = multi_cam_filter_inlier_observations(
            state, wide_observations, args.max_reprojection_error,
        )
        print(f"  kept {len(observations)}/{len(wide_observations)} inlier observations", flush=True)

        # Final clean BA
        diagnostics = solve_multi_cam_bundle_adjustment(
            state, observations,
            max_iterations=args.max_iterations,
            huber_delta=args.huber_delta,
            loss_relative_tolerance=args.loss_tolerance,
            loss_patience=args.loss_patience,
            fixed_point_ids=fixed_point_ids,
        )
        print(f"  final BA: {diagnostics.summary}", flush=True)

    # ── Results ──────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    reproj_errors = multi_cam_compute_reprojection_errors(state, observations)
    track_ids = {tid for _, _, tid, _ in observations}
    frame_stems = {stem for _, stem, _, _ in observations}

    overall = {
        "total_observations": len(observations),
        "total_tracks": len(track_ids),
        "total_frames": len(frame_stems),
        "mean_reprojection_error": float(np.mean(reproj_errors)) if reproj_errors.size else 0.0,
        "rms_reprojection_error": float(np.sqrt(np.mean(np.square(reproj_errors)))) if reproj_errors.size else 0.0,
        "iterations": diagnostics.iterations,
        "initial_cost": diagnostics.initial_cost,
        "final_cost": diagnostics.final_cost,
    }

    # Per-camera results
    per_camera: Dict[str, dict] = {}
    for cam_name in camera_names:
        cam_obs = [(cn, s, t, p) for cn, s, t, p in observations if cn == cam_name]
        cam_errors = multi_cam_compute_reprojection_errors(
            state, cam_obs,
        )
        intr = state.camera_intrinsics[cam_name]
        per_camera[cam_name] = {
            "observations": len(cam_obs),
            "mean_reprojection_error": float(np.mean(cam_errors)) if cam_errors.size else 0.0,
            "rms_reprojection_error": float(np.sqrt(np.mean(np.square(cam_errors)))) if cam_errors.size else 0.0,
            "intrinsics": {
                "fx": float(intr[0]), "fy": float(intr[1]),
                "cx": float(intr[2]), "cy": float(intr[3]),
                "k1": float(intr[4]), "k2": float(intr[5]),
                "p1": float(intr[6]), "p2": float(intr[7]),
            },
        }
    overall["cameras"] = per_camera

    # Relative poses
    rel_poses_out: Dict[str, dict] = {}
    for cam_name in camera_names:
        rp = state.relative_poses[cam_name]
        rel_poses_out[cam_name] = {
            "rvec": rp[:3].tolist(),
            "tvec": rp[3:].tolist(),
        }
    overall["relative_poses"] = rel_poses_out

    summary_path = output_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)

    # Save convergence plot
    save_convergence_plot("combined", diagnostics.cost_history, output_dir)

    # Save COLMAP output
    colmap_dir = save_multi_cam_colmap_output(state, detections_by_camera, output_dir)
    print(f"COLMAP output: {colmap_dir}", flush=True)

    # Save human-readable report
    report_path = save_report(state, observations, detections_by_camera, diagnostics, output_dir)
    print(f"Report: {report_path}", flush=True)

    # Save Kalibr camchain
    kalibr_path = save_kalibr_camchain(state, detections_by_camera, output_dir)
    print(f"Kalibr camchain: {kalibr_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
