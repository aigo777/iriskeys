from __future__ import annotations
import json
import math
import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from calibration import Calibrator

Point = Tuple[float, float]
Gaze = Tuple[float, float]

POSE_CANDIDATE_INDICES = [
    1,      # often nose tip
    2, 4,   # possible nose region
    152,    # often chin
    33,     # left eye outer
    263,    # right eye outer
    61,     # left mouth corner
    291,    # right mouth corner
]

POSE_LANDMARKS = {
    "nose_tip": 4,
    "chin": 152,
    "l_eye_outer": 33,
    "r_eye_outer": 263,
    "l_mouth": 61,
    "r_mouth": 291,
}

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye outer
        (43.3, 32.7, -26.0),      # Right eye outer
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1),     # Right mouth corner
    ],
    dtype=np.float32,
)

POSE_SMOOTH_MIN_CUTOFF = 0.8
POSE_SMOOTH_BETA = 0.02
POSE_SMOOTH_D_CUTOFF = 1.0
POSE_HOLD_MS = 150
ML_FEATURE_ORDER = ("gx", "gy", "yaw", "pitch", "roll", "tz")
ML_MODEL_VERSION = 1


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0
        self._t_prev: Optional[float] = None
        self.last_alpha = 0.0
        self.last_cutoff = float(min_cutoff)
        self.last_dx = 0.0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            self.last_alpha = 1.0
            self.last_cutoff = self.min_cutoff
            self.last_dx = 0.0
            return x

        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self.last_alpha = alpha
        self.last_cutoff = cutoff
        self.last_dx = dx_hat
        return x_hat


class OneEuroFilter2D:
    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
        min_cutoff_y: Optional[float] = None,
        beta_y: Optional[float] = None,
    ) -> None:
        # Y gets a slightly higher cutoff/beta to keep vertical motion responsive.
        if min_cutoff_y is None:
            min_cutoff_y = min_cutoff * 1.35
        if beta_y is None:
            beta_y = beta * 1.5
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff_y, beta=beta_y, d_cutoff=d_cutoff)
        self.last_alpha = 0.0
        self.last_cutoff = 0.0
        self.last_velocity = 0.0

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        fx = self.fx.filter(x, t)
        fy = self.fy.filter(y, t)
        self.last_alpha = (self.fx.last_alpha + self.fy.last_alpha) * 0.5
        self.last_cutoff = (self.fx.last_cutoff + self.fy.last_cutoff) * 0.5
        self.last_velocity = float(math.hypot(self.fx.last_dx, self.fy.last_dx))
        return fx, fy


class GazeTracker:
    """Compute a normalized gaze coordinate from MediaPipe Face Mesh landmarks."""

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    LEFT_OUTER = 33
    LEFT_INNER = 133
    LEFT_UPPER = 159
    LEFT_LOWER = 145

    RIGHT_OUTER = 263
    RIGHT_INNER = 362
    RIGHT_UPPER = 386
    RIGHT_LOWER = 374

    LEFT_UPPER_LID = [159, 158]
    LEFT_LOWER_LID = [145, 153]
    RIGHT_UPPER_LID = [386, 387]
    RIGHT_LOWER_LID = [374, 373]

    CALIB_TARGETS = ("center", "tl", "tr", "br", "bl")
    PANEL_TARGETS = {
        "center": (0.5, 0.5),
        "tl": (0.08, 0.08),
        "tr": (0.92, 0.08),
        "br": (0.92, 0.92),
        "bl": (0.08, 0.92),
    }

    def __init__(
        self,
        camera_index: int = 0,
        max_num_faces: int = 1,
        ema_alpha: float = 0.35,
        gain_x: float = 3.0,
        gain_y: float = 3.0,
        mirror_view: bool = True,
        invert_x: bool = False,
        invert_y: bool = False,
        calib_samples: int = 45,
    ) -> None:
        self.camera_index = camera_index
        self.max_num_faces = max_num_faces
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 0.95))
        self.gain_x = float(np.clip(gain_x, 0.5, 8.0))
        self.gain_y = float(np.clip(gain_y, 0.5, 8.0))
        self.mirror_view = bool(mirror_view)
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.vertical_gain = 1.6
        self._vertical_gain_base = self.vertical_gain
        self.deadzone = 0.03
        self.gamma = 0.7
        self.axis_flip_x = False
        self.axis_flip_y = False
        self._map_x: Optional[Dict[str, float]] = None
        self._map_y: Optional[Dict[str, float]] = None
        self._map_spans: Optional[Dict[str, float]] = None
        self._piecewise_ok = False
        self._edge_counter_x = 0
        self._edge_counter_y = 0
        self._edge_hold_frames = 6
        self._edge_threshold = 0.08
        self._edge_pad_x = 0.06
        self._edge_pad_y = 0.02
        self._edge_overshoot_x = 0.05
        self._edge_overshoot_y = 0.06
        self._soft_pad = 0.06
        self._soft_k = 0.35
        self._min_span = 0.04
        self._deadband_enter = 0.018
        self._deadband_exit = 0.030
        self._stable_lock = False
        self._stable_gaze: Optional[Gaze] = None
        self._raw_center_radius = 0.03
        self._center_pull = 0.15
        self._gaze_window: Deque[Gaze] = deque(maxlen=5)
        self._last_gaze_time: Optional[float] = None
        self._hold_ms = 150
        self._last_velocity = 0.0
        self._last_alpha = self.ema_alpha
        self._last_cutoff = 0.0
        self._one_euro = OneEuroFilter2D(
            min_cutoff=0.8,
            beta=0.02,
            d_cutoff=1.0,
            min_cutoff_y=1.1,
            beta_y=0.04,
        )
        self._calib_quality: Dict[str, Dict[str, float]] = {}
        self._last_openness: Optional[float] = None
        self._open_ref: Optional[float] = None
        self._open_beta_y: float = 0.0
        self._blink_freeze = True
        self._blink_ratio = 0.60
        self._open_min_abs = 0.12
        self._last_good_gaze: Optional[Gaze] = None
        self._invalid_since: Optional[float] = None
        self._reacquire_count = 0
        self._reacquiring = False
        self._recovery_hold_s = 0.18
        self._reacquire_needed_frames = 3

        self._gaze_smooth: Optional[Gaze] = None
        self._calib_center: Optional[Gaze] = None
        self._calib: Dict[str, Gaze] = {}
        self._calib_range: Optional[Tuple[float, float, float, float]] = None
        self._calib_active = False
        self._calib_index = 0
        self._calib_samples: List[Gaze] = []
        self._calib_samples_needed = int(max(5, calib_samples))

        self._last_raw: Optional[Gaze] = None
        self._last_mapped: Optional[Gaze] = None
        self._pose_prev: Optional[Tuple[float, float, float]] = None
        self._pose_unwrap_debug = False
        self._reset_pose_filters()
        self._clear_ml_calibration()

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, object]:
        """Run MediaPipe Face Mesh and compute gaze data for a single frame."""
        timestamp = time.time()
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._gaze_smooth = None
            self._last_raw = None
            self._last_mapped = None
            self._last_openness = None
            self._last_head_pose_for_ml = None
            self._reset_pose_filters()
            return {
                "face_detected": False,
                "landmarks_norm": None,
                "points_norm": None,
                "points_px": None,
                "iris_ring_px": None,
                "gaze_raw_uncal": None,
                "gaze_mapped": None,
                "gaze_smooth": None,
                "iris_radius": None,
                "eye_openness": None,
                "head_pose": None,
                "head_pose_raw": None,
                "gaze_features": None,
                "timestamp": timestamp,
            }

        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks_norm: List[Point] = [(lm.x, lm.y) for lm in face_landmarks]
        head_pose_raw = self._estimate_head_pose(landmarks_norm, w, h)
        head_pose = self._smooth_head_pose(head_pose_raw, timestamp)
        self._last_head_pose_for_ml = head_pose

        left_iris_pts = [landmarks_norm[i] for i in self.LEFT_IRIS]
        right_iris_pts = [landmarks_norm[i] for i in self.RIGHT_IRIS]
        left_upper_pts = [landmarks_norm[i] for i in self.LEFT_UPPER_LID]
        left_lower_pts = [landmarks_norm[i] for i in self.LEFT_LOWER_LID]
        right_upper_pts = [landmarks_norm[i] for i in self.RIGHT_UPPER_LID]
        right_lower_pts = [landmarks_norm[i] for i in self.RIGHT_LOWER_LID]
        left_upper_mean = self._mean_point(left_upper_pts)
        left_lower_mean = self._mean_point(left_lower_pts)
        right_upper_mean = self._mean_point(right_upper_pts)
        right_lower_mean = self._mean_point(right_lower_pts)
        left_center, left_radius = self._iris_center_and_radius(left_iris_pts)
        right_center, right_radius = self._iris_center_and_radius(right_iris_pts)

        points_norm = {
            "left_outer": landmarks_norm[self.LEFT_OUTER],
            "left_inner": landmarks_norm[self.LEFT_INNER],
            "left_upper": left_upper_mean,
            "left_lower": left_lower_mean,
            "right_outer": landmarks_norm[self.RIGHT_OUTER],
            "right_inner": landmarks_norm[self.RIGHT_INNER],
            "right_upper": right_upper_mean,
            "right_lower": right_lower_mean,
            "left_iris_center": left_center,
            "right_iris_center": right_center,
            "left_upper_pts": left_upper_pts,
            "left_lower_pts": left_lower_pts,
            "right_upper_pts": right_upper_pts,
            "right_lower_pts": right_lower_pts,
        }

        gaze_raw_uncal, eye_openness, iris_radius, gaze_features = self._compute_gaze(
            points_norm, left_radius, right_radius
        )
        self._last_openness = eye_openness
        eye_width_px = self._compute_eye_width_px(points_norm, w, h)
        face_confidence = self._face_confidence(iris_radius)

        if self._reject_frame(iris_radius, eye_openness, eye_width_px):
            gaze_raw_uncal = None

        self._last_raw = gaze_raw_uncal
        self.update_calibration_state(gaze_raw_uncal)
        gaze_mapped = self.map_gaze(gaze_raw_uncal)
        self._last_mapped = gaze_mapped
        gaze_smooth = self._smooth_gaze(gaze_mapped)

        points_px = {
            name: self._norm_to_px(pt, w, h)
            for name, pt in points_norm.items()
            if isinstance(pt, tuple)
        }
        iris_ring_px = {
            "left": [self._norm_to_px(pt, w, h) for pt in left_iris_pts],
            "right": [self._norm_to_px(pt, w, h) for pt in right_iris_pts],
        }

        return {
            "face_detected": True,
            "landmarks_norm": landmarks_norm,
            "points_norm": points_norm,
            "points_px": points_px,
            "iris_ring_px": iris_ring_px,
            "gaze_raw_uncal": gaze_raw_uncal,
            "gaze_mapped": gaze_mapped,
            "gaze_smooth": gaze_smooth,
            "iris_radius": iris_radius,
            "eye_openness": eye_openness,
            "head_pose": head_pose,
            "head_pose_raw": head_pose_raw,
            "gaze_features": gaze_features,
            "eye_width_px": eye_width_px,
            "face_confidence": face_confidence,
            "timestamp": timestamp,
        }

    def get_normalized_gaze(self, result_dict: Dict[str, object]) -> Optional[Gaze]:
        """Compute normalized raw gaze (x, y) in a 0..1 range from iris/eyelid landmarks."""
        points_norm = result_dict.get("points_norm")
        if not isinstance(points_norm, dict):
            return None
        gaze_raw, _, _, _ = self._compute_gaze(points_norm, None, None)
        return gaze_raw

    def get_mapped_gaze(self, result_dict: Dict[str, object]) -> Optional[Gaze]:
        """Return the mapped (optionally smoothed) gaze for quick rendering."""
        gaze = result_dict.get("gaze_smooth") or result_dict.get("gaze_mapped")
        if isinstance(gaze, tuple):
            return gaze
        return None

    def get_quality_metrics(self, result_dict: Dict[str, object]) -> Dict[str, Optional[float]]:
        """Return simple quality metrics for debug and gating."""
        iris_radius = result_dict.get("iris_radius")
        eye_openness = result_dict.get("eye_openness")
        eye_width_px = result_dict.get("eye_width_px")
        face_confidence = result_dict.get("face_confidence")
        return {
            "iris_radius": iris_radius if isinstance(iris_radius, float) else None,
            "eye_openness": eye_openness if isinstance(eye_openness, float) else None,
            "eye_width_px": float(eye_width_px) if isinstance(eye_width_px, (int, float)) else None,
            "face_confidence": float(face_confidence) if isinstance(face_confidence, (int, float)) else None,
        }

    def set_openness_compensation(self, open_ref: float, beta_y: float) -> None:
        self._open_ref = float(open_ref)
        self._open_beta_y = float(beta_y)

    def is_ml_ready(self) -> bool:
        return bool(self._ml_ready and self._ml_calibrator is not None)

    def _clear_ml_calibration(self) -> None:
        self._ml_calibrator: Optional[Calibrator] = None
        self._ml_ready = False
        self._ml_feature_order = ML_FEATURE_ORDER
        self._last_head_pose_for_ml: Optional[Dict[str, float]] = None
        self._ml_meta: Dict[str, object] = {
            "ready": False,
            "user_id": None,
            "model_file": None,
            "feature_order": list(ML_FEATURE_ORDER),
            "alpha": None,
            "version": ML_MODEL_VERSION,
        }

    def _build_ml_features(self, gaze_raw: Gaze, head_pose: Optional[Dict[str, float]]) -> Optional[np.ndarray]:
        if head_pose is None:
            return None

        try:
            gx = float(gaze_raw[0])
            gy = float(gaze_raw[1])
            yaw = float(head_pose["yaw"])
            pitch = float(head_pose["pitch"])
            roll = float(head_pose["roll"])
            tz = float(head_pose["tz"])
        except (KeyError, TypeError, ValueError):
            return None

        features = np.array([gx, gy, yaw, pitch, roll, tz], dtype=np.float32)
        if not np.isfinite(features).all():
            return None
        return features

    def fit_ml_calibration(
        self,
        X: List[List[float]],
        Y: List[List[float]],
        alpha: float = 0.5,
    ) -> bool:
        calibrator = Calibrator(feature_order=self._ml_feature_order)
        if not calibrator.fit(X, Y, alpha=alpha):
            return False

        self._ml_calibrator = calibrator
        self._ml_ready = True
        self._ml_meta = {
            "ready": True,
            "user_id": self._ml_meta.get("user_id"),
            "model_file": self._ml_meta.get("model_file"),
            "feature_order": list(calibrator.feature_order),
            "alpha": float(calibrator.alpha),
            "version": int(calibrator.version),
        }
        return True

    def predict_ml_gaze(self, features: np.ndarray) -> Optional[Gaze]:
        if not self.is_ml_ready() or self._ml_calibrator is None:
            return None
        try:
            pred = self._ml_calibrator.predict(features.reshape(1, -1))
        except (ValueError, TypeError):
            return None
        if pred.shape != (1, 2):
            return None
        px = float(pred[0, 0])
        py = float(pred[0, 1])
        if not np.isfinite([px, py]).all():
            return None
        return self._clamp01(px), self._clamp01(py)

    def save_ml_model(self, model_path: str, user_id: Optional[str] = None) -> bool:
        if not self.is_ml_ready() or self._ml_calibrator is None:
            return False

        try:
            self._ml_calibrator.save_npz(model_path)
        except (OSError, ValueError):
            return False

        self._ml_meta = {
            "ready": True,
            "user_id": user_id,
            "model_file": os.path.basename(model_path),
            "feature_order": list(self._ml_calibrator.feature_order),
            "alpha": float(self._ml_calibrator.alpha),
            "version": int(self._ml_calibrator.version),
        }
        return True

    def load_ml_model(self, model_path: str, user_id: Optional[str] = None) -> bool:
        if not os.path.exists(model_path):
            return False

        try:
            calibrator = Calibrator.load_npz(model_path)
        except (OSError, ValueError, KeyError):
            self._clear_ml_calibration()
            return False

        self._ml_calibrator = calibrator
        self._ml_ready = True
        self._ml_meta = {
            "ready": True,
            "user_id": user_id,
            "model_file": os.path.basename(model_path),
            "feature_order": list(calibrator.feature_order),
            "alpha": float(calibrator.alpha),
            "version": int(calibrator.version),
        }
        return True

    def _load_ml_from_json(self, data: Dict[str, object], calib_path: str) -> None:
        self._clear_ml_calibration()

        ml_data = data.get("ml")
        if not isinstance(ml_data, dict):
            return
        if not bool(ml_data.get("ready", False)):
            return

        model_file = ml_data.get("model_file")
        if not isinstance(model_file, str) or not model_file:
            return

        model_path = model_file
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(calib_path), model_file)

        user_id = ml_data.get("user_id")
        user_id_str = str(user_id) if isinstance(user_id, str) else None
        if not self.load_ml_model(model_path, user_id=user_id_str):
            return

        feature_order = ml_data.get("feature_order")
        if isinstance(feature_order, list) and all(isinstance(v, str) for v in feature_order):
            self._ml_meta["feature_order"] = feature_order
        alpha = ml_data.get("alpha")
        if isinstance(alpha, (int, float)):
            self._ml_meta["alpha"] = float(alpha)
        version = ml_data.get("version")
        if isinstance(version, int):
            self._ml_meta["version"] = version

    def draw_debug(self, frame_bgr: np.ndarray, result_dict: Dict[str, object]) -> np.ndarray:
        """Draw eye landmarks, gaze text, and a virtual screen panel."""
        face_detected = bool(result_dict.get("face_detected"))
        points_px = result_dict.get("points_px") if face_detected else None
        iris_ring_px = result_dict.get("iris_ring_px") if face_detected else None
        landmarks_norm = result_dict.get("landmarks_norm") if face_detected else None
        head_pose = result_dict.get("head_pose") if face_detected else None

        if face_detected and isinstance(points_px, dict):
            self._draw_eye(
                frame_bgr,
                points_px,
                "left_outer",
                "left_inner",
                "left_upper",
                "left_lower",
                "left_iris_center",
            )
            self._draw_eye(
                frame_bgr,
                points_px,
                "right_outer",
                "right_inner",
                "right_upper",
                "right_lower",
                "right_iris_center",
            )

        if isinstance(iris_ring_px, dict):
            for pt in iris_ring_px.get("left", []):
                cv2.circle(frame_bgr, pt, 2, (0, 0, 255), -1)
            for pt in iris_ring_px.get("right", []):
                cv2.circle(frame_bgr, pt, 2, (0, 0, 255), -1)

        if result_dict.get("show_pose_indices", False) and isinstance(landmarks_norm, list):
            h, w = frame_bgr.shape[:2]
            for idx in POSE_CANDIDATE_INDICES:
                if idx < len(landmarks_norm):
                    x_norm, y_norm = landmarks_norm[idx]
                    x_px = int(x_norm * w)
                    y_px = int(y_norm * h)
                    cv2.circle(frame_bgr, (x_px, y_px), 3, (255, 0, 255), -1)
                    cv2.putText(
                        frame_bgr,
                        str(idx),
                        (x_px + 5, y_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                    )

        gaze_raw = result_dict.get("gaze_raw_uncal")
        gaze_mapped = result_dict.get("gaze_mapped")
        gaze_smooth = result_dict.get("gaze_smooth")
        iris_radius = result_dict.get("iris_radius")
        eye_openness = result_dict.get("eye_openness")

        raw_text = self._format_gaze_text("raw", gaze_raw)
        mapped_text = self._format_gaze_text("map", gaze_mapped)
        smooth_text = self._format_gaze_text("smooth", gaze_smooth)
        radius_text = f"iris_r={iris_radius:.4f}" if isinstance(iris_radius, float) else "iris_r=None"
        open_text = f"open={eye_openness:.3f}" if isinstance(eye_openness, float) else "open=None"

        face_text = "face=YES" if face_detected else "face=NO"
        mirror_text = "mirror=ON" if self.mirror_view else "mirror=OFF"
        calib_override = result_dict.get("calib_status")
        if isinstance(calib_override, str):
            calib_text = f"calib={calib_override}"
        else:
            calib_text = f"calib={self._calib_status_text()}"
        ml_override = result_dict.get("ml_status")
        if isinstance(ml_override, str):
            ml_text = f"ml={ml_override}"
        else:
            ml_text = f"ml={'ON' if self.is_ml_ready() else 'OFF'}"
        gain_text = f"gain=({self.gain_x:.2f},{self.gain_y:.2f})"
        alpha_text = f"alpha={self._last_alpha:.2f} cutoff={self._last_cutoff:.2f} vel={self._last_velocity:.3f}"
        map_x_text = "map_x=None"
        map_y_text = "map_y=None"
        spans_text = "spans=None"
        piecewise_text = f"piecewise={'ON' if self._piecewise_ok else 'OFF'} edge=({self._edge_counter_x},{self._edge_counter_y})"
        pose_text = "pose=None"
        if isinstance(head_pose, dict):
            try:
                yaw = float(head_pose.get("yaw", 0.0))
                pitch = float(head_pose.get("pitch", 0.0))
                roll = float(head_pose.get("roll", 0.0))
                tz = float(head_pose.get("tz", 0.0))
                pose_text = f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} tz={tz:.1f}"
            except (TypeError, ValueError):
                pose_text = "pose=None"
        if self._map_x is not None:
            map_x_text = f"map_x=({self._map_x['L']:.3f},{self._map_x['C']:.3f},{self._map_x['R']:.3f})"
        if self._map_y is not None:
            map_y_text = f"map_y=({self._map_y['T']:.3f},{self._map_y['C']:.3f},{self._map_y['B']:.3f})"
        if self._map_spans is not None:
            spans_text = (
                f"spans=({self._map_spans['spanL']:.3f},{self._map_spans['spanR']:.3f},"
                f"{self._map_spans['spanT']:.3f},{self._map_spans['spanB']:.3f})"
            )

        lines = [
            face_text,
            mirror_text,
            calib_text,
            ml_text,
            f"{gain_text} {alpha_text}",
            map_x_text,
            map_y_text,
            spans_text,
            piecewise_text,
            pose_text,
            raw_text,
            mapped_text,
            smooth_text,
            radius_text,
            open_text,
        ]

        y = 20
        for line in lines:
            cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y += 18

        self._draw_gaze_panel(frame_bgr, gaze_smooth or gaze_mapped or gaze_raw, self.current_calibration_target())

        return frame_bgr

    def set_center_calibration(self, gaze: Gaze) -> None:
        """Store a center gaze reference for calibration and clear full calibration."""
        self._calib_center = gaze
        self._calib = {}
        self._calib_range = None
        self.axis_flip_x = False
        self.axis_flip_y = False
        self.vertical_gain = self._vertical_gain_base
        self._map_x = None
        self._map_y = None
        self._map_spans = None
        self._piecewise_ok = False
        self._edge_counter_x = 0
        self._edge_counter_y = 0
        self._stable_lock = False
        self._stable_gaze = None
        self._calib_active = False
        self._calib_index = 0
        self._calib_samples = []
        self._gaze_smooth = None
        self._gaze_window.clear()
        self._last_gaze_time = None
        self._calib_quality = {}

    def start_full_calibration(self) -> None:
        """Begin full (center + corners) calibration."""
        self._calib_active = True
        self._calib_index = 0
        self._calib_samples = []
        self._calib = {}
        self._calib_range = None
        self._calib_center = None
        self.axis_flip_x = False
        self.axis_flip_y = False
        self.vertical_gain = self._vertical_gain_base
        self._map_x = None
        self._map_y = None
        self._map_spans = None
        self._piecewise_ok = False
        self._edge_counter_x = 0
        self._edge_counter_y = 0
        self._stable_lock = False
        self._stable_gaze = None
        self._gaze_smooth = None
        self._gaze_window.clear()
        self._last_gaze_time = None
        self._calib_quality = {}

    def update_calibration_state(self, gaze_raw: Optional[Gaze]) -> None:
        """Collect samples for the active calibration target."""
        if not self._calib_active:
            return
        if gaze_raw is None:
            return

        self._calib_samples.append(gaze_raw)
        if len(self._calib_samples) < self._calib_samples_needed:
            return

        samples_arr = np.array(self._calib_samples, dtype=np.float32)
        med = np.median(samples_arr, axis=0)
        gx = float(np.clip(med[0], 0.0, 1.0))
        gy = float(np.clip(med[1], 0.0, 1.0))

        if self._calib_index < len(self.CALIB_TARGETS):
            target = self.CALIB_TARGETS[self._calib_index]
            self._calib[target] = (gx, gy)

        self._calib_index += 1
        self._calib_samples = []

        if self._calib_index >= len(self.CALIB_TARGETS):
            self._calib_active = False
            self._calib_center = self._calib.get("center")
            self._calib_range = self._compute_calibration_range(self._calib)

    def _auto_scale_vertical_gain(self, calib: Dict[str, Gaze]) -> None:
        """Auto-scale vertical gain from calibration span to normalize vertical motion."""
        def median_y(keys: List[str]) -> Optional[float]:
            vals = [calib[k][1] for k in keys if k in calib]
            if not vals:
                return None
            return float(np.median(vals))

        top_y = median_y(["tl", "t", "tr"])
        bottom_y = median_y(["bl", "b", "br"])
        if top_y is None or bottom_y is None:
            top_y = median_y(["tl", "tr"])
            bottom_y = median_y(["bl", "br"])
        if top_y is None or bottom_y is None:
            return

        span_y = abs(float(bottom_y - top_y))
        gain = float(np.clip(0.45 / max(span_y, 1e-3), 1.2, 3.5))
        old_gain = float(max(self.vertical_gain, 1e-6))
        self.vertical_gain = gain

        # Rescale calibration Y around 0.5 so mapping stays consistent with new gain.
        scale = gain / old_gain
        if abs(scale - 1.0) < 1e-6:
            return
        for key, val in list(calib.items()):
            y_new = 0.5 + (val[1] - 0.5) * scale
            calib[key] = (float(val[0]), float(np.clip(y_new, 0.0, 1.0)))
        self._calib_center = calib.get("center")

    def set_full_calibration(self, calib: Dict[str, Gaze], pad: float = 0.03) -> bool:
        """Set full calibration data and compute normalized ranges."""
        self.axis_flip_x, self.axis_flip_y = self._infer_axis_flips(calib)
        corrected = {
            key: self.apply_axis_flip((float(val[0]), float(val[1])))
            for key, val in calib.items()
        }
        self._calib = corrected
        self._calib_center = corrected.get("center")
        # Auto-scale vertical gain based on calibration span (prevents Y compression).
        self._auto_scale_vertical_gain(self._calib)
        self._map_x, self._map_y = self._compute_piecewise_maps(self._calib)
        self._calib_range = self._compute_calibration_range(self._calib, pad=pad)
        self._edge_counter_x = 0
        self._edge_counter_y = 0
        self._stable_lock = False
        self._stable_gaze = None
        self._calib_active = False
        self._calib_index = 0
        self._calib_samples = []
        self._gaze_smooth = None
        self._gaze_window.clear()
        self._last_gaze_time = None
        return self._calib_range is not None

    def has_full_calibration(self) -> bool:
        return self._calib_range is not None

    def save_calibration(self, path: str) -> bool:
        """Save calibration data to disk."""
        if self._calib_range is not None:
            print("Vertical span:", self._calib_range[3] - self._calib_range[2])
        if self._calib_range is None and not self._calib:
            return False
        ml_model_file = self._ml_meta.get("model_file")
        ml_ready_for_save = bool(
            self._ml_meta.get("ready", False)
            and self.is_ml_ready()
            and isinstance(ml_model_file, str)
            and ml_model_file
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calib": {k: [v[0], v[1]] for k, v in self._calib.items()},
            "axis_flip": {"x": self.axis_flip_x, "y": self.axis_flip_y},
            "map_x": self._map_x,
            "map_y": self._map_y,
            "quality": self._calib_quality,
            "openness_comp": {"ref": self._open_ref, "beta_y": self._open_beta_y},
            "filter": {
                "min_cutoff": self._one_euro.fx.min_cutoff,
                "beta": self._one_euro.fx.beta,
                "d_cutoff": self._one_euro.fx.d_cutoff,
            },
            "range": None,
            "ml": {
                "ready": ml_ready_for_save,
                "user_id": self._ml_meta.get("user_id"),
                "model_file": ml_model_file,
                "feature_order": self._ml_meta.get("feature_order", list(ML_FEATURE_ORDER)),
                "alpha": self._ml_meta.get("alpha"),
                "version": self._ml_meta.get("version", ML_MODEL_VERSION),
            },
        }

        if self._calib_range is not None:
            gx_min, gx_max, gy_min, gy_max = self._calib_range
            data["range"] = {
                "gx_min": gx_min,
                "gx_max": gx_max,
                "gy_min": gy_min,
                "gy_max": gy_max,
            }
            print(f"gx_min: {gx_min}, gx_max: {gx_max}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True

    def load_calibration(self, path: str, pad: float = 0.03) -> bool:
        """Load calibration data from disk."""
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        axis_flip = data.get("axis_flip")
        if isinstance(axis_flip, dict):
            self.axis_flip_x = bool(axis_flip.get("x", False))
            self.axis_flip_y = bool(axis_flip.get("y", False))
        else:
            self.axis_flip_x = False
            self.axis_flip_y = False

        quality = data.get("quality")
        if isinstance(quality, dict):
            self._calib_quality = quality

        openness_comp = data.get("openness_comp")
        if isinstance(openness_comp, dict):
            ref = openness_comp.get("ref")
            beta_y = openness_comp.get("beta_y")
            if isinstance(ref, (int, float)):
                self._open_ref = float(ref)
            if isinstance(beta_y, (int, float)):
                self._open_beta_y = float(beta_y)

        filt = data.get("filter")
        if isinstance(filt, dict):
            try:
                min_cutoff = float(filt.get("min_cutoff", self._one_euro.fx.min_cutoff))
                beta = float(filt.get("beta", self._one_euro.fx.beta))
                d_cutoff = float(filt.get("d_cutoff", self._one_euro.fx.d_cutoff))
                self._one_euro = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            except (TypeError, ValueError):
                pass

        calib = data.get("calib")
        if isinstance(calib, dict) and calib:
            parsed = {k: (float(v[0]), float(v[1])) for k, v in calib.items()}
            if "axis_flip" in data:
                self._calib = parsed
                self._calib_center = parsed.get("center")
                map_x = data.get("map_x")
                map_y = data.get("map_y")
                if isinstance(map_x, dict) and isinstance(map_y, dict):
                    try:
                        self._map_x = {
                            "L": float(map_x["L"]),
                            "C": float(map_x["C"]),
                            "R": float(map_x["R"]),
                        }
                        self._map_y = {
                            "T": float(map_y["T"]),
                            "C": float(map_y["C"]),
                            "B": float(map_y["B"]),
                        }
                        spanL = self._map_x["C"] - self._map_x["L"]
                        spanR = self._map_x["R"] - self._map_x["C"]
                        spanT = self._map_y["C"] - self._map_y["T"]
                        spanB = self._map_y["B"] - self._map_y["C"]
                        self._map_spans = {"spanL": spanL, "spanR": spanR, "spanT": spanT, "spanB": spanB}
                        self._piecewise_ok = min(spanL, spanR, spanT, spanB) >= self._min_span
                    except (KeyError, TypeError, ValueError):
                        self._map_x, self._map_y = self._compute_piecewise_maps(parsed)
                else:
                    self._map_x, self._map_y = self._compute_piecewise_maps(parsed)
                # Auto-scale vertical gain from stored calibration points.
                self._auto_scale_vertical_gain(self._calib)
                self._calib_range = self._compute_calibration_range(self._calib, pad=pad)
                self._calib_active = False
                self._calib_index = 0
                self._calib_samples = []
                self._gaze_smooth = None
                self._gaze_window.clear()
                self._last_gaze_time = None
                ok = self._calib_range is not None
                if ok:
                    self._load_ml_from_json(data, path)
                return ok
            ok = self.set_full_calibration(parsed, pad=pad)
            if ok:
                self._load_ml_from_json(data, path)
            return ok

        rng = data.get("range")
        if isinstance(rng, dict):
            try:
                self._calib_range = (
                    float(rng["gx_min"]),
                    float(rng["gx_max"]),
                    float(rng["gy_min"]),
                    float(rng["gy_max"]),
                )
            except (KeyError, TypeError, ValueError):
                return False
            self._calib = {}
            self._calib_center = None
            self._map_x = None
            self._map_y = None
            self._piecewise_ok = False
            self._calib_active = False
            self._calib_index = 0
            self._calib_samples = []
            self._gaze_smooth = None
            self._gaze_window.clear()
            self._last_gaze_time = None
            self._load_ml_from_json(data, path)
            return True

        return False

    def map_gaze(self, gaze_raw: Optional[Gaze]) -> Optional[Gaze]:
        """Map raw gaze to normalized screen coordinates using calibration."""
        if gaze_raw is None:
            return None

        if False and self._blink_freeze and self._open_ref is not None and self._last_openness is not None:
            open_now = float(self._last_openness)
            if open_now < max(self._open_min_abs, self._open_ref * self._blink_ratio):
                if self._last_good_gaze is not None:
                    return self._last_good_gaze

        gx, gy = gaze_raw
        # Openness is kept for diagnostics only; it does not shift gaze.
        gx, gy = self.apply_axis_flip((gx, gy))

        if self.is_ml_ready():
            features = self._build_ml_features((gx, gy), self._last_head_pose_for_ml)
            if features is not None:
                pred = self.predict_ml_gaze(features)
                if pred is not None:
                    self._last_good_gaze = pred
                    return pred

        if self._calib_range is not None:
            if self._piecewise_ok and self._map_x is not None and self._map_y is not None:
                gx01 = self._map_piecewise(gx, self._map_x["L"], self._map_x["C"], self._map_x["R"])
                gy01 = self._map_piecewise(gy, self._map_y["T"], self._map_y["C"], self._map_y["B"])
                mapped = self._post_map_adjust((gx01, gy01), (gx, gy))
                self._last_good_gaze = mapped
                return mapped

            gx_min, gx_max, gy_min, gy_max = self._calib_range
            if gx_max - gx_min > 1e-6 and gy_max - gy_min > 1e-6:
                # Linear, reversible mapping for calibration.
                gx01 = (gx - gx_min) / (gx_max - gx_min)
                gy01 = (gy - gy_min) / (gy_max - gy_min)
                mapped = self._post_map_adjust((gx01, gy01), (gx, gy))
                self._last_good_gaze = mapped
                return mapped

        center = self._calib_center if self._calib_center is not None else (0.5, 0.5)
        gx01 = 0.5 + (gx - center[0]) * self.gain_x
        gy01 = 0.5 + (gy - center[1]) * self.gain_y
        gx01 = self._apply_deadzone(gx01)
        gy01 = self._apply_deadzone(gy01)
        gx01 = self._apply_gamma(gx01)
        gy01 = self._apply_gamma(gy01)
        mapped = self._post_map_adjust((self._clamp01(gx01), self._clamp01(gy01)), (gx, gy))
        self._last_good_gaze = mapped
        return mapped

    def reset_calibration(self) -> None:
        """Clear all calibration data."""
        self._calib_center = None
        self._calib = {}
        self._calib_range = None
        self.axis_flip_x = False
        self.axis_flip_y = False
        self.vertical_gain = self._vertical_gain_base
        self._map_x = None
        self._map_y = None
        self._map_spans = None
        self._piecewise_ok = False
        self._edge_counter_x = 0
        self._edge_counter_y = 0
        self._stable_lock = False
        self._stable_gaze = None
        self._calib_active = False
        self._calib_index = 0
        self._calib_samples = []
        self._gaze_smooth = None
        self._gaze_window.clear()
        self._last_gaze_time = None
        self._calib_quality = {}
        self._clear_ml_calibration()

    def toggle_mirror(self) -> bool:
        """Toggle mirror view and return the new state."""
        self.mirror_view = not self.mirror_view
        self._clear_ml_calibration()
        if self._calib_range is not None or self._calib_active:
            self._calib = {}
            self._calib_range = None
            self._calib_active = False
            self._calib_index = 0
            self._calib_samples = []
            self._gaze_smooth = None
        return self.mirror_view

    def adjust_gain(self, delta: float) -> None:
        """Adjust gain and clamp to a safe range."""
        self.gain_x = float(np.clip(self.gain_x + delta, 0.5, 8.0))
        self.gain_y = float(np.clip(self.gain_y + delta, 0.5, 8.0))
        self._gaze_smooth = None

    def adjust_sensitivity(self, delta: float) -> None:
        """Backwards-compatible wrapper for Stage 1.5."""
        self.adjust_gain(delta)

    def adjust_ema_alpha(self, delta: float) -> None:
        """Adjust EMA alpha to balance smoothness and responsiveness."""
        self.ema_alpha = float(np.clip(self.ema_alpha + delta, 0.05, 0.95))
        self._gaze_smooth = None

    def current_calibration_target(self) -> Optional[str]:
        if not self._calib_active:
            return None
        if self._calib_index >= len(self.CALIB_TARGETS):
            return None
        return self.CALIB_TARGETS[self._calib_index]

    def _calib_status_text(self) -> str:
        if self._calib_active:
            target = self.current_calibration_target() or "done"
            count = len(self._calib_samples)
            return f"CAL {target} {count}/{self._calib_samples_needed}"
        if self._calib_range is not None:
            return "FULL(5-pt)"
        if self._calib_center is not None:
            return "CENTER"
        return "NONE"

    def _compute_calibration_range(
        self,
        calib: Dict[str, Gaze],
        pad: float = 0.03,
    ) -> Optional[Tuple[float, float, float, float]]:
        needed = {"tl", "tr", "br", "bl"}
        if not needed.issubset(calib.keys()):
            return None

        center = calib.get("center")
        if center is None:
            return None

        gx_min = min(calib["tl"][0], calib["bl"][0], center[0])
        gx_max = max(calib["tr"][0], calib["br"][0], center[0])
        gy_min = min(calib["tl"][1], calib["tr"][1])
        gy_max = max(calib["bl"][1], calib["br"][1])

        pad = float(np.clip(pad, 0.0, 0.2))
        gx_min = self._clamp01(gx_min - pad)
        gx_max = self._clamp01(gx_max + pad)
        gy_min = self._clamp01(gy_min - pad)
        gy_max = self._clamp01(gy_max + pad)

        if gx_max - gx_min < 1e-6 or gy_max - gy_min < 1e-6:
            return None

        return gx_min, gx_max, gy_min, gy_max

    def _compute_piecewise_maps(
        self,
        calib: Dict[str, Gaze],
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        self._map_spans = None
        self._piecewise_ok = False
        def median_for(keys: List[str], idx: int) -> Optional[float]:
            vals = [calib[k][idx] for k in keys if k in calib]
            if not vals:
                return None
            return float(np.median(vals))

        xL = median_for(["tl", "l", "bl"], 0)
        xC = median_for(["t", "center", "b"], 0)
        xR = median_for(["tr", "r", "br"], 0)
        yT = median_for(["tl", "t", "tr"], 1)
        yC = median_for(["l", "center", "r"], 1)
        yB = median_for(["bl", "b", "br"], 1)

        if xL is None or xC is None or xR is None or yT is None or yC is None or yB is None:
            xL = median_for(["tl", "bl"], 0)
            xC = median_for(["center"], 0)
            xR = median_for(["tr", "br"], 0)
            yT = median_for(["tl", "tr"], 1)
            yC = median_for(["center"], 1)
            yB = median_for(["bl", "br"], 1)

        if xL is None or xC is None or xR is None or yT is None or yC is None or yB is None:
            return None, None

        spanL = xC - xL
        spanR = xR - xC
        spanT = yC - yT
        spanB = yB - yC
        self._map_spans = {"spanL": spanL, "spanR": spanR, "spanT": spanT, "spanB": spanB}
        if min(spanL, spanR, spanT, spanB) < self._min_span:
            self._piecewise_ok = False
            return None, None

        self._piecewise_ok = True
        map_x = {"L": xL, "C": xC, "R": xR}
        map_y = {"T": yT, "C": yC, "B": yB}
        return map_x, map_y

    def _map_piecewise(self, value: float, left: float, center: float, right: float) -> float:
        left_p = left - self._soft_pad
        right_p = right + self._soft_pad
        if value <= center:
            denom = max(center - left_p, 1e-6)
            u = 0.5 * (value - left_p) / denom
        else:
            denom = max(right_p - center, 1e-6)
            u = 0.5 + 0.5 * (value - center) / denom

        u = 0.5 + 0.5 * math.tanh((u - 0.5) / self._soft_k)
        return u

    def _infer_axis_flips(self, calib: Dict[str, Gaze]) -> Tuple[bool, bool]:
        def mean_for(keys: List[str], idx: int) -> Optional[float]:
            vals = [calib[k][idx] for k in keys if k in calib]
            if not vals:
                return None
            return float(np.mean(vals))

        left_x = mean_for(["tl", "l", "bl"], 0)
        right_x = mean_for(["tr", "r", "br"], 0)
        top_y = mean_for(["tl", "t", "tr"], 1)
        bottom_y = mean_for(["bl", "b", "br"], 1)

        if left_x is None or right_x is None:
            left_x = mean_for(["tl", "bl"], 0)
            right_x = mean_for(["tr", "br"], 0)
        if top_y is None or bottom_y is None:
            top_y = mean_for(["tl", "tr"], 1)
            bottom_y = mean_for(["bl", "br"], 1)

        if left_x is None or right_x is None or top_y is None or bottom_y is None:
            return False, False

        flip_x = left_x > right_x
        flip_y = top_y > bottom_y
        return flip_x, flip_y

    def apply_axis_flip(self, gaze: Gaze) -> Gaze:
        gx, gy = gaze
        if self.axis_flip_x:
            gx = 1.0 - gx
        if self.axis_flip_y:
            gy = 1.0 - gy
        return gx, gy

    def _iris_center_and_radius(self, iris_pts: List[Point]) -> Tuple[Point, float]:
        iris_arr = np.array(iris_pts, dtype=np.float32)
        center = iris_arr.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(iris_arr - center, axis=1)))
        return (float(center[0]), float(center[1])), radius

    def _compute_gaze(
        self,
        points_norm: Dict[str, object],
        left_radius: Optional[float],
        right_radius: Optional[float],
    ) -> Tuple[Optional[Gaze], Optional[float], Optional[float], Optional[Dict[str, float]]]:
        try:
            left_outer = points_norm["left_outer"]
            left_inner = points_norm["left_inner"]
            left_upper = points_norm["left_upper"]
            left_lower = points_norm["left_lower"]
            left_iris = points_norm["left_iris_center"]
            right_outer = points_norm["right_outer"]
            right_inner = points_norm["right_inner"]
            right_upper = points_norm["right_upper"]
            right_lower = points_norm["right_lower"]
            right_iris = points_norm["right_iris_center"]
        except KeyError:
            return None, None, None, None

        left_upper_pts = points_norm.get("left_upper_pts") or [left_upper]
        left_lower_pts = points_norm.get("left_lower_pts") or [left_lower]
        right_upper_pts = points_norm.get("right_upper_pts") or [right_upper]
        right_lower_pts = points_norm.get("right_lower_pts") or [right_lower]

        left = self._gaze_for_eye(left_outer, left_inner, left_upper_pts, left_lower_pts, left_iris)
        right = self._gaze_for_eye(right_outer, right_inner, right_upper_pts, right_lower_pts, right_iris)
        if left is None or right is None:
            return None, None, None, None

        gx = 0.5 * (left["gx"] + right["gx"])
        gy = 0.5 * (left["gy"] + right["gy"])
        openness = 0.5 * (left["openness"] + right["openness"])

        if self.invert_x:
            gx = 1.0 - gx
        if self.invert_y:
            gy = 1.0 - gy

        gaze_raw = (float(np.clip(gx, 0.0, 1.0)), float(np.clip(gy, 0.0, 1.0)))

        iris_radius = None
        if left_radius is not None and right_radius is not None:
            iris_radius = (left_radius + right_radius) * 0.5

        features = {
            "gx": float(gx),
            "gy": float(gy),
            "lx": float(left["gx"]),
            "ly": float(left["gy"]),
            "rx": float(right["gx"]),
            "ry": float(right["gy"]),
            "open": float(openness),
            "lopen": float(left["openness"]),
            "ropen": float(right["openness"]),
            "eye_w": float(0.5 * (left["eye_width"] + right["eye_width"])),
            "lv": float(left["v_norm"]),
            "rv": float(right["v_norm"]),
            "llid": float(left["lid_norm"]),
            "rlid": float(right["lid_norm"]),
        }

        return gaze_raw, openness, iris_radius, features

    def _gaze_for_eye(
        self,
        outer: Point,
        inner: Point,
        upper_pts: List[Point],
        lower_pts: List[Point],
        iris: Point,
    ) -> Optional[Dict[str, float]]:
        if not upper_pts or not lower_pts:
            return None

        upper_y = float(np.median([pt[1] for pt in upper_pts]))
        lower_y = float(np.median([pt[1] for pt in lower_pts]))

        # Prefer horizontal x-ratio for better left/right sensitivity.
        x_min = min(outer[0], inner[0])
        x_max = max(outer[0], inner[0])
        x_span = x_max - x_min
        eye_vec = (inner[0] - outer[0], inner[1] - outer[1])
        denom = eye_vec[0] * eye_vec[0] + eye_vec[1] * eye_vec[1]
        eye_width = float(math.hypot(eye_vec[0], eye_vec[1]))
        if eye_width < 1e-6 or denom < 1e-6:
            return None

        if x_span > 1e-6:
            gx = float(np.clip((iris[0] - x_min) / x_span, 0.0, 1.0))
        else:
            t = ((iris[0] - outer[0]) * eye_vec[0] + (iris[1] - outer[1]) * eye_vec[1]) / denom
            gx = float(np.clip(t, 0.0, 1.0))

        # Vertical gaze from iris projection onto eye-axis normal (blink-invariant).
        eye_center = ((outer[0] + inner[0]) * 0.5, (outer[1] + inner[1]) * 0.5)
        nx = -eye_vec[1] / eye_width
        ny = eye_vec[0] / eye_width
# --- Iris-based vertical ---
        v = (iris[0] - eye_center[0]) * nx + (iris[1] - eye_center[1]) * ny
        v_norm = v / max(eye_width, 1e-6)

        # --- Eyelid-based vertical (stronger signal) ---
        lid_mid = 0.5 * (upper_y + lower_y)
        lid_span = max(abs(lower_y - upper_y), 1e-6)
        lid_norm = (lid_mid - eye_center[1]) / lid_span

        # --- Fuse signals ---
        # Iris = precise but weak
        # Lid = strong but noisy
        eye_height = max(abs(lower_y - upper_y), 1e-6)
        openness = float(eye_height / max(eye_width, 1e-6))
        w_lid = float(np.clip((openness - 0.10) / (0.28 - 0.10), 0.0, 1.0))
        w_lid = w_lid * w_lid
        gy_raw = (1.0 - w_lid) * v_norm + w_lid * lid_norm

        gy = float(np.clip(0.5 + gy_raw * self.vertical_gain, 0.0, 1.0))

        return {
            "gx": gx,
            "gy": gy,
            "openness": openness,
            "eye_width": eye_width,
            "v_norm": float(v_norm),
            "lid_norm": float(lid_norm),
            "upper_y": float(upper_y),
            "lower_y": float(lower_y),
        }

    def _smooth_gaze(self, gaze: Optional[Gaze]) -> Optional[Gaze]:
        now = time.time()
        if gaze is None:
            if self._gaze_smooth is None:
                return None
            if self._last_gaze_time is not None and now - self._last_gaze_time <= self._hold_ms / 1000.0:
                return self._gaze_smooth
            self._gaze_smooth = None
            self._gaze_window.clear()
            self._stable_lock = False
            self._stable_gaze = None
            return None

        self._last_gaze_time = now
        self._gaze_window.append(gaze)
        xs = [pt[0] for pt in self._gaze_window]
        ys = [pt[1] for pt in self._gaze_window]
        med = (float(np.median(xs)), float(np.median(ys)))

        gx, gy = self._one_euro.filter(med[0], med[1], now)
        # Reduce smoothing when nearly stationary for finer aiming.
        if self._one_euro.last_velocity < 0.012:
            gx = 0.7 * gx + 0.3 * med[0]
            gy = 0.7 * gy + 0.3 * med[1]
        self._last_alpha = self._one_euro.last_alpha
        self._last_cutoff = self._one_euro.last_cutoff
        self._last_velocity = self._one_euro.last_velocity
        self._gaze_smooth = (gx, gy)
        return self._gaze_smooth

    def _reset_gaze_recovery_state(self) -> None:
        self._invalid_since = None
        self._reacquire_count = 0
        self._reacquiring = False
        self._gaze_smooth = None
        self._gaze_window.clear()
        self._last_gaze_time = None
        self._stable_lock = False
        self._stable_gaze = None
        self._last_alpha = self.ema_alpha
        self._last_cutoff = 0.0
        self._last_velocity = 0.0
        self._one_euro = OneEuroFilter2D(
            min_cutoff=self._one_euro.fx.min_cutoff,
            beta=self._one_euro.fx.beta,
            d_cutoff=self._one_euro.fx.d_cutoff,
            min_cutoff_y=self._one_euro.fy.min_cutoff,
            beta_y=self._one_euro.fy.beta,
        )

    def _norm_to_px(self, pt: Point, w: int, h: int) -> Tuple[int, int]:
        return int(pt[0] * w), int(pt[1] * h)

    def _reset_pose_filters(self) -> None:
        self._pose_filt_yaw = OneEuroFilter(
            min_cutoff=POSE_SMOOTH_MIN_CUTOFF,
            beta=POSE_SMOOTH_BETA,
            d_cutoff=POSE_SMOOTH_D_CUTOFF,
        )
        self._pose_filt_pitch = OneEuroFilter(
            min_cutoff=POSE_SMOOTH_MIN_CUTOFF,
            beta=POSE_SMOOTH_BETA,
            d_cutoff=POSE_SMOOTH_D_CUTOFF,
        )
        self._pose_filt_roll = OneEuroFilter(
            min_cutoff=POSE_SMOOTH_MIN_CUTOFF,
            beta=POSE_SMOOTH_BETA,
            d_cutoff=POSE_SMOOTH_D_CUTOFF,
        )
        self._pose_filt_tz = OneEuroFilter(
            min_cutoff=POSE_SMOOTH_MIN_CUTOFF,
            beta=POSE_SMOOTH_BETA,
            d_cutoff=POSE_SMOOTH_D_CUTOFF,
        )
        self._pose_last_valid_ts = None
        self._pose_last_smoothed = None

    def _smooth_head_pose(self, raw_pose: Optional[Dict[str, float]], t: float) -> Optional[Dict[str, float]]:
        def hold_pose() -> Optional[Dict[str, float]]:
            if self._pose_last_valid_ts is None or self._pose_last_smoothed is None:
                return None
            if t - self._pose_last_valid_ts <= POSE_HOLD_MS / 1000.0:
                return self._pose_last_smoothed
            return None

        if raw_pose is None:
            return hold_pose()

        try:
            yaw = float(raw_pose["yaw"])
            pitch = float(raw_pose["pitch"])
            roll = float(raw_pose["roll"])
            tz = float(raw_pose["tz"])
        except (TypeError, ValueError, KeyError):
            return hold_pose()

        if not np.isfinite([yaw, pitch, roll, tz]).all():
            return hold_pose()

        yaw_s = self._pose_filt_yaw.filter(yaw, t)
        pitch_s = self._pose_filt_pitch.filter(pitch, t)
        roll_s = self._pose_filt_roll.filter(roll, t)
        tz_s = self._pose_filt_tz.filter(tz, t)

        smoothed = {
            "yaw": float(yaw_s),
            "pitch": float(pitch_s),
            "roll": float(roll_s),
            "tz": float(tz_s),
        }
        self._pose_last_valid_ts = t
        self._pose_last_smoothed = smoothed
        return smoothed

    def _estimate_head_pose(
        self,
        landmarks_norm: List[Point],
        w: int,
        h: int,
    ) -> Optional[Dict[str, float]]:
        if not isinstance(landmarks_norm, list):
            return None
        required = list(POSE_LANDMARKS.values())
        if not landmarks_norm or max(required) >= len(landmarks_norm):
            return None

        try:
            image_points_2d = np.array(
                [
                    (landmarks_norm[POSE_LANDMARKS["nose_tip"]][0] * w, landmarks_norm[POSE_LANDMARKS["nose_tip"]][1] * h),
                    (landmarks_norm[POSE_LANDMARKS["chin"]][0] * w, landmarks_norm[POSE_LANDMARKS["chin"]][1] * h),
                    (landmarks_norm[POSE_LANDMARKS["l_eye_outer"]][0] * w, landmarks_norm[POSE_LANDMARKS["l_eye_outer"]][1] * h),
                    (landmarks_norm[POSE_LANDMARKS["r_eye_outer"]][0] * w, landmarks_norm[POSE_LANDMARKS["r_eye_outer"]][1] * h),
                    (landmarks_norm[POSE_LANDMARKS["l_mouth"]][0] * w, landmarks_norm[POSE_LANDMARKS["l_mouth"]][1] * h),
                    (landmarks_norm[POSE_LANDMARKS["r_mouth"]][0] * w, landmarks_norm[POSE_LANDMARKS["r_mouth"]][1] * h),
                ],
                dtype=np.float32,
            )
        except (TypeError, ValueError, IndexError):
            return None

        fx = float(w)
        fy = float(w)
        cx = float(w) * 0.5
        cy = float(h) * 0.5
        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not success:
            return None

        try:
            rot_matrix, _ = cv2.Rodrigues(rvec)
        except cv2.error:
            return None
        if not np.isfinite(rot_matrix).all() or not np.isfinite(tvec).all():
            return None

        try:
            yaw_deg, pitch_deg, roll_deg = self._rotationMatrixToEulerAngles(rot_matrix)
        except (ValueError, OverflowError):
            return None
        if not np.isfinite([yaw_deg, pitch_deg, roll_deg]).all():
            return None

        raw_yaw_deg = float(yaw_deg)
        raw_pitch_deg = float(pitch_deg)
        raw_roll_deg = float(roll_deg)
        if self._pose_prev is not None:
            raw_delta_yaw = raw_yaw_deg - self._pose_prev[0]
            raw_delta_pitch = raw_pitch_deg - self._pose_prev[1]
            raw_delta_roll = raw_roll_deg - self._pose_prev[2]
            yaw_deg = self._unwrap_deg(raw_yaw_deg, self._pose_prev[0])
            pitch_deg = self._unwrap_deg(raw_pitch_deg, self._pose_prev[1])
            roll_deg = self._unwrap_deg(raw_roll_deg, self._pose_prev[2])
            if (
                self._pose_unwrap_debug
                and (
                    abs(raw_delta_yaw) > 200.0
                    or abs(raw_delta_pitch) > 200.0
                    or abs(raw_delta_roll) > 200.0
                )
            ):
                print(
                    "[pose-unwrap] raw ypr="
                    f"({raw_yaw_deg:.1f},{raw_pitch_deg:.1f},{raw_roll_deg:.1f}) "
                    f"prev=({self._pose_prev[0]:.1f},{self._pose_prev[1]:.1f},{self._pose_prev[2]:.1f}) "
                    f"unwrapped=({yaw_deg:.1f},{pitch_deg:.1f},{roll_deg:.1f})"
                )

        tvec_flat = tvec.reshape(-1)
        if tvec_flat.size < 3 or not np.isfinite(tvec_flat[2]):
            return None
        self._pose_prev = (float(yaw_deg), float(pitch_deg), float(roll_deg))

        return {
            "yaw": float(yaw_deg),
            "pitch": float(pitch_deg),
            "roll": float(roll_deg),
            "tz": float(tvec_flat[2]),
        }

    def _unwrap_deg(self, curr: float, prev: float) -> float:
        delta = curr - prev
        while delta > 180.0:
            curr -= 360.0
            delta = curr - prev
        while delta < -180.0:
            curr += 360.0
            delta = curr - prev
        return curr

    def _rotationMatrixToEulerAngles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")

        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

    def _format_gaze_text(self, label: str, gaze: Optional[Gaze]) -> str:
        if isinstance(gaze, tuple):
            return f"{label}=({gaze[0]:.2f},{gaze[1]:.2f})"
        return f"{label}=(None)"

    def _apply_deadzone(self, u: float) -> float:
        deadzone = float(np.clip(self.deadzone, 0.0, 0.45))
        d = u - 0.5
        if abs(d) < deadzone:
            return 0.5
        span = 0.5 - deadzone
        scaled = (abs(d) - deadzone) / max(span, 1e-6)
        return self._clamp01(0.5 + np.sign(d) * 0.5 * scaled)

    def _apply_gamma(self, u: float) -> float:
        g = float(max(0.01, self.gamma))
        u = self._clamp01(u)
        if u < 0.5:
            return 0.5 * (2.0 * u) ** g
        return 1.0 - 0.5 * (2.0 * (1.0 - u)) ** g

    def _mean_point(self, points: List[Point]) -> Point:
        arr = np.array(points, dtype=np.float32)
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1])

    def _post_map_adjust(self, mapped: Gaze, raw_corrected: Gaze) -> Gaze:
        gx01, gy01 = mapped
        # Soft saturation after mapping to avoid snapping while keeping corners reachable.
        #gx01 = self._apply_edge_resistance(gx01, axis="x")
        #kills corners or something
        #gy01 = self._apply_edge_resistance(gy01, axis="y")
        return self._clamp01(gx01), self._clamp01(gy01)

    def _apply_edge_resistance(self, u: float, axis: str) -> float:
        u = float(u)
        if axis == "x":
            pad = self._edge_pad_x
            overshoot = self._edge_overshoot_x
        else:
            pad = self._edge_pad_y
            overshoot = self._edge_overshoot_y

        pad = float(np.clip(pad, 0.0, 0.25))
        overshoot = float(np.clip(overshoot, 0.0, 0.2))
        if pad <= 1e-6:
            return self._clamp01(u)

        # Allow slight overshoot before smooth compression.
        u = float(np.clip(u, -overshoot, 1.0 + overshoot))

        def smoothstep(t: float) -> float:
            t = float(np.clip(t, 0.0, 1.0))
            return t * t * (3.0 - 2.0 * t)

        if u <= 0.0:
            if overshoot <= 1e-6:
                return 0.0
            t = (u + overshoot) / max(overshoot, 1e-6)
            return pad * smoothstep(t)
        if u >= 1.0:
            if overshoot <= 1e-6:
                return 1.0
            t = (u - 1.0) / max(overshoot, 1e-6)
            return 1.0 - pad + pad * smoothstep(t)
        if u < pad:
            t = u / pad
            return pad * smoothstep(t)
        if u > 1.0 - pad:
            t = (1.0 - u) / pad
            return 1.0 - pad * smoothstep(t)
        return u

    def _apply_deadband(self, gaze: Gaze) -> Gaze:
        if self._stable_gaze is None:
            self._stable_gaze = gaze
            self._stable_lock = False
            return gaze

        dist = float(math.hypot(gaze[0] - self._stable_gaze[0], gaze[1] - self._stable_gaze[1]))
        if self._stable_lock:
            if dist > self._deadband_exit:
                self._stable_lock = False
                self._stable_gaze = gaze
                return gaze
            return self._stable_gaze

        if dist < self._deadband_enter:
            self._stable_lock = True
            return self._stable_gaze

        self._stable_gaze = gaze
        return gaze

    def compute_median_mad(self, samples: List[Gaze]) -> Tuple[Gaze, Gaze]:
        arr = np.array(samples, dtype=np.float32)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        return (float(med[0]), float(med[1])), (float(mad[0]), float(mad[1]))

    def set_calibration_quality(self, quality: Dict[str, Dict[str, float]]) -> None:
        self._calib_quality = quality

    def _compute_eye_width_px(self, points_norm: Dict[str, object], w: int, h: int) -> Optional[float]:
        left_outer = points_norm.get("left_outer")
        left_inner = points_norm.get("left_inner")
        right_outer = points_norm.get("right_outer")
        right_inner = points_norm.get("right_inner")
        if not all(isinstance(pt, tuple) for pt in (left_outer, left_inner, right_outer, right_inner)):
            return None

        lo = self._norm_to_px(left_outer, w, h)
        li = self._norm_to_px(left_inner, w, h)
        ro = self._norm_to_px(right_outer, w, h)
        ri = self._norm_to_px(right_inner, w, h)

        left_w = float(np.linalg.norm(np.array(lo) - np.array(li)))
        right_w = float(np.linalg.norm(np.array(ro) - np.array(ri)))
        return (left_w + right_w) * 0.5

    def _face_confidence(self, iris_radius: Optional[float]) -> float:
        if iris_radius is None:
            return 0.0
        if 0.002 <= iris_radius <= 0.03:
            return 1.0
        return 0.0

    def _reject_frame(
        self,
        iris_radius: Optional[float],
        eye_openness: Optional[float],
        eye_width_px: Optional[float],
    ) -> bool:
        if iris_radius is None or iris_radius < 0.002 or iris_radius > 0.03:
            return True
        if eye_openness is None or eye_openness < 0.10 or eye_openness > 0.65:
            return True
        if eye_width_px is None or eye_width_px < 25.0:
            return True
        return False

    def _draw_eye(
        self,
        frame_bgr: np.ndarray,
        points_px: Dict[str, Tuple[int, int]],
        outer_key: str,
        inner_key: str,
        upper_key: str,
        lower_key: str,
        iris_key: str,
    ) -> None:
        outer = points_px.get(outer_key)
        inner = points_px.get(inner_key)
        upper = points_px.get(upper_key)
        lower = points_px.get(lower_key)
        iris = points_px.get(iris_key)
        if outer is None or inner is None or upper is None or lower is None or iris is None:
            return

        cv2.line(frame_bgr, outer, inner, (0, 255, 255), 1)
        cv2.line(frame_bgr, upper, lower, (0, 255, 255), 1)

        cv2.circle(frame_bgr, outer, 2, (0, 255, 0), -1)
        cv2.circle(frame_bgr, inner, 2, (0, 255, 0), -1)
        cv2.circle(frame_bgr, upper, 2, (0, 255, 0), -1)
        cv2.circle(frame_bgr, lower, 2, (0, 255, 0), -1)
        cv2.circle(frame_bgr, iris, 3, (0, 0, 255), -1)

    def _draw_crosshair(
        self,
        frame_bgr: np.ndarray,
        center: Tuple[int, int],
        size: int = 12,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> None:
        cx, cy = center
        cv2.line(frame_bgr, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(frame_bgr, (cx, cy - size), (cx, cy + size), color, thickness)

    def _draw_gaze_panel(
        self,
        frame_bgr: np.ndarray,
        gaze: Optional[Gaze],
        calib_target: Optional[str],
    ) -> None:
        h, w = frame_bgr.shape[:2]
        margin = 10
        panel_w = int(w * 0.25)
        panel_h = int(h * 0.6)
        panel_size = max(140, min(panel_w, panel_h))

        x0 = w - panel_size - margin
        y0 = margin
        x1 = w - margin
        y1 = y0 + panel_size

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (200, 200, 200), 2)
        cv2.putText(
            frame_bgr,
            "Virtual Screen",
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        if calib_target in self.PANEL_TARGETS:
            tx_norm, ty_norm = self.PANEL_TARGETS[calib_target]
            tx = int(x0 + tx_norm * (panel_size - 1))
            ty = int(y0 + ty_norm * (panel_size - 1))
            self._draw_crosshair(frame_bgr, (tx, ty))

        if gaze is None:
            return

        gx = float(np.clip(gaze[0], 0.0, 1.0))
        gy = float(np.clip(gaze[1], 0.0, 1.0))
        px = int(x0 + gx * (panel_size - 1))
        py = int(y0 + gy * (panel_size - 1))
        cv2.circle(frame_bgr, (px, py), 9, (255, 120, 0), -1)

    def _clamp01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))
