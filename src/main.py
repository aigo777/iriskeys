from __future__ import annotations
import argparse
a=0
import os
import re
import time
from collections import deque
import math
from typing import TypedDict
import cv2
import numpy as np
from gaze_tracker import GazeTracker
from demo_ui import DemoUI, local_to_desktop_px, normalized_to_local_px


class KeyboardKey(TypedDict):
    id: str
    label: str
    kind: str
    value: str
    rect: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IrisKeys gaze demo")
    parser.add_argument("--os-click", choices=("off", "on"), default="off")
    parser.add_argument("--assist", choices=("off", "on"), default="off")
    parser.add_argument("--drift", choices=("off", "on"), default="off")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os_click_enabled = args.os_click == "on"
    assist_enabled = args.assist == "on"
    drift_enabled = args.drift == "on"

    print("IrisKeys - Stage 3.0")
    print(
        "Controls: q quit | esc cancel armed | space confirm | p pause demo | v toggle demo/keyboard | m toggle mouse | b toggle blink-click | s screenshot | k calibrate | r reset | l load calibration | t test | [/] edge_gain | 9/0 y_edge_gain | i/k y_scale | o/l y_offset | y flip | ,/. spring_k"
    )

    tracker = GazeTracker()
    cap = cv2.VideoCapture(tracker.camera_index)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    debug_win = "IrisKeys Debug"
    pointer_win = "IrisKeys Pointer"
    cv2.namedWindow(pointer_win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(pointer_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass
    cv2.namedWindow(debug_win, cv2.WINDOW_NORMAL)

    calib_dir = os.path.join(os.getcwd(), "calibration")
    calib_path = os.path.join(calib_dir, "calibration_data.json")
    os.makedirs(calib_dir, exist_ok=True)

    try:
        user_id = os.getlogin()
    except OSError:
        user_id = os.environ.get("USERNAME", "default")
    user_id = user_id or "default"
    user_id_safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id) or "default"
    ml_model_filename = f"ml_model_{user_id_safe}.npz"
    ml_model_path = os.path.join(calib_dir, ml_model_filename)

    if tracker.load_calibration(calib_path):
        print(f"Loaded calibration from {calib_path}")
        if not tracker.is_ml_ready() and os.path.exists(ml_model_path):
            tracker.load_ml_model(ml_model_path, user_id=user_id)
        print(f"ML mapper: {'ON' if tracker.is_ml_ready() else 'OFF'}")
    elif os.path.exists(ml_model_path):
        if tracker.load_ml_model(ml_model_path, user_id=user_id):
            print(f"Loaded standalone ML model from {ml_model_path}")

    calib_targets = [
        ("tl", "TOP-LEFT", (0.08, 0.08)),
        ("t", "TOP", (0.50, 0.08)),
        ("tr", "TOP-RIGHT", (0.92, 0.08)),
        ("l", "LEFT", (0.08, 0.50)),
        ("center", "CENTER", (0.50, 0.50)),
        ("r", "RIGHT", (0.92, 0.50)),
        ("bl", "BOTTOM-LEFT", (0.08, 0.92)),
        ("b", "BOTTOM", (0.50, 0.92)),
        ("br", "BOTTOM-RIGHT", (0.92, 0.92)),
    ]
    calib_active = False
    calib_phase = "idle"
    calib_index = 0
    pursuit_duration_s = 10.0
    pursuit_sample_interval_s = 0.05
    static_ml_sample_interval_s = 0.20
    pursuit_start_time = 0.0
    last_pursuit_sample_time = 0.0
    last_static_ml_sample_time = 0.0
    pursuit_target_xy: tuple[float, float] | None = None
    pursuit_points = [
        (0.08, 0.08),
        (0.50, 0.08),
        (0.92, 0.08),
        (0.92, 0.50),
        (0.92, 0.92),
        (0.50, 0.92),
        (0.08, 0.92),
        (0.08, 0.50),
        (0.50, 0.50),
        (0.08, 0.08),
    ]
    calib_samples: list[tuple[float, float, float]] = []
    calib_data: dict[str, tuple[float, float]] = {}
    calib_quality: dict[str, dict[str, float]] = {}
    calib_open: dict[str, float] = {}
    train_X: list[list[float]] = []
    train_Y: list[list[float]] = []
    center_open_list: list[float] = []
    center_gy_list: list[float] = []
    calib_pose_ref: tuple[float, float, float, float] | None = None
    calib_settle_s = 0.8
    calib_samples_needed = 75
    calib_phase_start = 0.0
    calib_pad = 0.03
    calib_mad_thresh = 0.015
    calib_pose_yaw_thresh = 3.0
    calib_pose_pitch_thresh = 3.0
    calib_pose_roll_thresh = 4.0
    test_active = False
    test_start = 0.0
    test_duration = 1.0
    show_pose_indices = False

    screen_w = None
    screen_h = None
    vx = 0
    vy = 0
    vw = None
    vh = None
    cursor_backend = "ctypes"
    user32 = None
    wintypes = None

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(0))
        screen_h = int(user32.GetSystemMetrics(1))
        vx = int(user32.GetSystemMetrics(76))
        vy = int(user32.GetSystemMetrics(77))
        vw = int(user32.GetSystemMetrics(78))
        vh = int(user32.GetSystemMetrics(79))
    except Exception:
        screen_w = None
        screen_h = None
        vx = 0
        vy = 0
        vw = None
        vh = None

    if vw is None or vh is None or vw <= 0 or vh <= 0:
        vw = screen_w
        vh = screen_h

    APP_MODE_DEMO = "demo"
    APP_MODE_KEYBOARD = "keyboard"
    app_mode = APP_MODE_DEMO
    keyboard_page = "letters"
    keyboard_text = ""
    keyboard_focus_key: str | None = None
    keyboard_focus_start: float | None = None
    keyboard_dwell_progress = 0.0
    keyboard_armed_key: str | None = None
    keyboard_candidate_key: str | None = None
    keyboard_candidate_start: float | None = None
    keyboard_pages: dict[str, list[KeyboardKey]] = {"letters": []}
    keyboard_layout_size: tuple[int, int] | None = None

    demo_ui: DemoUI | None = None
    demo_drift_x = 0.0
    demo_drift_y = 0.0
    demo_drift_max = 0.006

    mouse_enabled = False
    mouse_paused = False
    lost_frames = 0
    last_face_time = time.time()
    last_cursor_time = time.time()
    sx = None
    sy = None
    vx_c = 0.0
    vy_c = 0.0
    edge_gain = 0.25
    # Cursor dynamics (tune for smoothness)
    spring_k = 55.0
    spring_d = 14.0
    max_speed = 2200.0
    max_accel = 9000.0
    drift_x = 0.0
    drift_y = 0.0
    drift_max = 0.04
    y_scale = 1.35
    y_offset = 0.0
    y_flip = False
    y_edge_gain = 0.18
    blink_enabled = True
    blink_closed = False
    blink_down_ts = 0.0
    last_click_ts = 0.0
    open_ref: float | None = None
    open_ref_samples = deque(maxlen=30)

    BLINK_CLOSE_RATIO = 0.55
    BLINK_OPEN_RATIO = 0.80
    BLINK_MIN_HOLD_S = 0.10
    CLICK_COOLDOWN_S = 0.65
    BASELINE_UPDATE_VEL = 0.015
    BASELINE_MIN = 0.10
    BASELINE_MAX = 0.65
    KEYBOARD_DWELL_S = 0.8
    KEYBOARD_FOCUS_SWITCH_S = 0.12

    print("Cursor backend: ctypes")

    def build_keyboard_letters_page(screen_w: int, screen_h: int) -> list[KeyboardKey]:
        sw = int(max(1, screen_w))
        sh = int(max(1, screen_h))
        layout_w = max(1, int(round(sw * 0.84)))
        layout_h = max(1, int(round(sh * 0.46)))
        layout_x = max(0, int(round((sw - layout_w) * 0.5)))
        layout_y = max(0, int(round(sh * 0.45)))
        max_layout_y = max(0, sh - layout_h)
        if layout_y > max_layout_y:
            layout_y = max_layout_y

        gap_x = max(16, int(0.016 * sw))
        gap_y = max(18, int(0.024 * sh))
        row_h = max(1, int((layout_h - (3 * gap_y)) / 4))

        def make_key(key_id: str, label: str, kind: str, value: str, rect: tuple[int, int, int, int]) -> KeyboardKey:
            left, top, right, bottom = rect
            return {
                "id": key_id,
                "label": label,
                "kind": kind,
                "value": value,
                "rect": (int(left), int(top), int(right), int(bottom)),
            }

        keys: list[KeyboardKey] = []
        row_defs = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M"],
        ]

        for row_index, row_labels in enumerate(row_defs):
            count = len(row_labels)
            row_gap_total = gap_x * max(0, count - 1)
            key_w = max(1, int((layout_w - row_gap_total) / count))
            row_w = key_w * count + row_gap_total
            row_x = layout_x + max(0, int((layout_w - row_w) / 2))
            top = layout_y + row_index * (row_h + gap_y)
            bottom = top + row_h
            for col_index, label in enumerate(row_labels):
                left = row_x + col_index * (key_w + gap_x)
                right = left + key_w
                value = label.lower()
                keys.append(make_key(f"key_{value}", label, "char", value, (left, top, right, bottom)))

        bottom_top = layout_y + 3 * (row_h + gap_y)
        bottom_bottom = bottom_top + row_h
        bottom_defs = [
            ("key_space", "Space", "action", "space", 4.0),
            ("key_backspace", "Backspace", "action", "backspace", 2.0),
            ("key_enter", "Enter", "action", "enter", 2.0),
            ("key_demo", "Demo", "action", "demo", 1.5),
        ]
        total_weight = sum(item[4] for item in bottom_defs)
        bottom_gap_total = gap_x * max(0, len(bottom_defs) - 1)
        unit_w = max(1.0, float(layout_w - bottom_gap_total) / float(total_weight))
        widths = [max(1, int(round(unit_w * item[4]))) for item in bottom_defs]
        width_error = layout_w - (sum(widths) + bottom_gap_total)
        widths[0] += width_error
        bottom_row_w = sum(widths) + bottom_gap_total
        bottom_x = layout_x + max(0, int((layout_w - bottom_row_w) / 2))
        cursor_x = bottom_x
        for width, (key_id, label, kind, value, _) in zip(widths, bottom_defs):
            left = cursor_x
            right = left + width
            keys.append(make_key(key_id, label, kind, value, (left, bottom_top, right, bottom_bottom)))
            cursor_x = right + gap_x

        return keys

    def refresh_keyboard_layout() -> None:
        nonlocal keyboard_pages, keyboard_layout_size
        if screen_w is None or screen_h is None:
            return
        layout_size = (int(screen_w), int(screen_h))
        if keyboard_layout_size == layout_size:
            return
        keyboard_pages["letters"] = build_keyboard_letters_page(layout_size[0], layout_size[1])
        keyboard_layout_size = layout_size

    refresh_keyboard_layout()

    def clamp01(val: float) -> float:
        return float(np.clip(val, 0.0, 1.0))

    def draw_fitted_centered_text(
        frame: np.ndarray,
        text: str,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        max_scale: float = 1.0,
        thickness: int = 2,
    ) -> None:
        if not text:
            return
        left, top, right, bottom = rect
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad_x = 12
        pad_y = 10
        avail_w = max(1, right - left - (2 * pad_x))
        avail_h = max(1, bottom - top - (2 * pad_y))
        scale = float(max_scale)
        min_scale = 0.35
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        while scale > min_scale and (text_size[0] > avail_w or text_size[1] > avail_h):
            scale = max(min_scale, scale - 0.05)
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = left + max(0, (right - left - text_size[0]) // 2)
        text_y = top + max(0, (bottom - top + text_size[1]) // 2)
        cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)

    def hit_test_keyboard_key(pt: tuple[int, int] | None, page_keys: list[KeyboardKey]) -> str | None:
        if pt is None:
            return None
        x, y = pt
        for key in page_keys:
            left, top, right, bottom = key["rect"]
            if left <= x <= right and top <= y <= bottom:
                return key["id"]
        return None

    def reset_keyboard_focus_state() -> None:
        nonlocal keyboard_focus_key, keyboard_focus_start, keyboard_dwell_progress, keyboard_armed_key
        nonlocal keyboard_candidate_key, keyboard_candidate_start
        keyboard_focus_key = None
        keyboard_focus_start = None
        keyboard_dwell_progress = 0.0
        keyboard_armed_key = None
        keyboard_candidate_key = None
        keyboard_candidate_start = None

    def execute_keyboard_key(key: KeyboardKey) -> None:
        nonlocal app_mode, keyboard_text
        if key["kind"] == "char":
            keyboard_text += key["value"]
            return

        action = key["value"]
        if action == "space":
            keyboard_text += " "
        elif action == "backspace":
            if keyboard_text:
                keyboard_text = keyboard_text[:-1]
        elif action == "enter":
            keyboard_text += "\n"
        elif action == "demo":
            app_mode = APP_MODE_DEMO
            reset_keyboard_focus_state()

    def soft_edge_curve(u: float, strength: float) -> float:
        # strength in [0..1], 0 = linear, 1 = strong
        u = float(np.clip(u, 0.0, 1.0))
        s = float(np.clip(strength, 0.0, 1.0))
        smooth = u * u * (3.0 - 2.0 * u)
        return (1.0 - s) * u + s * smooth

    def mid_edge_expand(u: float, v: float, strength: float = 0.18) -> float:
        """
        Expands coordinate u when the orthogonal coordinate v is near center.
        Used to fix mid-edge compression.
        """
        v_dist = abs(v - 0.5) * 2.0
        center_weight = 1.0 - np.clip(v_dist, 0.0, 1.0)
        expand = strength * center_weight
        return float(np.clip(0.5 + (u - 0.5) * (1.0 + expand), 0.0, 1.0))

    def vertical_extreme_damp(u: float, strength: float = 0.35) -> float:
        """
        Damp movement near top/bottom edges to prevent snapping.
        """
        d = abs(u - 0.5) * 2.0
        if d < 0.75:
            return u
        t = (d - 0.75) / 0.25
        damp = 1.0 - strength * t * t
        return float(np.clip(0.5 + (u - 0.5) * damp, 0.0, 1.0))

    def cornerness(x: float, y: float) -> float:
        # 0 = center, 1 = corner
        return max(abs(x - 0.5), abs(y - 0.5)) * 2.0

    def transform_gaze(gaze: tuple[float, float]) -> tuple[float, float]:
        gx = clamp01(gaze[0])
        gy = gaze[1]
        if y_flip:
            gy = 1.0 - gy
        gy = (gy - 0.5) * y_scale + 0.5 + y_offset
        gy = clamp01(gy)

        #gx = mid_edge_expand(gx, gy, strength=0.18)
        #gy = mid_edge_expand(gy, gx, strength=0.08)

        c = cornerness(gx, gy)
        if c < 0.75:
            #gx = soft_edge_curve(gx, edge_gain)
            gy = soft_edge_curve(gy, y_edge_gain)
            gy = vertical_extreme_damp(gy, strength=0.30)

        #reach = 1.08
        #gx = clamp01(0.5 + (gx - 0.5) * reach)
        #gy = clamp01(0.5 + (gy - 0.5) * reach)

        vel = tracker._last_velocity
        #if vel is not None and vel < 0.015:
            #precision_gain = 1.12
            #gx = clamp01(0.5 + (gx - 0.5) * precision_gain)
            #gy = clamp01(0.5 + (gy - 0.5) * precision_gain)
        return gx, gy

    def parse_pose_features(pose: object) -> tuple[float, float, float, float] | None:
        if not isinstance(pose, dict):
            return None
        try:
            yaw = float(pose.get("yaw", 0.0))
            pitch = float(pose.get("pitch", 0.0))
            roll = float(pose.get("roll", 0.0))
            tz = float(pose.get("tz", 0.0))
        except (TypeError, ValueError):
            return None
        if not np.isfinite([yaw, pitch, roll, tz]).all():
            return None
        return yaw, pitch, roll, tz

    def pose_is_stable_for_calibration(
        pose_vals: tuple[float, float, float, float] | None,
        pose_ref: tuple[float, float, float, float] | None,
    ) -> bool:
        if pose_vals is None or pose_ref is None:
            return False
        return (
            abs(pose_vals[0] - pose_ref[0]) <= calib_pose_yaw_thresh
            and abs(pose_vals[1] - pose_ref[1]) <= calib_pose_pitch_thresh
            and abs(pose_vals[2] - pose_ref[2]) <= calib_pose_roll_thresh
        )

    def append_ml_sample(
        gaze_raw_sample: object,
        pose_sample: object,
        target_xy: tuple[float, float],
    ) -> bool:
        if not isinstance(gaze_raw_sample, tuple):
            return False
        pose_vals = parse_pose_features(pose_sample)
        if pose_vals is None:
            return False
        gaze_ax = tracker.apply_axis_flip((float(gaze_raw_sample[0]), float(gaze_raw_sample[1])))
        feat = [float(gaze_ax[0]), float(gaze_ax[1]), pose_vals[0], pose_vals[1], pose_vals[2], pose_vals[3]]
        if not np.isfinite(feat).all():
            return False
        train_X.append(feat)
        train_Y.append([float(target_xy[0]), float(target_xy[1])])
        return True

    def pursuit_target_at(elapsed_s: float) -> tuple[float, float]:
        if elapsed_s <= 0.0:
            return pursuit_points[0]
        t = float(np.clip(elapsed_s / pursuit_duration_s, 0.0, 1.0))
        seg_count = len(pursuit_points) - 1
        pos = t * seg_count
        seg_idx = min(int(pos), seg_count - 1)
        frac = pos - seg_idx
        x0, y0 = pursuit_points[seg_idx]
        x1, y1 = pursuit_points[seg_idx + 1]
        return float(x0 + (x1 - x0) * frac), float(y0 + (y1 - y0) * frac)

    def get_cursor_pos() -> tuple[int, int]:
        if user32 is None or wintypes is None:
            return 0, 0
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def set_cursor_pos(x_px: int, y_px: int) -> None:
        if user32 is None:
            return
        user32.SetCursorPos(int(x_px), int(y_px))

    def os_left_click(x_px: int, y_px: int) -> bool:
        if user32 is None:
            return False
        x0 = int(vx)
        y0 = int(vy)
        w_span = int(vw) if isinstance(vw, int) and vw > 0 else int(screen_w or 1)
        h_span = int(vh) if isinstance(vh, int) and vh > 0 else int(screen_h or 1)
        x1 = x0 + max(1, w_span) - 1
        y1 = y0 + max(1, h_span) - 1
        cx = int(np.clip(x_px, x0, x1))
        cy = int(np.clip(y_px, y0, y1))
        try:
            user32.SetCursorPos(cx, cy)
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
        except Exception:
            return False
        return True

    def send_left_click() -> bool:
        if user32 is None:
            return False
        try:
            if ctypes is None:
                raise RuntimeError("ctypes unavailable")

            if wintypes is None:
                raise RuntimeError("wintypes unavailable")

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", wintypes.ULONG_PTR),
                ]

            class INPUT(ctypes.Structure):
                class _INPUT_UNION(ctypes.Union):
                    _fields_ = [("mi", MOUSEINPUT)]

                _anonymous_ = ("u",)
                _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]

            INPUT_MOUSE = 0
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004

            down = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0))
            up = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0))
            sent = user32.SendInput(2, ctypes.byref((INPUT * 2)(down, up)), ctypes.sizeof(INPUT))
            if int(sent) == 2:
                return True
            raise RuntimeError(f"SendInput returned {sent}")
        except Exception:
            try:
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
                return True
            except Exception:
                return False

    frame_times = deque(maxlen=30)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: failed to read frame.")
            break

        frame = cv2.flip(frame, 1)

        result = tracker.process_frame(frame)
        face_detected = bool(result.get("face_detected"))
        gaze_raw = result.get("gaze_raw_uncal")
        eye_open = result.get("eye_openness")
        head_pose = result.get("head_pose")
        gaze_pointer = tracker.get_mapped_gaze(result)

        if isinstance(gaze_pointer, tuple):
            if mouse_enabled:
                drift_rate = 0.002
                drift_x = float(
                    np.clip(drift_x + (gaze_pointer[0] - 0.5) * drift_rate, -drift_max, drift_max)
                )
                drift_y = float(
                    np.clip(drift_y + (gaze_pointer[1] - 0.5) * drift_rate, -drift_max, drift_max)
                )
            else:
                drift_x *= 0.995
                drift_y *= 0.995

        display_gaze = None
        target_x = None
        target_y = None
        if isinstance(gaze_pointer, tuple) and vw is not None and vh is not None:
            display_gaze = transform_gaze(gaze_pointer)
            dx = clamp01(display_gaze[0] + drift_x)
            dy = clamp01(display_gaze[1] + drift_y)
            target_x = vx + int(dx * (vw - 1))
            target_y = vy + int(dy * (vh - 1))

        if screen_w is None or screen_h is None:
            screen_h, screen_w = frame.shape[:2]
        if vw is None or vh is None or vw <= 0 or vh <= 0:
            vw = screen_w
            vh = screen_h
        refresh_keyboard_layout()
        if demo_ui is None:
            demo_ui = DemoUI(int(screen_w), int(screen_h), assist_on=assist_enabled)
        else:
            demo_ui.set_screen_size(int(screen_w), int(screen_h))

        now = time.time()
        if face_detected:
            last_face_time = now
            lost_frames = 0
        else:
            lost_frames += 1
            blink_closed = False
            if now - last_face_time > 0.25 and mouse_enabled:
                mouse_enabled = False
                sx = None
                sy = None
                vx_c = 0.0
                vy_c = 0.0
                print("Mouse control auto-disabled (face lost).")

        if calib_active:
            if calib_phase == "settle":
                if now - calib_phase_start >= calib_settle_s:
                    calib_phase = "capture"
                    calib_samples = []
                    calib_pose_ref = parse_pose_features(head_pose)
                    last_static_ml_sample_time = 0.0
            elif calib_phase == "capture":
                if calib_index >= len(calib_targets):
                    calib_phase = "fit"
                else:
                    target_xy = calib_targets[calib_index][2]
                    if face_detected and isinstance(gaze_raw, tuple) and isinstance(eye_open, float):
                        pose_vals = parse_pose_features(head_pose)
                        if pose_vals is not None:
                            if calib_pose_ref is None:
                                calib_pose_ref = pose_vals
                            if pose_is_stable_for_calibration(pose_vals, calib_pose_ref):
                                calib_samples.append((gaze_raw[0], gaze_raw[1], eye_open))
                                if last_static_ml_sample_time == 0.0 or now - last_static_ml_sample_time >= static_ml_sample_interval_s:
                                    if append_ml_sample(gaze_raw, head_pose, target_xy):
                                        last_static_ml_sample_time = now
                    if len(calib_samples) >= calib_samples_needed:
                        gaze_only = [(s[0], s[1]) for s in calib_samples]
                        opens = [s[2] for s in calib_samples]
                        med, mad = tracker.compute_median_mad(gaze_only)
                        open_med = float(np.median(opens)) if opens else 0.0
                        if max(mad[0], mad[1]) > calib_mad_thresh:
                            print(f"Calibration point unstable ({calib_targets[calib_index][1]}), retrying.")
                            calib_phase = "settle"
                            calib_phase_start = now
                            calib_samples = []
                            calib_pose_ref = None
                            continue

                        name = calib_targets[calib_index][0]
                        calib_data[name] = (float(med[0]), float(med[1]))
                        calib_open[name] = open_med
                        if name == "center":
                            center_open_list = opens[:]
                            center_gy_list = [s[1] for s in calib_samples]
                        calib_quality[name] = {
                            "median_x": float(med[0]),
                            "median_y": float(med[1]),
                            "mad_x": float(mad[0]),
                            "mad_y": float(mad[1]),
                        }
                        # Ensure each static anchor contributes at least one ML sample.
                        append_ml_sample((float(med[0]), float(med[1])), head_pose, target_xy)

                        calib_index += 1
                        calib_samples = []
                        calib_pose_ref = None
                        if calib_index < len(calib_targets):
                            calib_phase = "settle"
                            calib_phase_start = now
                        else:
                            success = tracker.set_full_calibration(calib_data, pad=calib_pad)
                            if not success:
                                print("Calibration failed: range invalid.")
                                calib_active = False
                                calib_phase = "idle"
                                pursuit_target_xy = None
                            else:
                                open_ref = calib_open.get("center")
                                beta_y = 0.0
                                if center_open_list:
                                    open_arr = np.array(center_open_list, dtype=np.float32)
                                    gy_arr = np.array(center_gy_list, dtype=np.float32)
                                    mean_open = float(np.mean(open_arr))
                                    mean_gy = float(np.mean(gy_arr))
                                    cov = float(np.mean((gy_arr - mean_gy) * (open_arr - mean_open)))
                                    var = float(np.mean((open_arr - mean_open) ** 2))
                                    if var > 1e-6:
                                        beta_y = cov / var
                                if open_ref is not None:
                                    tracker.set_openness_compensation(open_ref, beta_y)
                                tracker.set_calibration_quality(calib_quality)
                                top_vals = [calib_data[k][1] for k in ("tl", "t", "tr") if k in calib_data]
                                bot_vals = [calib_data[k][1] for k in ("bl", "b", "br") if k in calib_data]
                                top_vals = [tracker.apply_axis_flip((0.5, y))[1] for y in top_vals]
                                bot_vals = [tracker.apply_axis_flip((0.5, y))[1] for y in bot_vals]
                                if top_vals and bot_vals:
                                    span = abs(float(np.median(bot_vals)) - float(np.median(top_vals)))
                                    if span < 0.10:
                                        print("Vertical span too small; keep head fixed, avoid eyebrow movement; try again.")
                                        y_scale = float(np.clip(0.18 / max(span, 1e-3), 0.8, 2.5))
                                pursuit_start_time = now
                                last_pursuit_sample_time = 0.0
                                pursuit_target_xy = pursuit_points[0]
                                calib_phase = "pursuit"
                                print("[ML] static calibration done. Starting pursuit capture...")
            elif calib_phase == "pursuit":
                elapsed = now - pursuit_start_time
                pursuit_target_xy = pursuit_target_at(elapsed)
                if (
                    face_detected
                    and isinstance(gaze_raw, tuple)
                    and (last_pursuit_sample_time == 0.0 or now - last_pursuit_sample_time >= pursuit_sample_interval_s)
                ):
                    if append_ml_sample(gaze_raw, head_pose, pursuit_target_xy):
                        last_pursuit_sample_time = now
                if elapsed >= pursuit_duration_s:
                    calib_phase = "fit"
            elif calib_phase == "fit":
                print(f"[ML] fitting on {len(train_X)} samples")
                ml_ok = tracker.fit_ml_calibration(train_X, train_Y, alpha=0.5)
                if ml_ok:
                    if tracker.save_ml_model(ml_model_path, user_id=user_id):
                        print(f"[ML] model saved to {ml_model_path}")
                    else:
                        print("[ML] fit succeeded, but model file save failed.")
                else:
                    print("[ML] fit failed, keeping fallback calibration mapping.")

                if tracker.save_calibration(calib_path):
                    print(f"Calibration saved to {calib_path}")
                else:
                    print("Calibration save failed.")
                if tracker._calib_range is not None:
                    print("gx_min:", tracker._calib_range[0], "gx_max:", tracker._calib_range[1])
                    print("gy_min:", tracker._calib_range[2], "gy_max:", tracker._calib_range[3])
                    print(
                        "span_x:",
                        tracker._calib_range[1] - tracker._calib_range[0],
                        "span_y:",
                        tracker._calib_range[3] - tracker._calib_range[2],
                    )
                print(f"[ML] mapper status: {'ON' if tracker.is_ml_ready() else 'OFF'}")
                calib_active = False
                calib_phase = "idle"
                pursuit_target_xy = None

        should_update_baseline = (
            face_detected
            and isinstance(eye_open, float)
            and mouse_enabled
            and not mouse_paused
            and not calib_active
            and tracker.has_full_calibration()
        )
        vel = tracker._last_velocity
        if should_update_baseline and vel is not None and vel < BASELINE_UPDATE_VEL:
            eye_open_val = float(eye_open)
            if np.isfinite(eye_open_val):
                open_ref_samples.append(eye_open_val)
                if len(open_ref_samples) >= 10:
                    med_open = float(np.median(np.array(open_ref_samples, dtype=np.float32)))
                    open_ref = float(np.clip(med_open, BASELINE_MIN, BASELINE_MAX))

        frame_times.append(now)
        fps = 0.0
        if len(frame_times) >= 2:
            span = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / span if span > 0 else 0.0

        h, w = frame.shape[:2]
        calib_status = "FULL" if tracker.has_full_calibration() else "NONE"
        if calib_active:
            if calib_phase in ("settle", "capture") and calib_index < len(calib_targets):
                target_label = calib_targets[calib_index][1]
                if calib_phase == "settle":
                    calib_status = f"IN_PROGRESS {target_label} (settle)"
                else:
                    calib_status = f"IN_PROGRESS {target_label} ({len(calib_samples)}/{calib_samples_needed})"
            elif calib_phase == "pursuit":
                calib_status = "IN_PROGRESS PURSUIT"
            elif calib_phase == "fit":
                calib_status = "IN_PROGRESS FIT"

        result["calib_status"] = calib_status
        result["ml_status"] = f"{'ON' if tracker.is_ml_ready() else 'OFF'} samples={len(train_X)}"
        result["show_pose_indices"] = show_pose_indices
        debug_frame = tracker.draw_debug(frame, result)

        cv2.putText(
            debug_frame,
            f"FPS: {fps:5.1f}",
            (w - 140, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        mouse_status = f"mouse={'ON' if mouse_enabled else 'OFF'} pause={'ON' if mouse_paused else 'OFF'}"
        backend_text = f"backend={cursor_backend} lost={lost_frames} mode={app_mode}"
        desktop_text = f"desktop=({vx},{vy},{vw},{vh}) target=({target_x},{target_y})"
        cv2.putText(
            debug_frame,
            mouse_status,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            backend_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            desktop_text,
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        eye_open_text = f"{float(eye_open):.3f}" if isinstance(eye_open, float) else "None"
        open_ref_text = f"{open_ref:.3f}" if open_ref is not None else "None"
        blink_text = f"blink={'ON' if blink_enabled else 'OFF'} closed={blink_closed}"
        blink_ref_text = f"eye={eye_open_text} ref={open_ref_text}"
        cv2.putText(
            debug_frame,
            blink_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            debug_frame,
            blink_ref_text,
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )

        cv2.imshow(debug_win, debug_frame)

        pointer_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        overlay_active = tracker.has_full_calibration() and not calib_active
        demo_active = demo_ui is not None and overlay_active and app_mode == APP_MODE_DEMO
        keyboard_active = overlay_active and app_mode == APP_MODE_KEYBOARD
        if demo_active and demo_ui is not None:
            raw_local_px = None
            raw_desktop_px = None
            if isinstance(display_gaze, tuple):
                gx, gy = clamp01(display_gaze[0]), clamp01(display_gaze[1])
                raw_local_px = normalized_to_local_px((gx, gy), int(screen_w), int(screen_h))
                raw_desktop_px = local_to_desktop_px(
                    raw_local_px,
                    int(screen_w),
                    int(screen_h),
                    int(vx),
                    int(vy),
                    int(vw),
                    int(vh),
                )

            if drift_enabled and isinstance(display_gaze, tuple):
                demo_drift_rate = 0.0005
                demo_drift_x = float(
                    np.clip(demo_drift_x + (display_gaze[0] - 0.5) * demo_drift_rate, -demo_drift_max, demo_drift_max)
                )
                demo_drift_y = float(
                    np.clip(demo_drift_y + (display_gaze[1] - 0.5) * demo_drift_rate, -demo_drift_max, demo_drift_max)
                )
            else:
                demo_drift_x *= 0.95
                demo_drift_y *= 0.95

            drift_px = (
                int(round(demo_drift_x * max(1, int(screen_w) - 1))),
                int(round(demo_drift_y * max(1, int(screen_h) - 1))),
            )
            demo_ui.set_assist_enabled(assist_enabled)
            demo_ui.update(
                int(now * 1000.0),
                raw_local_px,
                drift_offset_px=drift_px,
                face_detected=face_detected,
                raw_desktop_px=raw_desktop_px,
            )
            demo_ui.render(pointer_frame, int(now * 1000.0))

            assist_local = getattr(demo_ui, "assist_px", None)
            assist_valid = (
                assist_enabled
                and demo_active
                and isinstance(raw_local_px, tuple)
                and isinstance(assist_local, tuple)
                and len(assist_local) == 2
            )
            if assist_valid:
                try:
                    ax_l = int(round(float(assist_local[0])))
                    ay_l = int(round(float(assist_local[1])))
                except (TypeError, ValueError):
                    ax_l = None
                    ay_l = None
                if ax_l is None or ay_l is None:
                    assist_valid = False

            if assist_valid:
                dx = float(ax_l - raw_local_px[0])
                dy = float(ay_l - raw_local_px[1])
                dist = (dx * dx + dy * dy) ** 0.5

                SNAP_R = 90.0
                if dist <= SNAP_R:
                    ax, ay = local_to_desktop_px(
                        (ax_l, ay_l),
                        int(screen_w),
                        int(screen_h),
                        int(vx),
                        int(vy),
                        int(vw),
                        int(vh),
                    )
                    target_x, target_y = int(ax), int(ay)

                    cv2.circle(pointer_frame, (ax_l, ay_l), 6, (0, 255, 0), -1)
                    cv2.putText(
                        pointer_frame,
                        f"ASSIST dist={dist:.0f}",
                        (24, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
        elif keyboard_active:
            pointer_frame[:] = (12, 14, 20)
            title = "Typing mode"
            page_keys = keyboard_pages.get(keyboard_page)
            keyboard_gaze_px = None
            if isinstance(gaze_pointer, tuple):
                gx = clamp01(gaze_pointer[0])
                gy = clamp01(gaze_pointer[1])
                keyboard_gaze_px = normalized_to_local_px((gx, gy), int(screen_w), int(screen_h))
            title_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            title_x = max(0, (int(screen_w) - title_size[0]) // 2)

            if not page_keys:
                reset_keyboard_focus_state()
                cv2.putText(
                    pointer_frame,
                    title,
                    (title_x, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (245, 245, 245),
                    2,
                )
                fallback = "Keyboard layout unavailable."
                fallback_size, _ = cv2.getTextSize(fallback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(
                    pointer_frame,
                    fallback,
                    (max(0, (int(screen_w) - fallback_size[0]) // 2), max(90, int(screen_h) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (210, 210, 210),
                    2,
                )
            else:
                focused_key_id = hit_test_keyboard_key(keyboard_gaze_px, page_keys)
                if keyboard_focus_key is None:
                    keyboard_candidate_key = None
                    keyboard_candidate_start = None
                    if focused_key_id is not None:
                        keyboard_focus_key = focused_key_id
                        keyboard_focus_start = now
                        keyboard_dwell_progress = 0.0
                        keyboard_armed_key = focused_key_id
                elif focused_key_id == keyboard_focus_key:
                    keyboard_candidate_key = None
                    keyboard_candidate_start = None
                elif focused_key_id is None:
                    reset_keyboard_focus_state()
                else:
                    if focused_key_id != keyboard_candidate_key:
                        keyboard_candidate_key = focused_key_id
                        keyboard_candidate_start = now
                    elif keyboard_candidate_start is None:
                        keyboard_candidate_start = now
                    elif (now - keyboard_candidate_start) >= KEYBOARD_FOCUS_SWITCH_S:
                        keyboard_focus_key = focused_key_id
                        keyboard_focus_start = now
                        keyboard_dwell_progress = 0.0
                        keyboard_armed_key = focused_key_id
                        keyboard_candidate_key = None
                        keyboard_candidate_start = None

                if keyboard_focus_key is None or keyboard_focus_start is None:
                    keyboard_dwell_progress = 0.0
                else:
                    elapsed = max(0.0, now - keyboard_focus_start)
                    keyboard_dwell_progress = float(np.clip(elapsed / KEYBOARD_DWELL_S, 0.0, 1.0))

                if (
                    keyboard_focus_key is not None
                    and keyboard_armed_key == keyboard_focus_key
                    and keyboard_dwell_progress >= 1.0
                ):
                    active_key = next((key for key in page_keys if key["id"] == keyboard_focus_key), None)
                    if active_key is not None:
                        execute_keyboard_key(active_key)
                    keyboard_armed_key = None

                keys_left = min(key["rect"][0] for key in page_keys)
                keys_top = min(key["rect"][1] for key in page_keys)
                keys_right = max(key["rect"][2] for key in page_keys)
                keys_bottom = max(key["rect"][3] for key in page_keys)

                panel_pad_x = max(16, int(0.02 * screen_w))
                panel_pad_y = max(16, int(0.02 * screen_h))
                field_gap = max(16, int(0.02 * screen_h))
                field_h = max(56, int(0.09 * screen_h))
                field_rect = (
                    keys_left,
                    max(24, keys_top - field_h - field_gap),
                    keys_right,
                    max(24 + field_h, keys_top - field_gap),
                )
                panel_rect = (
                    max(0, keys_left - panel_pad_x),
                    max(0, field_rect[1] - panel_pad_y - 56),
                    min(int(screen_w) - 1, keys_right + panel_pad_x),
                    min(int(screen_h) - 1, keys_bottom + panel_pad_y),
                )

                cv2.rectangle(
                    pointer_frame,
                    (panel_rect[0], panel_rect[1]),
                    (panel_rect[2], panel_rect[3]),
                    (24, 28, 38),
                    -1,
                )
                cv2.rectangle(
                    pointer_frame,
                    (panel_rect[0], panel_rect[1]),
                    (panel_rect[2], panel_rect[3]),
                    (58, 66, 84),
                    2,
                )

                title_y = max(40, field_rect[1] - 18)
                cv2.putText(
                    pointer_frame,
                    title,
                    (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (245, 245, 245),
                    2,
                )

                cv2.rectangle(
                    pointer_frame,
                    (field_rect[0], field_rect[1]),
                    (field_rect[2], field_rect[3]),
                    (16, 18, 26),
                    -1,
                )
                cv2.rectangle(
                    pointer_frame,
                    (field_rect[0], field_rect[1]),
                    (field_rect[2], field_rect[3]),
                    (160, 168, 186),
                    2,
                )

                display_text = keyboard_text
                field_font = cv2.FONT_HERSHEY_SIMPLEX
                field_scale = 0.9
                field_thickness = 2
                text_pad_x = 18
                text_avail_w = max(1, field_rect[2] - field_rect[0] - (2 * text_pad_x))
                while display_text:
                    text_w = cv2.getTextSize(display_text, field_font, field_scale, field_thickness)[0][0]
                    if text_w <= text_avail_w:
                        break
                    display_text = "..." + display_text[1:]
                if display_text:
                    text_size, _ = cv2.getTextSize(display_text, field_font, field_scale, field_thickness)
                    text_x = field_rect[0] + text_pad_x
                    text_y = field_rect[1] + max(0, (field_rect[3] - field_rect[1] + text_size[1]) // 2)
                    cv2.putText(
                        pointer_frame,
                        display_text,
                        (text_x, text_y),
                        field_font,
                        field_scale,
                        (245, 245, 245),
                        field_thickness,
                    )

                for key in page_keys:
                    left, top, right, bottom = key["rect"]
                    is_focused = key["id"] == keyboard_focus_key
                    if key["kind"] == "action":
                        fill_color = (58, 68, 92)
                        border_color = (196, 208, 224)
                    else:
                        fill_color = (44, 50, 66)
                        border_color = (180, 188, 206)
                    label_color = (245, 245, 245)
                    border_thickness = 2
                    if is_focused:
                        fill_color = (88, 110, 148) if key["kind"] == "action" else (76, 96, 132)
                        border_color = (255, 244, 160)
                        label_color = (255, 255, 255)
                        border_thickness = 4
                    cv2.rectangle(pointer_frame, (left, top), (right, bottom), fill_color, -1)
                    cv2.rectangle(pointer_frame, (left, top), (right, bottom), border_color, border_thickness)
                    if is_focused:
                        bar_pad_x = max(8, int(0.08 * max(1, right - left)))
                        bar_h = int(np.clip(bottom - top, 6, 8))
                        bar_left = left + bar_pad_x
                        bar_right = right - bar_pad_x
                        bar_bottom = bottom - 8
                        bar_top = max(top + 8, bar_bottom - bar_h)
                        if bar_right > bar_left and bar_bottom > bar_top:
                            cv2.rectangle(
                                pointer_frame,
                                (bar_left, bar_top),
                                (bar_right, bar_bottom),
                                (28, 30, 36),
                                -1,
                            )
                            fill_right = bar_left + int(round((bar_right - bar_left) * keyboard_dwell_progress))
                            if fill_right > bar_left:
                                cv2.rectangle(
                                    pointer_frame,
                                    (bar_left, bar_top),
                                    (fill_right, bar_bottom),
                                    (255, 220, 110),
                                    -1,
                                )
                    draw_fitted_centered_text(
                        pointer_frame,
                        key["label"],
                        (left, top, right, bottom),
                        label_color,
                        max_scale=0.9 if key["kind"] == "char" else 0.75,
                        thickness=2,
                    )
        elif test_active:
            if isinstance(display_gaze, tuple):
                px = int(np.clip(display_gaze[0], 0.0, 1.0) * (screen_w - 1))
                py = int(np.clip(display_gaze[1], 0.0, 1.0) * (screen_h - 1))
                cv2.circle(pointer_frame, (px, py), 18, (255, 0, 0), -1)

            elapsed = time.time() - test_start
            if elapsed >= test_duration * 2:
                test_active = False
            else:
                phase = 0 if elapsed < test_duration else 1
                label = "TEST TOP" if phase == 0 else "TEST BOTTOM"
                pos = (0.5, 0.15) if phase == 0 else (0.5, 0.85)
                tx = int(pos[0] * (screen_w - 1))
                ty = int(pos[1] * (screen_h - 1))
                cv2.line(pointer_frame, (tx - 25, ty), (tx + 25, ty), (255, 255, 255), 2)
                cv2.line(pointer_frame, (tx, ty - 25), (tx, ty + 25), (255, 255, 255), 2)

                raw_text = "raw gy=None"
                if isinstance(gaze_raw, tuple):
                    raw_text = f"raw gy={gaze_raw[1]:.3f}"
                trans_text = "gy2=None"
                if isinstance(display_gaze, tuple):
                    trans_text = f"gy2={display_gaze[1]:.3f}"
                y_target_text = f"target_y={target_y}"
                cv2.putText(
                    pointer_frame,
                    f"{label} {raw_text} {trans_text} {y_target_text}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
        elif calib_active and calib_phase in ("settle", "capture") and calib_index < len(calib_targets):
            _, label, pos = calib_targets[calib_index]
            tx = int(pos[0] * (screen_w - 1))
            ty = int(pos[1] * (screen_h - 1))
            cv2.line(pointer_frame, (tx - 25, ty), (tx + 25, ty), (255, 255, 255), 2)
            cv2.line(pointer_frame, (tx, ty - 25), (tx, ty + 25), (255, 255, 255), 2)

            instruction = f"LOOK AT {label}"
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        elif calib_active and calib_phase == "pursuit" and pursuit_target_xy is not None:
            tx = int(pursuit_target_xy[0] * (screen_w - 1))
            ty = int(pursuit_target_xy[1] * (screen_h - 1))
            cv2.line(pointer_frame, (tx - 20, ty), (tx + 20, ty), (255, 255, 255), 2)
            cv2.line(pointer_frame, (tx, ty - 20), (tx, ty + 20), (255, 255, 255), 2)
            instruction = "FOLLOW THE TARGET"
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        elif calib_active and calib_phase == "fit":
            instruction = "FITTING ML MODEL..."
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        elif not tracker.has_full_calibration():
            instruction = "PRESS K TO CALIBRATE"
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            if isinstance(display_gaze, tuple):
                px = int(np.clip(display_gaze[0], 0.0, 1.0) * (screen_w - 1))
                py = int(np.clip(display_gaze[1], 0.0, 1.0) * (screen_h - 1))
                cv2.circle(pointer_frame, (px, py), 16, (255, 0, 0), -1)

        cv2.imshow(pointer_win, pointer_frame)

        should_move = (
            mouse_enabled
            and not mouse_paused
            and tracker.has_full_calibration()
            and not calib_active
            and face_detected
            and isinstance(display_gaze, tuple)
            and target_x is not None
            and target_y is not None
        )
        if should_move:
            try:
                if sx is None or sy is None:
                    cx, cy = get_cursor_pos()
                    sx = float(cx)
                    sy = float(cy)
                    vx_c = 0.0
                    vy_c = 0.0

                dt = float(np.clip(now - last_cursor_time, 1e-4, 0.05))
                last_cursor_time = now

                ex = float(target_x - sx)
                ey = float(target_y - sy)
                ax = spring_k * ex - spring_d * vx_c
                edge_y = abs((sy / vh) - 0.5) * 2.0 if vh else 0.0
                y_damp = 1.0 - 0.35 * np.clip(edge_y, 0.0, 1.0)
                ay = (spring_k * y_damp) * ey - spring_d * vy_c
                ax = float(np.clip(ax, -max_accel, max_accel))
                ay = float(np.clip(ay, -max_accel, max_accel))
                vx_c += ax * dt
                vy_c += ay * dt
                vx_c = float(np.clip(vx_c, -max_speed, max_speed))
                vy_c = float(np.clip(vy_c, -max_speed, max_speed))
                sx += vx_c * dt
                sy += vy_c * dt
                set_cursor_pos(int(round(sx)), int(round(sy)))
            except Exception as exc:
                print(f"Mouse control error: {exc}")
                mouse_enabled = False

        can_detect_blink = (
            blink_enabled
            and mouse_enabled
            and not mouse_paused
            and tracker.has_full_calibration()
            and not calib_active
            and face_detected
            and isinstance(eye_open, float)
            and open_ref is not None
        )
        if not face_detected:
            blink_closed = False
        elif can_detect_blink:
            eye_open_val = float(eye_open)
            closed_now = eye_open_val < float(open_ref) * BLINK_CLOSE_RATIO
            open_now = eye_open_val > float(open_ref) * BLINK_OPEN_RATIO

            if not blink_closed and closed_now:
                blink_closed = True
                blink_down_ts = now
            elif blink_closed and open_now:
                duration = now - blink_down_ts
                blink_closed = False
                if duration >= BLINK_MIN_HOLD_S and (now - last_click_ts) >= CLICK_COOLDOWN_S:
                    send_left_click()
                    last_click_ts = now
        else:
            if not mouse_enabled or mouse_paused or calib_active:
                blink_closed = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 27:
            if demo_active and demo_ui is not None:
                demo_ui.cancel_armed()
                print("Armed selection canceled.")
            else:
                mouse_enabled = False
                print("Mouse control canceled.")
        if key == ord("s"):
            filename = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
            path = os.path.join(os.getcwd(), filename)
            cv2.imwrite(path, debug_frame)
            print(f"Saved screenshot: {path}")
        if key == ord("k"):
            if mouse_enabled or mouse_paused:
                y_scale = float(np.clip(y_scale - 0.05, 0.6, 2.5))
                print(f"y_scale={y_scale:.2f}")
            else:
                calib_active = True
                calib_phase = "settle"
                calib_index = 0
                calib_phase_start = time.time()
                pursuit_start_time = 0.0
                pursuit_target_xy = None
                last_pursuit_sample_time = 0.0
                last_static_ml_sample_time = 0.0
                calib_pose_ref = None
            train_X = []
            train_Y = []
            calib_samples = []
            calib_data = {}
            calib_quality = {}
            calib_open = {}
            center_open_list = []
            center_gy_list = []
            tracker.reset_calibration()
            mouse_enabled = False
            sx = None
            sy = None
            vx_c = 0.0
            vy_c = 0.0
            print("Calibration started: 3x3 grid (top-left -> top -> top-right -> left -> center -> right -> bottom-left -> bottom -> bottom-right)")
        if key == ord("r"):
            tracker.reset_calibration()
            calib_active = False
            calib_phase = "idle"
            calib_index = 0
            pursuit_start_time = 0.0
            pursuit_target_xy = None
            last_pursuit_sample_time = 0.0
            last_static_ml_sample_time = 0.0
            calib_pose_ref = None
            calib_samples = []
            calib_data = {}
            calib_quality = {}
            calib_open = {}
            center_open_list = []
            center_gy_list = []
            train_X = []
            train_Y = []
            mouse_enabled = False
            sx = None
            sy = None
            vx_c = 0.0
            vy_c = 0.0
            print("Calibration reset.")
        if key == ord("l"):
            if mouse_enabled or mouse_paused:
                y_offset = float(np.clip(y_offset - 0.01, -0.25, 0.25))
                print(f"y_offset={y_offset:.2f}")
            else:
                if tracker.load_calibration(calib_path):
                    print(f"Loaded calibration from {calib_path}")
                    if not tracker.is_ml_ready() and os.path.exists(ml_model_path):
                        tracker.load_ml_model(ml_model_path, user_id=user_id)
                    print(f"ML mapper: {'ON' if tracker.is_ml_ready() else 'OFF'}")
                else:
                    print("Load failed: no calibration data found.")
        if key == ord("m"):
            if not tracker.has_full_calibration():
                print("Mouse control requires full calibration.")
            elif calib_active:
                print("Mouse control disabled during calibration.")
            else:
                mouse_enabled = not mouse_enabled
                blink_closed = False
                if mouse_enabled:
                    cx, cy = get_cursor_pos()
                    sx = float(cx)
                    sy = float(cy)
                    vx_c = 0.0
                    vy_c = 0.0
                print(f"Mouse control {'enabled' if mouse_enabled else 'disabled'}.")
        if key == ord("v"):
            if calib_active:
                print("Mode switch unavailable during calibration.")
            else:
                if app_mode == APP_MODE_DEMO:
                    app_mode = APP_MODE_KEYBOARD
                else:
                    app_mode = APP_MODE_DEMO
                    reset_keyboard_focus_state()
                print(f"Overlay mode: {app_mode}.")
        if key == ord("b"):
            blink_enabled = not blink_enabled
            blink_closed = False
            print(f"Blink click {'enabled' if blink_enabled else 'disabled'}.")
        if key == ord(" "):
            if demo_active and demo_ui is not None:
                event = demo_ui.confirm(int(now * 1000.0))
                if event.get("type") == "success":
                    if os_click_enabled:
                        click_pt = event.get("click_px")
                        if isinstance(click_pt, tuple) and len(click_pt) == 2:
                            desktop_click = local_to_desktop_px(
                                (int(click_pt[0]), int(click_pt[1])),
                                int(screen_w),
                                int(screen_h),
                                int(vx),
                                int(vy),
                                int(vw),
                                int(vh),
                            )
                            did_click = os_left_click(int(desktop_click[0]), int(desktop_click[1]))
                            if not did_click:
                                print("OS click failed (backend unavailable).")
                elif event.get("type") == "false_select":
                    pass
            else:
                mouse_paused = not mouse_paused
        if key == ord("["):
            edge_gain = float(np.clip(edge_gain + 0.02, 0.0, 0.5))
            print(f"edge_gain={edge_gain:.2f}")
        if key == ord("]"):
            edge_gain = float(np.clip(edge_gain - 0.02, 0.0, 0.5))
            print(f"edge_gain={edge_gain:.2f}")
        if key == ord(","):
            spring_k = float(np.clip(spring_k - 5.0, 20.0, 120.0))
            print(f"spring_k={spring_k:.0f}")
        if key == ord("."):
            spring_k = float(np.clip(spring_k + 5.0, 20.0, 120.0))
            print(f"spring_k={spring_k:.0f}")
        if key == ord("i"):
            y_scale = float(np.clip(y_scale + 0.05, 0.6, 2.5))
            print(f"y_scale={y_scale:.2f}")
        if key == ord("o"):
            y_offset = float(np.clip(y_offset + 0.01, -0.25, 0.25))
            print(f"y_offset={y_offset:.2f}")
        if key == ord("p"):
            if demo_active and demo_ui is not None:
                paused_now = demo_ui.toggle_pause()
                print(f"Demo pause {'ON' if paused_now else 'OFF'}.")
            else:
                show_pose_indices = not show_pose_indices
                print(f"Pose index debug: {show_pose_indices}")
        if key == ord("y"):
            y_flip = not y_flip
            print(f"y_flip={y_flip}")
        if key == ord("9"):
            y_edge_gain = float(np.clip(y_edge_gain + 0.02, 0.0, 0.5))
            print(f"y_edge_gain={y_edge_gain:.2f}")
        if key == ord("0"):
            y_edge_gain = float(np.clip(y_edge_gain - 0.02, 0.0, 0.5))
            print(f"y_edge_gain={y_edge_gain:.2f}")
        if key == ord("t"):
            test_active = True
            test_start = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
