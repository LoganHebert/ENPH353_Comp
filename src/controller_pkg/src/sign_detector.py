#!/usr/bin/env python3
"""
Sign detector + OCR + score-tracker publisher for ENPH353 Fizz Detective (ROS Noetic).

Executive summary (short)
- Detects the dark-blue framed clue board robustly (HSV + morphology + contour scoring + edge rejection).
- Warps the board to a fixed size for stable OCR.
- Finds the inner grey face automatically (border-projection first; grey-HSV fallback; conservative inset last).
- Locates top/bottom text lines dynamically via row-projection bands.
- Cleans per-line masks and removes edge strips only when they look like border bars.
- Segments characters (projection primary + contour fallback) and splits merged blobs via intra-box valley split.
- Preprocesses characters exactly like your CNN training pipeline and runs a TFLite model (finetuned preferred).
- Stabilizes output using temporal voting (pair vote over (type,value) to avoid mismatched top/bottom).
- Publishes to /score_tracker when a stable clue is achieved (and can auto-start/stop timer).
- Optional GUI that is stable (runs in a dedicated GUI thread to stop flicker).

Expected debug outputs (written to /tmp/sign_debug, overwritten periodically)
- frame.jpg
- sign_mask.jpg
- sign_debug.jpg
- warped.jpg
- face_mask.jpg
- face_debug.jpg
- face_crop.jpg
- top_line.jpg
- bottom_line.jpg
- top_char_mask.jpg
- bottom_char_mask.jpg
- top_char_debug.jpg
- bottom_char_debug.jpg
- top_chars_vis.jpg
- bottom_chars_vis.jpg

OpenCV docs referenced (primary/official)
- inRange (HSV masking): https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html
- morphologyEx (open/close): https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- findContours / approxPolyDP: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- getPerspectiveTransform / warpPerspective: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- threshold + Otsu: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- equalizeHist: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- GaussianBlur: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

Notes
- Official competition notes mention /B1/pi_camera/image_raw, /score_tracker messaging format, and start/stop
  timer special messages. This node supports that; edit Params.image_topic and Params.team_* accordingly.
"""

from __future__ import annotations

import os
import json
import time
import threading
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque, defaultdict

import cv2
import numpy as np
import rospy
import tensorflow as tf

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String as StringMsg
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Trigger, TriggerResponse


# =========================
# Params (EDIT THESE)
# =========================

@dataclass(frozen=True)
class Params:
    # ROS
    node_name: str = "sign_detector"

    # Competition notes nominal camera topic is /B1/pi_camera/image_raw
    # You can keep /B1/rrbot/camera1/image_raw if that's what your sim publishes.
    image_topic: str = "/B1/rrbot/camera1/image_raw"

    # Models / classes
    model_path_finetuned: str = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_model_finetuned.tflite")
    model_path_base: str = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_model.tflite")
    classes_path: str = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_classes.json")

    # Debug
    debug_dir: str = "/tmp/sign_debug"
    debug_save_every_sec: float = 1.0

    # Warp size
    warp_w: int = 400
    warp_h: int = 300

    # GUI
    enable_gui: bool = True
    gui_hz: float = 20.0  # stable update rate
    gui_main_window: str = "Sign Detector GUI"
    gui_warp_window: str = "Warped Sign View"
    gui_panel_w: int = 390

    # Internal JSON status publisher (for your own tools)
    status_topic: str = "/sign_detector/status_json"
    status_rate_hz: float = 5.0

    # Sample buffer + dump (for labelling)
    sample_buffer_len: int = 150
    dump_dir: str = "/tmp/sign_dataset_dump"
    dump_service_name: str = "/sign_detector/dump_samples"
    dump_trigger_topic: str = "/sign_detector/dump_samples_trigger"

    # Score tracker publishing (/score_tracker)
    enable_score_tracker_pub: bool = True
    score_tracker_topic: str = "/score_tracker"
    team_id: str = "TeamName"        # MUST be <=8 chars, no spaces
    team_password: str = "password"  # MUST be <=8 chars, no spaces
    auto_start_timer: bool = False   # set True for time-trials/competition
    auto_stop_timer_on_shutdown: bool = False
    min_score_pub_interval_sec: float = 0.25  # don't spam /score_tracker
    submit_only_once_per_clue: bool = True

    # Vote / stability
    vote_window: int = 15
    vote_min_count: int = 5                # require at least N samples in window for a stable output
    vote_min_pair_weight: float = 5.0      # minimum accumulated weight for best pair
    char_conf_threshold: float = 0.55
    word_mean_conf_threshold: float = 0.60
    min_char_conf_frac: float = 0.60       # at least this fraction of chars >= char_conf_threshold
    clear_vote_after_misses: int = 10      # don't clear vote on a single missed frame

    # Valid clue types
    valid_top_words: Tuple[str, ...] = (
        "SIZE", "VICTIM", "CRIME", "TIME",
        "PLACE", "MOTIVE", "WEAPON", "BANDIT"
    )

    # Sign detection HSV for dark-blue frame (reject sky by bounding V max and requiring high S)
    sign_blue_lower: Tuple[int, int, int] = (100, 120, 25)
    sign_blue_upper: Tuple[int, int, int] = (140, 255, 200)

    sign_open_k: Tuple[int, int] = (3, 3)
    sign_close_k: Tuple[int, int] = (7, 7)
    sign_dilate_k: Tuple[int, int] = (7, 7)
    sign_open_iter: int = 1
    sign_close_iter: int = 1
    sign_dilate_iter: int = 1

    min_sign_area_frac: float = 0.005
    edge_touch_margin_px: int = 6

    sign_aspect_min: float = 1.0
    sign_aspect_max: float = 2.8
    expected_sign_aspect: float = 1.6  # used as soft score term

    # Candidate scoring weights
    score_w_area: float = 3.0
    score_w_extent: float = 0.8
    score_w_aspect: float = 0.8
    score_w_solidity: float = 0.4
    score_w_mask_fill: float = 0.6
    score_w_downstream: float = 2.0

    # Blue text mask (more permissive to preserve thin strokes like '0')
    text_blue_lower: Tuple[int, int, int] = (90, 40, 35)
    text_blue_upper: Tuple[int, int, int] = (150, 255, 255)

    text_median_blur: int = 3
    text_open_k: Tuple[int, int] = (2, 2)
    text_open_iter: int = 1
    text_close_k: Tuple[int, int] = (3, 3)
    text_close_iter: int = 1

    # Inner face detection: border projection
    border_ratio_rel_peak: float = 0.65
    border_ratio_abs_min: float = 0.25
    border_streak: int = 3
    face_min_area_frac: float = 0.22
    face_max_area_frac: float = 0.92
    face_pad_px: int = 2
    face_fallback_inset_frac: float = 0.12

    # Grey HSV fallback
    grey_s_max: int = 90
    grey_v_min: int = 40
    grey_v_max: int = 245
    grey_close_k: Tuple[int, int] = (7, 7)
    grey_close_iter: int = 2
    grey_open_k: Tuple[int, int] = (3, 3)
    grey_open_iter: int = 1

    # Line finding (row projection)
    row_smooth_win: int = 9
    row_ratio_abs_min: float = 0.01
    row_ratio_rel_peak: float = 0.20
    min_band_h_px: int = 6
    line_pad_x_frac: float = 0.08
    line_pad_y_frac: float = 0.25

    # Conservative fallback line rects inside face (relative)
    fallback_top_rect: Tuple[float, float, float, float] = (0.30, 0.05, 0.92, 0.40)  # x0,y0,x1,y1
    fallback_bot_rect: Tuple[float, float, float, float] = (0.05, 0.55, 0.92, 0.90)

    # Edge strip removal
    edge_strip_ratio_thresh: float = 0.65
    edge_strip_streak: int = 3
    edge_strip_max_frac: float = 0.12

    # Character segmentation
    proj_col_thresh: float = 0.05
    proj_min_w_px: int = 3
    proj_min_h_frac: float = 0.30
    proj_max_w_frac: float = 0.45

    contour_min_area: int = 20
    contour_fill_min: float = 0.08

    # Merged-letter splitting
    merge_w_factor: float = 1.45
    split_search_l_frac: float = 0.22
    split_search_r_frac: float = 0.78
    valley_rel: float = 0.45

    char_pad_frac: float = 0.12


P = Params()


# =========================
# Helper functions
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smooth1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def closest_valid_top(raw: str) -> str:
    raw = (raw or "").upper()
    best = ""
    best_d = 10**9
    for w in P.valid_top_words:
        d = levenshtein(raw, w)
        if d < best_d:
            best_d = d
            best = w
    return best


def build_vis_strip(processed_32: List[np.ndarray]) -> Optional[np.ndarray]:
    if not processed_32:
        return None
    gap = 6
    h = 32
    w = len(processed_32) * 32 + (len(processed_32) - 1) * gap
    canvas = np.full((h, w), 127, dtype=np.uint8)
    x = 0
    for img in processed_32:
        if img.shape != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        canvas[:, x:x + 32] = img
        x += 32 + gap
    return canvas


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    rect = order_points(pts)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def hsv_inrange(bgr: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def clamp_rect(r: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = r
    x0 = int(max(0, min(W - 1, x0)))
    y0 = int(max(0, min(H - 1, y0)))
    x1 = int(max(x0 + 1, min(W, x1)))
    y1 = int(max(y0 + 1, min(H, y1)))
    return x0, y0, x1, y1


def rect_area(r: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = r
    return max(0, x1 - x0) * max(0, y1 - y0)


def crop_rect(img: np.ndarray, r: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = r
    return img[y0:y1, x0:x1]


def contour_touches_edge(cnt: np.ndarray, W: int, H: int, margin: int) -> bool:
    x, y, w, h = cv2.boundingRect(cnt)
    return x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin)


# =========================
# CNN preprocessing (IDENTICAL to training)
# =========================

def preprocess_char_for_cnn(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      x: (1,32,32,1) float32 in [0,1]
      vis: (32,32) uint8 preview

    Steps (must match training):
    grayscale -> equalizeHist -> GaussianBlur(3x3) -> Otsu inverse threshold
    -> square pad (side=max(h,w)+10) -> resize 40 -> resize 32
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape[:2]
    side = max(h, w) + 10
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = bw

    canvas = cv2.resize(canvas, (40, 40), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)

    x = canvas.astype(np.float32) / 255.0
    x = x.reshape(1, 32, 32, 1)
    return x, canvas


# =========================
# Sign candidate detection
# =========================

def morph_sign_mask(mask: np.ndarray) -> np.ndarray:
    out = mask
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones(P.sign_close_k, np.uint8), iterations=P.sign_close_iter)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones(P.sign_open_k, np.uint8), iterations=P.sign_open_iter)
    if P.sign_dilate_iter > 0:
        out = cv2.dilate(out, np.ones(P.sign_dilate_k, np.uint8), iterations=P.sign_dilate_iter)
    return out


def approx_quad(cnt: np.ndarray) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect).astype(int)


def score_sign_contour(cnt: np.ndarray, frame_area: float, sign_mask: np.ndarray) -> float:
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return -1e9

    x, y, w, h = cv2.boundingRect(cnt)
    rect_a = float(w * h) + 1e-6

    extent = area / rect_a
    aspect = float(w) / float(h + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = area / hull_area

    roi = sign_mask[y:y + h, x:x + w]
    fill = float(np.count_nonzero(roi)) / float(roi.size + 1e-6)

    area_frac = area / (frame_area + 1e-6)
    aspect_score = 1.0 - min(1.0, abs(aspect - P.expected_sign_aspect) / (P.expected_sign_aspect + 1e-6))

    return (
        P.score_w_area * area_frac +
        P.score_w_extent * extent +
        P.score_w_aspect * aspect_score +
        P.score_w_solidity * solidity +
        P.score_w_mask_fill * fill
    )


# =========================
# Inner face detection
# =========================

def scan_inner_boundary(ratio: np.ndarray, thr: float, from_left: bool, streak: int) -> Optional[int]:
    n = len(ratio)
    below = 0
    idxs = range(n) if from_left else range(n - 1, -1, -1)
    for i in idxs:
        if ratio[i] < thr:
            below += 1
        else:
            below = 0
        if below >= streak:
            if from_left:
                return max(0, i - streak + 1)
            return min(n - 1, i + streak - 1)
    return None


def find_inner_face_by_border(warped_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    H, W = warped_bgr.shape[:2]
    blue = hsv_inrange(warped_bgr, P.text_blue_lower, P.text_blue_upper)
    if P.text_median_blur >= 3:
        blue = cv2.medianBlur(blue, P.text_median_blur)

    col_ratio = np.sum(blue > 0, axis=0).astype(np.float32) / float(H + 1e-6)
    row_ratio = np.sum(blue > 0, axis=1).astype(np.float32) / float(W + 1e-6)

    col_s = smooth1d(col_ratio, 7)
    row_s = smooth1d(row_ratio, 7)

    col_peak = float(np.max(col_s)) if col_s.size else 0.0
    row_peak = float(np.max(row_s)) if row_s.size else 0.0

    if col_peak < 0.10 or row_peak < 0.10:
        return None, blue

    col_thr = max(P.border_ratio_abs_min, P.border_ratio_rel_peak * col_peak)
    row_thr = max(P.border_ratio_abs_min, P.border_ratio_rel_peak * row_peak)

    left = scan_inner_boundary(col_s, col_thr, True, P.border_streak)
    right = scan_inner_boundary(col_s, col_thr, False, P.border_streak)
    top = scan_inner_boundary(row_s, row_thr, True, P.border_streak)
    bottom = scan_inner_boundary(row_s, row_thr, False, P.border_streak)

    if left is None or right is None or top is None or bottom is None:
        return None, blue

    r = clamp_rect((int(left), int(top), int(right + 1), int(bottom + 1)), W, H)
    r = (r[0] + P.face_pad_px, r[1] + P.face_pad_px, r[2] - P.face_pad_px, r[3] - P.face_pad_px)
    r = clamp_rect(r, W, H)

    frac = rect_area(r) / float(W * H + 1e-6)
    if not (P.face_min_area_frac <= frac <= P.face_max_area_frac):
        return None, blue

    return r, blue


def find_inner_face_by_grey(warped_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    H, W = warped_bgr.shape[:2]
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, P.grey_v_min], dtype=np.uint8)
    upper = np.array([180, P.grey_s_max, P.grey_v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(P.grey_close_k, np.uint8), iterations=P.grey_close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(P.grey_open_k, np.uint8), iterations=P.grey_open_iter)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    best = None
    best_a = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = float(w * h)
        if a < 0.10 * W * H:
            continue
        asp = float(w) / float(h + 1e-6)
        if not (1.1 <= asp <= 2.6):
            continue
        if a > best_a:
            best_a = a
            best = (x, y, x + w, y + h)

    if best is None:
        return None, mask

    r = clamp_rect(best, W, H)
    r = (r[0] + P.face_pad_px, r[1] + P.face_pad_px, r[2] - P.face_pad_px, r[3] - P.face_pad_px)
    r = clamp_rect(r, W, H)

    frac = rect_area(r) / float(W * H + 1e-6)
    if not (P.face_min_area_frac <= frac <= P.face_max_area_frac):
        return None, mask

    return r, mask


def find_inner_face(warped_bgr: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray, str]:
    H, W = warped_bgr.shape[:2]

    face_rect, face_mask = find_inner_face_by_border(warped_bgr)
    method = "border"
    if face_rect is None:
        face_rect, face_mask = find_inner_face_by_grey(warped_bgr)
        method = "grey"
    if face_rect is None:
        inset_x = int(P.face_fallback_inset_frac * W)
        inset_y = int(P.face_fallback_inset_frac * H)
        face_rect = clamp_rect((inset_x, inset_y, W - inset_x, H - inset_y), W, H)
        face_mask = hsv_inrange(warped_bgr, P.text_blue_lower, P.text_blue_upper)
        method = "fallback"

    dbg = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = face_rect
    cv2.rectangle(dbg, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    cv2.putText(dbg, f"face:{method}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return face_rect, face_mask, dbg, method


# =========================
# Line finding
# =========================

def find_text_bands(blue_mask: np.ndarray) -> List[Tuple[int, int]]:
    H, W = blue_mask.shape[:2]
    m = blue_mask.copy()
    if P.text_median_blur >= 3:
        m = cv2.medianBlur(m, P.text_median_blur)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(P.text_open_k, np.uint8), iterations=P.text_open_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones(P.text_close_k, np.uint8), iterations=P.text_close_iter)

    row_ratio = np.sum(m > 0, axis=1).astype(np.float32) / float(W + 1e-6)
    row_s = smooth1d(row_ratio, P.row_smooth_win)
    peak = float(np.max(row_s)) if row_s.size else 0.0
    thr = max(P.row_ratio_abs_min, P.row_ratio_rel_peak * peak)

    bands = []
    in_band = False
    start = 0
    for y, v in enumerate(row_s.tolist()):
        if v > thr and not in_band:
            start = y
            in_band = True
        elif v <= thr and in_band:
            end = y
            if end - start >= P.min_band_h_px:
                bands.append((start, end))
            in_band = False
    if in_band:
        end = H
        if end - start >= P.min_band_h_px:
            bands.append((start, end))
    return bands


def band_bbox(mask: np.ndarray, band: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    H, W = mask.shape[:2]
    y0, y1 = band
    roi = mask[y0:y1, :]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(np.min(xs))
    x1 = int(np.max(xs) + 1)
    yy0 = int(np.min(ys) + y0)
    yy1 = int(np.max(ys) + y0 + 1)
    return clamp_rect((x0, yy0, x1, yy1), W, H)


def pad_rect(r: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = r
    pad_x = int(P.line_pad_x_frac * (x1 - x0))
    pad_y = int(P.line_pad_y_frac * (y1 - y0))
    rr = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
    return clamp_rect(rr, W, H)


def compute_line_rects(face_bgr: np.ndarray) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], np.ndarray]:
    H, W = face_bgr.shape[:2]
    blue = hsv_inrange(face_bgr, P.text_blue_lower, P.text_blue_upper)

    bands = find_text_bands(blue)
    if len(bands) >= 2:
        top_bb = band_bbox(blue, bands[0])
        bot_bb = band_bbox(blue, bands[-1])
        if top_bb is not None and bot_bb is not None:
            return pad_rect(top_bb, W, H), pad_rect(bot_bb, W, H), blue

    fx0, fy0, fx1, fy1 = P.fallback_top_rect
    bx0, by0, bx1, by1 = P.fallback_bot_rect
    top = clamp_rect((int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H)), W, H)
    bot = clamp_rect((int(bx0 * W), int(by0 * H), int(bx1 * W), int(by1 * H)), W, H)
    return top, bot, blue


# =========================
# Mask cleanup + segmentation
# =========================

def clean_line_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if P.text_median_blur >= 3:
        m = cv2.medianBlur(m, P.text_median_blur)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(P.text_open_k, np.uint8), iterations=P.text_open_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones(P.text_close_k, np.uint8), iterations=P.text_close_iter)
    return m


def edge_strip_widths(mask: np.ndarray) -> Tuple[int, int]:
    H, W = mask.shape[:2]
    col_ratio = np.sum(mask > 0, axis=0).astype(np.float32) / float(H + 1e-6)
    max_strip = int(P.edge_strip_max_frac * W)

    def scan_left() -> int:
        streak = 0
        wcount = 0
        for i in range(W):
            if col_ratio[i] >= P.edge_strip_ratio_thresh:
                streak += 1
                wcount = i + 1
            else:
                streak = 0
            if streak >= P.edge_strip_streak and wcount <= max_strip:
                continue
            if wcount > 0 and streak == 0:
                break
        return min(wcount, max_strip)

    def scan_right() -> int:
        streak = 0
        wcount = 0
        for k, i in enumerate(range(W - 1, -1, -1)):
            if col_ratio[i] >= P.edge_strip_ratio_thresh:
                streak += 1
                wcount = k + 1
            else:
                streak = 0
            if streak >= P.edge_strip_streak and wcount <= max_strip:
                continue
            if wcount > 0 and streak == 0:
                break
        return min(wcount, max_strip)

    return scan_left(), scan_right()


def strip_edges_if_needed(mask: np.ndarray) -> np.ndarray:
    left_w, right_w = edge_strip_widths(mask)
    if left_w == 0 and right_w == 0:
        return mask
    m = mask.copy()
    H, W = m.shape[:2]
    if left_w > 0:
        m[:, :left_w] = 0
    if right_w > 0:
        m[:, W - right_w:] = 0
    return m


def projection_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = mask.shape[:2]
    col_sum = np.sum(mask > 0, axis=0).astype(np.float32)
    peak = float(np.max(col_sum)) if col_sum.size else 0.0
    if peak <= 0:
        return []
    col_norm = col_sum / (peak + 1e-6)
    thr = P.proj_col_thresh

    spans: List[Tuple[int, int]] = []
    in_char = False
    start = 0
    for x, v in enumerate(col_norm.tolist()):
        if v > thr and not in_char:
            start = x
            in_char = True
        elif v <= thr and in_char:
            spans.append((start, x))
            in_char = False
    if in_char:
        spans.append((start, W))

    boxes: List[Tuple[int, int, int, int]] = []
    for x0, x1 in spans:
        w = x1 - x0
        if w < P.proj_min_w_px:
            continue
        if w > P.proj_max_w_frac * W:
            continue
        roi = mask[:, x0:x1]
        rows = np.where(np.sum(roi > 0, axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        y0 = int(rows[0])
        y1 = int(rows[-1] + 1)
        h = y1 - y0
        if h < P.proj_min_h_frac * H:
            continue
        boxes.append((x0, y0, w, h))
    return boxes


def contour_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < P.contour_min_area:
            continue
        if h < P.proj_min_h_frac * H:
            continue
        if w > P.proj_max_w_frac * W:
            continue
        fill = float(cv2.contourArea(c)) / float(area + 1e-6)
        if fill < P.contour_fill_min:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: b[0])


def split_wide_box(mask: np.ndarray, box: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x, y, w, h = box
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0 or w < 8:
        return [box]
    col = np.sum(roi > 0, axis=0).astype(np.float32)
    peak = float(np.max(col)) if col.size else 0.0
    if peak <= 0:
        return [box]
    col = smooth1d(col, 5)

    l = max(2, int(P.split_search_l_frac * w))
    r = min(w - 2, int(P.split_search_r_frac * w))
    if r <= l:
        return [box]

    best_s = None
    best_val = None
    for s in range(l, r):
        valley = float(col[s])
        left_peak = float(np.max(col[:s])) if s > 0 else 0.0
        right_peak = float(np.max(col[s:])) if s < w else 0.0
        if left_peak < 1e-6 or right_peak < 1e-6:
            continue
        if valley <= P.valley_rel * min(left_peak, right_peak):
            if best_val is None or valley < best_val:
                best_val = valley
                best_s = s

    if best_s is None:
        return [box]

    s = int(best_s)
    left_roi = roi[:, :s]
    right_roi = roi[:, s:]

    def tight(sub: np.ndarray, x_off: int) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(sub > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        xx0 = int(np.min(xs))
        xx1 = int(np.max(xs) + 1)
        yy0 = int(np.min(ys))
        yy1 = int(np.max(ys) + 1)
        return (x + x_off + xx0, y + yy0, xx1 - xx0, yy1 - yy0)

    b1 = tight(left_roi, 0)
    b2 = tight(right_roi, s)

    out = []
    for b in (b1, b2):
        if b is None:
            continue
        bx, by, bw, bh = b
        if bw >= 2 and bh >= int(P.proj_min_h_frac * mask.shape[0]):
            out.append((bx, by, bw, bh))
    return out if len(out) >= 2 else [box]


def segment_char_boxes(line_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    m = clean_line_mask(line_mask)
    m = strip_edges_if_needed(m)

    boxes = projection_boxes(m)
    if not boxes:
        boxes = contour_boxes(m)
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    widths = np.array([b[2] for b in boxes], dtype=np.float32)
    median_w = float(np.median(widths)) if len(widths) else 0.0

    if median_w > 0:
        split = []
        for b in boxes:
            if b[2] > P.merge_w_factor * median_w:
                split.extend(split_wide_box(m, b))
            else:
                split.append(b)
        boxes = sorted(split, key=lambda b: b[0])

    return boxes


def segment_and_preprocess_line(line_bgr: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[Tuple[int, int, int, int]], List[np.ndarray]]:
    raw_mask = hsv_inrange(line_bgr, P.text_blue_lower, P.text_blue_upper)
    mask_used = clean_line_mask(raw_mask)
    mask_used = strip_edges_if_needed(mask_used)

    boxes = segment_char_boxes(mask_used)
    H, W = mask_used.shape[:2]

    crops_bgr: List[np.ndarray] = []
    processed_32: List[np.ndarray] = []

    for x, y, w, h in boxes:
        pad = max(2, int(P.char_pad_frac * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        crop = line_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops_bgr.append(crop)
        _, vis = preprocess_char_for_cnn(crop)
        processed_32.append(vis)

    dbg = line_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return crops_bgr, mask_used, dbg, boxes, processed_32


# =========================
# Temporal voting (PAIR vote)
# =========================

class TemporalVote:
    """
    Votes on (top_type, bottom_value) pairs.
    This prevents mismatched top from one frame and bottom from another.
    """
    def __init__(self, maxlen: int):
        self.hist: Deque[Tuple[str, str, float]] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self.hist.clear()

    def add(self, top: str, bottom: str, weight: float) -> None:
        if not top or not bottom:
            return
        self.hist.append((top, bottom, float(weight)))

    def best_pair(self) -> Tuple[str, str, float, int]:
        scores: Dict[Tuple[str, str], float] = defaultdict(float)
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for t, b, w in self.hist:
            key = (t, b)
            scores[key] += w
            counts[key] += 1
        if not scores:
            return "", "", 0.0, 0
        (bt, bb), s = max(scores.items(), key=lambda kv: kv[1])
        return bt, bb, float(s), int(counts[(bt, bb)])

    def __len__(self) -> int:
        return len(self.hist)


# =========================
# Score tracker client
# =========================

CLUE_ID_MAP = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8,
}


class ScoreTrackerClient:
    def __init__(self):
        self.pub = rospy.Publisher(P.score_tracker_topic, StringMsg, queue_size=1)
        self.last_pub_time = 0.0
        self.started = False
        self.submitted: Dict[int, str] = {}

        if P.auto_start_timer and P.enable_score_tracker_pub:
            # Competition notes recommend waiting ~1s after creating pubs/subs before sending messages.
            rospy.Timer(rospy.Duration(1.0), self._start_timer_once, oneshot=True)

        if P.auto_stop_timer_on_shutdown and P.enable_score_tracker_pub:
            rospy.on_shutdown(self.stop_timer)

    def _ok_team_fields(self) -> bool:
        if not P.team_id or not P.team_password:
            return False
        if " " in P.team_id or " " in P.team_password:
            return False
        if len(P.team_id) > 8 or len(P.team_password) > 8:
            return False
        return True

    def _can_pub(self) -> bool:
        if not P.enable_score_tracker_pub:
            return False
        if not self._ok_team_fields():
            rospy.logwarn_throttle(5.0, "ScoreTracker disabled: team_id/password invalid (must be <=8 chars, no spaces).")
            return False
        now = time.time()
        if now - self.last_pub_time < P.min_score_pub_interval_sec:
            return False
        self.last_pub_time = now
        return True

    def _start_timer_once(self, _evt) -> None:
        if self.started:
            return
        self.start_timer()

    def start_timer(self) -> None:
        if self.started:
            return
        if not self._can_pub():
            return
        msg = f"{P.team_id},{P.team_password},0,NA"
        self.pub.publish(StringMsg(data=msg))
        self.started = True
        rospy.loginfo(f"ScoreTracker: start timer -> {msg}")

    def stop_timer(self) -> None:
        if not self._can_pub():
            return
        msg = f"{P.team_id},{P.team_password},-1,NA"
        self.pub.publish(StringMsg(data=msg))
        rospy.loginfo(f"ScoreTracker: stop timer -> {msg}")

    def publish_clue(self, clue_type: str, clue_value: str) -> None:
        clue_type = (clue_type or "").upper()
        if clue_type not in CLUE_ID_MAP:
            return

        clue_id = CLUE_ID_MAP[clue_type]

        value = (clue_value or "").upper().replace(" ", "")
        if not value:
            return

        if P.submit_only_once_per_clue:
            prev = self.submitted.get(clue_id, None)
            if prev == value:
                return

        if not self._can_pub():
            return

        msg = f"{P.team_id},{P.team_password},{clue_id},{value}"
        self.pub.publish(StringMsg(data=msg))
        self.submitted[clue_id] = value
        rospy.loginfo(f"ScoreTracker: submit -> {msg}")


# =========================
# Stable GUI (no flicker): GUI thread
# =========================

class GUIThread:
    def __init__(self):
        self.enabled = P.enable_gui
        self.lock = threading.Lock()
        self.main_view: Optional[np.ndarray] = None
        self.warp_view: Optional[np.ndarray] = None
        self._stop = False
        self._fps_last = time.time()
        self._fps_count = 0
        self.fps = 0.0

        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def update(self, main_view: np.ndarray, warp_view: Optional[np.ndarray]) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.main_view = main_view
            self.warp_view = warp_view

    def stop(self) -> None:
        self._stop = True
        self.enabled = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _run(self) -> None:
        # NOTE: cv2.imshow / waitKey are not thread-safe with arbitrary threading in all environments,
        # but this pattern is much more stable than calling imshow directly inside ROS callbacks.
        try:
            cv2.namedWindow(P.gui_main_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(P.gui_warp_window, cv2.WINDOW_NORMAL)
        except Exception:
            self.enabled = False
            return

        rate = 1.0 / max(1.0, P.gui_hz)

        while not rospy.is_shutdown() and not self._stop and self.enabled:
            t0 = time.time()

            with self.lock:
                mv = None if self.main_view is None else self.main_view.copy()
                wv = None if self.warp_view is None else self.warp_view.copy()

            if mv is not None:
                try:
                    cv2.imshow(P.gui_main_window, mv)
                    if wv is not None:
                        cv2.imshow(P.gui_warp_window, wv)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        # close windows but keep node running
                        self.stop()
                        break
                except Exception:
                    self.enabled = False
                    break

            # FPS calculation (GUI loop FPS)
            self._fps_count += 1
            now = time.time()
            if now - self._fps_last >= 1.0:
                self.fps = self._fps_count / (now - self._fps_last)
                self._fps_count = 0
                self._fps_last = now

            dt = time.time() - t0
            sleep_t = max(0.0, rate - dt)
            time.sleep(sleep_t)


def safe_text(s: str) -> str:
    return (s or "").replace(" ", "␠")


def draw_conf_bar(img: np.ndarray, x: int, y: int, w: int, h: int, conf: float, label: str) -> None:
    conf = float(max(0.0, min(1.0, conf)))
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    fill_w = int(w * conf)
    if conf >= 0.80:
        color = (0, 200, 0)
    elif conf >= 0.55:
        color = (0, 180, 220)
    else:
        color = (0, 80, 220)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(img, f"{label}: {conf:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def make_main_gui(frame_bgr: np.ndarray,
                  quad: Optional[List[List[int]]],
                  status: str,
                  top_raw: str,
                  bottom_raw: str,
                  top_mean: float,
                  bottom_mean: float,
                  top_corr: str,
                  vote_type: str,
                  vote_val: str,
                  vote_score: float,
                  vote_count: int,
                  candidate_score: float,
                  face_method: str,
                  gui_fps: float) -> np.ndarray:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    if quad is not None and len(quad) == 4:
        cv2.polylines(frame, [np.array(quad, dtype=np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

    panel_w = P.gui_panel_w
    out = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    out[:, :w] = frame
    out[:, w:] = (24, 24, 24)

    px = w + 18
    y = 30

    cv2.putText(out, "SIGN DETECTOR", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30
    cv2.putText(out, f"STATUS: {status}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
    y += 24
    cv2.putText(out, f"cand={candidate_score:.2f} face={face_method}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 24
    cv2.putText(out, f"GUI FPS: {gui_fps:.1f}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26

    cv2.line(out, (px, y), (w + panel_w - 18, y), (80, 80, 80), 1)
    y += 28

    cv2.putText(out, "CURRENT OCR", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 34

    cv2.putText(out, f"Top RAW:  {safe_text(top_raw)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 120), 2, cv2.LINE_AA)
    y += 30
    cv2.putText(out, f"Top CORR: {safe_text(top_corr)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 190, 90), 2, cv2.LINE_AA)
    y += 34
    draw_conf_bar(out, px, y, 260, 16, top_mean, "Top mean")
    y += 42

    cv2.putText(out, f"Bottom:   {safe_text(bottom_raw)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2, cv2.LINE_AA)
    y += 34
    draw_conf_bar(out, px, y, 260, 16, bottom_mean, "Bottom mean")
    y += 46

    cv2.line(out, (px, y), (w + panel_w - 18, y), (80, 80, 80), 1)
    y += 28

    cv2.putText(out, "TEMPORAL VOTE (PAIR)", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 34

    cv2.putText(out, f"Type:  {safe_text(vote_type)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 120), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(out, f"Value: {safe_text(vote_val)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(out, f"Score: {vote_score:.2f}  Count: {vote_count}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(out, "Press q in GUI window to close windows", (px, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return out


def make_warp_gui(warped_bgr: np.ndarray,
                  face_rect: Tuple[int, int, int, int],
                  top_rect_w: Tuple[int, int, int, int],
                  bot_rect_w: Tuple[int, int, int, int],
                  top_boxes: List[Tuple[int, int, int, int]],
                  bot_boxes: List[Tuple[int, int, int, int]],
                  top_text: str,
                  bottom_text: str) -> np.ndarray:
    vis = warped_bgr.copy()

    fx0, fy0, fx1, fy1 = face_rect
    cv2.rectangle(vis, (fx0, fy0), (fx1 - 1, fy1 - 1), (0, 255, 255), 2)

    tx0, ty0, tx1, ty1 = top_rect_w
    bx0, by0, bx1, by1 = bot_rect_w
    cv2.rectangle(vis, (tx0, ty0), (tx1 - 1, ty1 - 1), (255, 100, 0), 2)
    cv2.rectangle(vis, (bx0, by0), (bx1 - 1, by1 - 1), (0, 100, 255), 2)

    for (x, y, w, h) in top_boxes:
        cv2.rectangle(vis, (tx0 + x, ty0 + y), (tx0 + x + w, ty0 + y + h), (0, 255, 0), 2)
    for (x, y, w, h) in bot_boxes:
        cv2.rectangle(vis, (bx0 + x, by0 + y), (bx0 + x + w, by0 + y + h), (0, 255, 0), 2)

    cv2.putText(vis, f"Top={safe_text(top_text)}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Bottom={safe_text(bottom_text)}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


# =========================
# Sample buffer (for dump)
# =========================

@dataclass
class Sample:
    t_sec: float
    quad: Optional[List[List[int]]]
    candidate_score: float
    face_method: str
    top_raw: str
    bottom_raw: str
    top_confs: List[float]
    bottom_confs: List[float]
    top_mean: float
    bottom_mean: float
    top_corrected: str
    voted_type: str
    voted_value: str
    voted_score: float
    voted_count: int
    frame_bgr: Optional[np.ndarray]
    warped_bgr: Optional[np.ndarray]
    face_crop_bgr: Optional[np.ndarray]
    top_line_bgr: Optional[np.ndarray]
    bottom_line_bgr: Optional[np.ndarray]


# =========================
# Main node
# =========================

class SignDetectorNode:
    def __init__(self):
        ensure_dir(P.debug_dir)
        ensure_dir(P.dump_dir)

        self.bridge = CvBridge()

        self.classes = self._load_classes(P.classes_path)
        self.interpreter, self.model_path = self._load_interpreter()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        rospy.loginfo(f"{P.node_name}: loaded model: {self.model_path}")
        rospy.loginfo(f"{P.node_name}: loaded classes: {P.classes_path} (n={len(self.classes)})")

        self.vote = TemporalVote(P.vote_window)
        self.miss_streak = 0

        self.samples: Deque[Sample] = deque(maxlen=P.sample_buffer_len)

        self.last_status: Dict = {}
        self.status_pub = rospy.Publisher(P.status_topic, StringMsg, queue_size=5)
        self.status_timer = rospy.Timer(rospy.Duration(1.0 / max(0.1, P.status_rate_hz)), self._publish_status_timer)

        self.score = ScoreTrackerClient()

        self.dump_srv = rospy.Service(P.dump_service_name, Trigger, self._handle_dump_service)
        self.dump_sub = rospy.Subscriber(P.dump_trigger_topic, EmptyMsg, self._handle_dump_trigger, queue_size=1)

        self.gui = GUIThread()

        self.last_debug_write = 0.0

        rospy.Subscriber(P.image_topic, Image, self.callback, queue_size=1)

    # ---------- model ----------
    @staticmethod
    def _load_classes(path: str) -> List[str]:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = sorted(data.keys(), key=lambda k: int(k))
            return [data[k] for k in keys]
        return list(data)

    @staticmethod
    def _load_interpreter() -> Tuple[tf.lite.Interpreter, str]:
        model_path = P.model_path_finetuned if os.path.exists(P.model_path_finetuned) else P.model_path_base
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, model_path

    # ---------- status publish ----------
    def _publish_status_timer(self, _evt) -> None:
        if not self.last_status:
            return
        try:
            self.status_pub.publish(StringMsg(data=json.dumps(self.last_status)))
        except Exception:
            pass

    # ---------- dump ----------
    def _handle_dump_trigger(self, _msg: EmptyMsg) -> None:
        self._dump_samples_to_disk()

    def _handle_dump_service(self, _req) -> TriggerResponse:
        path = self._dump_samples_to_disk()
        resp = TriggerResponse()
        resp.success = True
        resp.message = f"dumped to {path}"
        return resp

    def _dump_samples_to_disk(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(P.dump_dir, f"dump_{ts}")
        ensure_dir(out_dir)

        index_rows = []
        for i, s in enumerate(list(self.samples)):
            sid = f"{i:04d}"
            sd = os.path.join(out_dir, sid)
            ensure_dir(sd)

            def imwrite(name: str, img: Optional[np.ndarray]) -> None:
                if img is None:
                    return
                cv2.imwrite(os.path.join(sd, name), img)

            imwrite("frame.jpg", s.frame_bgr)
            imwrite("warped.jpg", s.warped_bgr)
            imwrite("face_crop.jpg", s.face_crop_bgr)
            imwrite("top_line.jpg", s.top_line_bgr)
            imwrite("bottom_line.jpg", s.bottom_line_bgr)

            meta = {
                "t_sec": s.t_sec,
                "quad": s.quad,
                "candidate_score": s.candidate_score,
                "face_method": s.face_method,
                "top_raw": s.top_raw,
                "bottom_raw": s.bottom_raw,
                "top_confs": s.top_confs,
                "bottom_confs": s.bottom_confs,
                "top_mean": s.top_mean,
                "bottom_mean": s.bottom_mean,
                "top_corrected": s.top_corrected,
                "voted_type": s.voted_type,
                "voted_value": s.voted_value,
                "voted_score": s.voted_score,
                "voted_count": s.voted_count,
                "model_path": self.model_path,
            }
            with open(os.path.join(sd, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            index_rows.append(meta)

        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump(index_rows, f, indent=2)

        rospy.loginfo(f"{P.node_name}: dumped {len(index_rows)} samples -> {out_dir}")
        return out_dir

    # ---------- inference ----------
    def predict_char(self, crop_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        x, vis = preprocess_char_for_cnn(crop_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        idx = int(np.argmax(out))
        conf = float(out[idx])
        ch = self.classes[idx] if 0 <= idx < len(self.classes) else "?"
        return ch, conf, vis

    def predict_string(self, crops_bgr: List[np.ndarray]) -> Tuple[str, List[float], List[Tuple[str, float]], List[np.ndarray]]:
        chars: List[str] = []
        confs: List[float] = []
        details: List[Tuple[str, float]] = []
        vis32: List[np.ndarray] = []
        for crop in crops_bgr:
            ch, conf, vis = self.predict_char(crop)
            chars.append(ch)
            confs.append(conf)
            details.append((ch, conf))
            vis32.append(vis)
        return "".join(chars), confs, details, vis32

    # ---------- sign detection ----------
    def detect_best_sign(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Optional[List[List[int]]], float]:
        H, W = frame_bgr.shape[:2]
        frame_area = float(H * W)

        sign_mask = hsv_inrange(frame_bgr, P.sign_blue_lower, P.sign_blue_upper)
        sign_mask = morph_sign_mask(sign_mask)

        cnts, _ = cv2.findContours(sign_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dbg = frame_bgr.copy()

        if not cnts:
            cv2.putText(dbg, "no contours", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return None, sign_mask, dbg, None, -1e9

        candidates: List[Tuple[float, np.ndarray]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < P.min_sign_area_frac * frame_area:
                continue
            if contour_touches_edge(c, W, H, P.edge_touch_margin_px):
                continue
            x, y, w, h = cv2.boundingRect(c)
            asp = float(w) / float(h + 1e-6)
            if not (P.sign_aspect_min <= asp <= P.sign_aspect_max):
                continue
            base = score_sign_contour(c, frame_area, sign_mask)
            candidates.append((base, c))

        if not candidates:
            # fallback: best area contour
            c = max(cnts, key=cv2.contourArea)
            candidates = [(score_sign_contour(c, frame_area, sign_mask), c)]

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:10]

        best_score = -1e18
        best_warp = None
        best_quad = None

        for base_score, c in candidates:
            quad = approx_quad(c)
            warped = four_point_transform(frame_bgr, quad, P.warp_w, P.warp_h)

            # downstream heuristic: if face/lines plausibly exist, reward
            downstream = 0.0
            try:
                face_rect, _, _, method = find_inner_face(warped)
                if face_rect is not None:
                    downstream += 1.0
                    face = crop_rect(warped, face_rect)
                    tr, br, _ = compute_line_rects(face)
                    if rect_area(tr) > 0 and rect_area(br) > 0:
                        downstream += 1.0
            except Exception:
                downstream = 0.0

            score = base_score + P.score_w_downstream * downstream
            if score > best_score:
                best_score = score
                best_warp = warped
                best_quad = quad

        quad_list = None
        if best_quad is not None:
            quad_list = [[int(p[0]), int(p[1])] for p in best_quad]
            cv2.drawContours(dbg, [np.array(best_quad, dtype=np.int32)], -1, (0, 255, 0), 3)
            cv2.putText(dbg, f"best={best_score:.2f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return best_warp, sign_mask, dbg, quad_list, float(best_score)

    # ---------- debug writing ----------
    def save_debug_bundle(self,
                          frame: np.ndarray,
                          sign_mask: np.ndarray,
                          sign_debug: np.ndarray,
                          warped: Optional[np.ndarray],
                          face_mask: Optional[np.ndarray],
                          face_debug: Optional[np.ndarray],
                          face_crop: Optional[np.ndarray],
                          top_line: Optional[np.ndarray],
                          bottom_line: Optional[np.ndarray],
                          top_char_mask: Optional[np.ndarray],
                          bottom_char_mask: Optional[np.ndarray],
                          top_char_debug: Optional[np.ndarray],
                          bottom_char_debug: Optional[np.ndarray],
                          top_vis: Optional[np.ndarray],
                          bottom_vis: Optional[np.ndarray]) -> None:
        ensure_dir(P.debug_dir)

        cv2.imwrite(os.path.join(P.debug_dir, "frame.jpg"), frame)
        cv2.imwrite(os.path.join(P.debug_dir, "sign_mask.jpg"), sign_mask)
        cv2.imwrite(os.path.join(P.debug_dir, "sign_debug.jpg"), sign_debug)

        if warped is None:
            warped_img = np.zeros((P.warp_h, P.warp_w, 3), dtype=np.uint8)
        else:
            warped_img = warped
        cv2.imwrite(os.path.join(P.debug_dir, "warped.jpg"), warped_img)

        if face_mask is None:
            face_mask_img = np.zeros((P.warp_h, P.warp_w), dtype=np.uint8)
        else:
            face_mask_img = face_mask
        cv2.imwrite(os.path.join(P.debug_dir, "face_mask.jpg"), face_mask_img)

        if face_debug is None:
            face_debug_img = warped_img.copy()
        else:
            face_debug_img = face_debug
        cv2.imwrite(os.path.join(P.debug_dir, "face_debug.jpg"), face_debug_img)

        if face_crop is None:
            face_crop_img = warped_img.copy()
        else:
            face_crop_img = face_crop
        cv2.imwrite(os.path.join(P.debug_dir, "face_crop.jpg"), face_crop_img)

        cv2.imwrite(os.path.join(P.debug_dir, "top_line.jpg"),
                    top_line if top_line is not None else np.zeros((80, 240, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(P.debug_dir, "bottom_line.jpg"),
                    bottom_line if bottom_line is not None else np.zeros((80, 240, 3), dtype=np.uint8))

        cv2.imwrite(os.path.join(P.debug_dir, "top_char_mask.jpg"),
                    top_char_mask if top_char_mask is not None else np.zeros((80, 240), dtype=np.uint8))
        cv2.imwrite(os.path.join(P.debug_dir, "bottom_char_mask.jpg"),
                    bottom_char_mask if bottom_char_mask is not None else np.zeros((80, 240), dtype=np.uint8))

        cv2.imwrite(os.path.join(P.debug_dir, "top_char_debug.jpg"),
                    top_char_debug if top_char_debug is not None else np.zeros((80, 240, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(P.debug_dir, "bottom_char_debug.jpg"),
                    bottom_char_debug if bottom_char_debug is not None else np.zeros((80, 240, 3), dtype=np.uint8))

        cv2.imwrite(os.path.join(P.debug_dir, "top_chars_vis.jpg"),
                    top_vis if top_vis is not None else np.zeros((32, 32), dtype=np.uint8))
        cv2.imwrite(os.path.join(P.debug_dir, "bottom_chars_vis.jpg"),
                    bottom_vis if bottom_vis is not None else np.zeros((32, 32), dtype=np.uint8))

    # ---------- main callback ----------
    def callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"{P.node_name}: CvBridge error: {e}")
            return

        warped, sign_mask, sign_dbg, quad_pts, cand_score = self.detect_best_sign(frame)

        # Defaults for GUI/status/debug
        status = "NO SIGN"
        face_method = ""
        face_rect = (0, 0, P.warp_w, P.warp_h)
        top_r = (0, 0, 1, 1)
        bot_r = (0, 0, 1, 1)
        top_boxes: List[Tuple[int, int, int, int]] = []
        bot_boxes: List[Tuple[int, int, int, int]] = []
        top_raw = ""
        bot_raw = ""
        top_confs: List[float] = []
        bot_confs: List[float] = []
        top_mean = 0.0
        bot_mean = 0.0
        top_corr = ""
        vote_type, vote_val, vote_score, vote_count = self.vote.best_pair()

        face_mask = None
        face_dbg = None
        face_crop = None
        top_line = None
        bottom_line = None
        top_char_mask = None
        bottom_char_mask = None
        top_char_dbg = None
        bottom_char_dbg = None
        top_vis = None
        bottom_vis = None
        warp_view = None

        if warped is None:
            self.miss_streak += 1
            if self.miss_streak >= P.clear_vote_after_misses:
                self.vote.clear()
            vote_type, vote_val, vote_score, vote_count = self.vote.best_pair()

            # Debug write
            if time.time() - self.last_debug_write >= P.debug_save_every_sec:
                self.save_debug_bundle(
                    frame=frame,
                    sign_mask=sign_mask,
                    sign_debug=sign_dbg,
                    warped=None,
                    face_mask=None,
                    face_debug=None,
                    face_crop=None,
                    top_line=None,
                    bottom_line=None,
                    top_char_mask=None,
                    bottom_char_mask=None,
                    top_char_debug=None,
                    bottom_char_debug=None,
                    top_vis=None,
                    bottom_vis=None
                )
                self.last_debug_write = time.time()

            # GUI update
            if self.gui.enabled:
                main_view = make_main_gui(
                    frame_bgr=frame,
                    quad=quad_pts,
                    status=status,
                    top_raw=top_raw,
                    bottom_raw=bot_raw,
                    top_mean=top_mean,
                    bottom_mean=bot_mean,
                    top_corr=top_corr,
                    vote_type=vote_type,
                    vote_val=vote_val,
                    vote_score=vote_score,
                    vote_count=vote_count,
                    candidate_score=cand_score,
                    face_method=face_method,
                    gui_fps=self.gui.fps
                )
                self.gui.update(main_view, None)

            self.last_status = {
                "ok": False,
                "reason": "no_sign",
                "t": rospy.Time.now().to_sec(),
                "cand_score": cand_score,
                "vote_type": vote_type,
                "vote_value": vote_val,
                "vote_score": vote_score,
                "vote_count": vote_count,
            }
            rospy.loginfo_throttle(1.0, f"{P.node_name}: no sign detected")
            return

        # We have a warped sign
        self.miss_streak = 0

        face_rect, face_mask, face_dbg, face_method = find_inner_face(warped)
        face_crop = crop_rect(warped, face_rect)

        top_r, bot_r, _ = compute_line_rects(face_crop)
        top_line = crop_rect(face_crop, top_r)
        bottom_line = crop_rect(face_crop, bot_r)

        top_crops_bgr, top_char_mask, top_char_dbg, top_boxes, top_processed = segment_and_preprocess_line(top_line)
        bot_crops_bgr, bottom_char_mask, bottom_char_dbg, bot_boxes, bot_processed = segment_and_preprocess_line(bottom_line)

        top_vis = build_vis_strip(top_processed)
        bottom_vis = build_vis_strip(bot_processed)

        if len(top_crops_bgr) == 0 or len(bot_crops_bgr) == 0:
            status = "NO CHARS"
            if self.miss_streak >= P.clear_vote_after_misses:
                self.vote.clear()
            vote_type, vote_val, vote_score, vote_count = self.vote.best_pair()
        else:
            status = "TRACKING"

            top_raw, top_confs, top_details, _ = self.predict_string(top_crops_bgr)
            bot_raw, bot_confs, bot_details, _ = self.predict_string(bot_crops_bgr)

            top_mean = float(np.mean(top_confs)) if top_confs else 0.0
            bot_mean = float(np.mean(bot_confs)) if bot_confs else 0.0

            top_corr = closest_valid_top(top_raw)
            bot_corr = bot_raw

            # Vote gating (less brittle than "all chars >= threshold")
            def conf_frac_ok(confs: List[float]) -> bool:
                if not confs:
                    return False
                good = sum(1 for c in confs if c >= P.char_conf_threshold)
                return (good / float(len(confs))) >= P.min_char_conf_frac

            conf_ok = (
                top_mean >= P.word_mean_conf_threshold and
                bot_mean >= P.word_mean_conf_threshold and
                conf_frac_ok(top_confs) and
                conf_frac_ok(bot_confs)
            )

            # Basic plausibility
            counts_ok = (top_corr in P.valid_top_words) and (3 <= len(bot_raw) <= 20)

            if conf_ok and counts_ok:
                # weight = sum of confidences encourages longer confident strings
                weight = float(sum(top_confs) + sum(bot_confs))
                self.vote.add(top_corr, bot_corr, weight)

            vote_type, vote_val, vote_score, vote_count = self.vote.best_pair()

            rospy.loginfo_throttle(
                0.7,
                f"{P.node_name}: cand={cand_score:.2f} face={face_method} "
                f"RAW top={top_raw}({top_mean:.2f}) bot={bot_raw}({bot_mean:.2f}) "
                f"VOTE ({vote_type},{vote_val}) score={vote_score:.2f} count={vote_count}"
            )

            # Stable -> publish to score tracker
            stable = (vote_count >= P.vote_min_count) and (vote_score >= P.vote_min_pair_weight)
            if stable and vote_type and vote_val:
                self.score.publish_clue(vote_type, vote_val)

        # Compose warp overlay rects in warp coordinates
        fx0, fy0, _, _ = face_rect
        tx0, ty0, tx1, ty1 = top_r
        bx0, by0, bx1, by1 = bot_r
        top_w = (fx0 + tx0, fy0 + ty0, fx0 + tx1, fy0 + ty1)
        bot_w = (fx0 + bx0, fy0 + by0, fx0 + bx1, fy0 + by1)

        warp_view = make_warp_gui(
            warped_bgr=warped,
            face_rect=face_rect,
            top_rect_w=top_w,
            bot_rect_w=bot_w,
            top_boxes=top_boxes,
            bot_boxes=bot_boxes,
            top_text=top_raw,
            bottom_text=bot_raw
        )

        # Debug write periodically
        if time.time() - self.last_debug_write >= P.debug_save_every_sec:
            self.save_debug_bundle(
                frame=frame,
                sign_mask=sign_mask,
                sign_debug=sign_dbg,
                warped=warped,
                face_mask=face_mask,
                face_debug=face_dbg,
                face_crop=face_crop,
                top_line=top_line,
                bottom_line=bottom_line,
                top_char_mask=top_char_mask,
                bottom_char_mask=bottom_char_mask,
                top_char_debug=top_char_dbg,
                bottom_char_debug=bottom_char_dbg,
                top_vis=top_vis,
                bottom_vis=bottom_vis
            )
            self.last_debug_write = time.time()

        # GUI update (threaded)
        if self.gui.enabled:
            main_view = make_main_gui(
                frame_bgr=frame,
                quad=quad_pts,
                status=status,
                top_raw=top_raw,
                bottom_raw=bot_raw,
                top_mean=top_mean,
                bottom_mean=bot_mean,
                top_corr=top_corr,
                vote_type=vote_type,
                vote_val=vote_val,
                vote_score=vote_score,
                vote_count=vote_count,
                candidate_score=cand_score,
                face_method=face_method,
                gui_fps=self.gui.fps
            )
            self.gui.update(main_view, warp_view)

        # Update JSON status
        self.last_status = {
            "ok": True,
            "t": rospy.Time.now().to_sec(),
            "status": status,
            "cand_score": cand_score,
            "face_method": face_method,
            "quad": quad_pts,
            "top_raw": top_raw,
            "bottom_raw": bot_raw,
            "top_mean": top_mean,
            "bottom_mean": bot_mean,
            "top_corrected": top_corr,
            "vote_type": vote_type,
            "vote_value": vote_val,
            "vote_score": vote_score,
            "vote_count": vote_count,
            "top_boxes": len(top_boxes),
            "bottom_boxes": len(bot_boxes),
            "model_path": self.model_path,
        }

        # Sample buffer (for labelling)
        self.samples.append(Sample(
            t_sec=self.last_status["t"],
            quad=quad_pts,
            candidate_score=cand_score,
            face_method=face_method,
            top_raw=top_raw,
            bottom_raw=bot_raw,
            top_confs=top_confs,
            bottom_confs=bot_confs,
            top_mean=top_mean,
            bottom_mean=bot_mean,
            top_corrected=top_corr,
            voted_type=vote_type,
            voted_value=vote_val,
            voted_score=vote_score,
            voted_count=vote_count,
            frame_bgr=frame.copy(),
            warped_bgr=warped.copy(),
            face_crop_bgr=None if face_crop is None else face_crop.copy(),
            top_line_bgr=None if top_line is None else top_line.copy(),
            bottom_line_bgr=None if bottom_line is None else bottom_line.copy(),
        ))


def main() -> None:
    rospy.init_node(P.node_name, anonymous=False)
    SignDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()

