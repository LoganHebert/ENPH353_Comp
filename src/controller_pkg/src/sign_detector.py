#!/usr/bin/env python3
"""
FINAL sign_detector.py

Features:
- robust dark-blue sign detection
- dynamic inner-face and text-line extraction
- 0-friendly OCR pipeline
- stable GUI (persistent windows, no flicker loop)
- pair-based temporal voting (type + value together)
- submission gating:
    * reject low-confidence frames before voting
    * minimum frame agreement
    * vote-score threshold
    * reread-on-doubt logic
- seen-sign detection:
    * resets vote when a genuinely new sign appears
- single submission per clue
- score tracker publishing
- optional auto start / auto stop timer
- JSON status publisher for debugging
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


# =========================
# CONFIG
# =========================

@dataclass(frozen=True)
class Params:
    node_name: str = "sign_detector"

    # Use the topic that actually works in your sim.
    # Competition notes mention /B1/pi_camera/image_raw, but your current setup has been
    # using /B1/rrbot/camera1/image_raw successfully.
    image_topic: str = "/B1/rrbot/camera1/image_raw"

    model_path_finetuned: str = os.path.expanduser(
        "~/ros_ws/src/controller_pkg/models/sign_char_model_finetuned.tflite"
    )
    model_path_base: str = os.path.expanduser(
        "~/ros_ws/src/controller_pkg/models/sign_char_model.tflite"
    )
    classes_path: str = os.path.expanduser(
        "~/ros_ws/src/controller_pkg/models/sign_char_classes.json"
    )

    debug_dir: str = "/tmp/sign_debug"

    # -------- Competition publishing --------
    team_name: str = "FlyTeam"       # <= 8 chars, no spaces
    team_password: str = "password"   # <= 8 chars, no spaces
    score_topic: str = "/score_tracker"

    # Toggle auto start/stop here
    auto_timer_messages: bool = False # this is true will make it auto start/stop

    # wait ~1s after ROS publishers are created before sending messages
    auto_start_delay_sec: float = 1.2
    auto_stop_when_all_submitted: bool = True

    # -------- GUI --------
    enable_gui: bool = True
    gui_hz: float = 20.0
    gui_window_main: str = "Sign Detector GUI"
    gui_window_warp: str = "Warped Sign View"
    gui_panel_w: int = 420

    # -------- Status publishing --------
    status_topic: str = "/sign_detector/status_json"
    status_rate_hz: float = 5.0

    # -------- Warp --------
    warp_w: int = 400
    warp_h: int = 300

    # -------- Debug save --------
    debug_save_every_sec: float = 1.0

    # -------- Valid clue types --------
    valid_top_words: Tuple[str, ...] = (
        "SIZE", "VICTIM", "CRIME", "TIME",
        "PLACE", "MOTIVE", "WEAPON", "BANDIT"
    )

    # Competition mapping inferred from course convention/example
    clue_type_to_id_map: Dict[str, int] = None

    # -------- Seen sign detection --------
    new_sign_center_dist_norm: float = 0.22
    new_sign_area_ratio_low: float = 0.28
    new_sign_area_ratio_high: float = 3.8
    sign_anchor_alpha: float = 0.25
    sign_miss_reset_frames: int = 12

    # -------- Voting / stability --------
    vote_window: int = 20
    vote_min_entries: int = 5
    vote_min_score: float = 8.0

    # -------- Confidence thresholds --------
    # hard reject before voting
    char_conf_threshold: float = 0.55
    word_mean_conf_threshold: float = 0.60

    # reread zone: plausible but not good enough to vote/publish
    doubt_word_mean_threshold: float = 0.45
    doubt_char_conf_threshold: float = 0.35

    # -------- Sign detection --------
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
    expected_sign_aspect: float = 1.6

    score_w_area: float = 3.0
    score_w_extent: float = 0.8
    score_w_aspect: float = 0.8
    score_w_solidity: float = 0.4
    score_w_mask_fill: float = 0.6
    score_w_downstream: float = 2.0

    # -------- Blue text / border --------
    # permissive to preserve thin strokes and 0s
    text_blue_lower: Tuple[int, int, int] = (90, 40, 35)
    text_blue_upper: Tuple[int, int, int] = (150, 255, 255)

    text_median_blur: int = 3
    text_close_k: Tuple[int, int] = (3, 3)
    text_close_iter: int = 1
    text_open_k: Tuple[int, int] = (2, 2)
    text_open_iter: int = 1

    # -------- Inner face detection --------
    border_ratio_rel_peak: float = 0.65
    border_ratio_abs_min: float = 0.25
    border_streak: int = 3

    face_min_area_frac: float = 0.22
    face_max_area_frac: float = 0.92
    face_pad_px: int = 2
    face_fallback_inset_frac: float = 0.12

    grey_s_max: int = 90
    grey_v_min: int = 40
    grey_v_max: int = 245
    grey_close_k: Tuple[int, int] = (7, 7)
    grey_close_iter: int = 2
    grey_open_k: Tuple[int, int] = (3, 3)
    grey_open_iter: int = 1

    # -------- Line finding --------
    row_smooth_win: int = 9
    row_ratio_abs_min: float = 0.01
    row_ratio_rel_peak: float = 0.20
    min_band_h_px: int = 6

    line_pad_x_frac: float = 0.08
    line_pad_y_frac: float = 0.25

    fallback_top_rect: Tuple[float, float, float, float] = (0.30, 0.05, 0.92, 0.40)
    fallback_bot_rect: Tuple[float, float, float, float] = (0.05, 0.55, 0.92, 0.90)

    # -------- Edge strips --------
    edge_strip_ratio_thresh: float = 0.65
    edge_strip_streak: int = 3
    edge_strip_max_frac: float = 0.12

    # -------- Character segmentation --------
    proj_col_thresh: float = 0.05
    proj_min_w_px: int = 3
    proj_min_h_frac: float = 0.30
    proj_max_w_frac: float = 0.45

    contour_min_area: int = 20
    contour_fill_min: float = 0.08

    merge_w_factor: float = 1.45
    split_search_l_frac: float = 0.22
    split_search_r_frac: float = 0.78
    valley_rel: float = 0.45

    char_pad_frac: float = 0.12

    # -------- Submission cooldown --------
    submit_cooldown_sec: float = 0.20


DEFAULT_CLUE_MAP = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8,
}

P = Params(clue_type_to_id_map=DEFAULT_CLUE_MAP)


# =========================
# HELPERS
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


def safe_text(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace(" ", "␠")


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
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

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


def quad_center_and_area(quad_pts: Optional[List[List[int]]]) -> Tuple[Optional[np.ndarray], float]:
    if quad_pts is None or len(quad_pts) != 4:
        return None, 0.0
    q = np.array(quad_pts, dtype=np.float32)
    center = np.mean(q, axis=0)
    area = abs(cv2.contourArea(q.astype(np.int32)))
    return center, float(area)


# =========================
# CNN PREPROCESSING
# =========================

def preprocess_char_for_cnn(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Must match training preprocessing.
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
# SIGN DETECTION
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
# INNER FACE DETECTION
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
# LINE FINDING
# =========================

def find_text_bands(mask: np.ndarray) -> List[Tuple[int, int]]:
    H, W = mask.shape[:2]
    m = mask.copy()

    if P.text_median_blur >= 3:
        m = cv2.medianBlur(m, P.text_median_blur)

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones(P.text_close_k, np.uint8), iterations=P.text_close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(P.text_open_k, np.uint8), iterations=P.text_open_iter)

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
# SEGMENTATION
# =========================

def clean_line_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if P.text_median_blur >= 3:
        m = cv2.medianBlur(m, P.text_median_blur)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones(P.text_close_k, np.uint8), iterations=P.text_close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(P.text_open_k, np.uint8), iterations=P.text_open_iter)
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

    spans = []
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

    boxes = []
    for x0, x1 in spans:
        ww = x1 - x0
        if ww < P.proj_min_w_px:
            continue
        if ww > P.proj_max_w_frac * W:
            continue

        roi = mask[:, x0:x1]
        rows = np.where(np.sum(roi > 0, axis=1) > 0)[0]
        if len(rows) == 0:
            continue

        y0 = int(rows[0])
        y1 = int(rows[-1] + 1)
        hh = y1 - y0

        if hh < P.proj_min_h_frac * H:
            continue

        boxes.append((x0, y0, ww, hh))

    return boxes


def contour_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
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
    crops_bgr = []
    processed_32 = []

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


# =========================
# TEMPORAL VOTE
# =========================

class TemporalVote:
    """
    Pair-based vote on (top_type, bottom_value).
    This prevents mixing top from one frame and bottom from another.
    """
    def __init__(self, maxlen: int):
        self.hist: Deque[Tuple[str, str, float]] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self.hist.clear()

    def add(self, top: str, bottom: str, weight: float) -> None:
        if not top or not bottom:
            return
        self.hist.append((str(top), str(bottom), float(weight)))

    def best(self) -> Tuple[str, str, float, int]:
        scores: Dict[Tuple[str, str], float] = defaultdict(float)
        counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for t, b, w in self.hist:
            key = (t, b)
            scores[key] += w
            counts[key] += 1

        if not scores:
            return "", "", 0.0, 0

        (bt, bb), sc = max(scores.items(), key=lambda kv: kv[1])
        return bt, bb, float(sc), int(counts[(bt, bb)])

    def __len__(self) -> int:
        return len(self.hist)


# =========================
# GUI
# =========================

def draw_conf_bar(img: np.ndarray, x: int, y: int, w: int, h: int, conf: float, label: str) -> None:
    conf = max(0.0, min(1.0, float(conf)))
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


class StableGUI:
    """
    Dedicated GUI thread to avoid flicker.
    """
    def __init__(self):
        self.enabled = P.enable_gui
        self._stop = False
        self._lock = threading.Lock()
        self._main: Optional[np.ndarray] = None
        self._warp: Optional[np.ndarray] = None

        self.fps = 0.0
        self._fps_last = time.time()
        self._fps_count = 0

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def update(self, main_view: np.ndarray, warp_view: Optional[np.ndarray]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._main = main_view
            self._warp = warp_view

    def close(self) -> None:
        self._stop = True
        self.enabled = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _run(self) -> None:
        try:
            cv2.namedWindow(P.gui_window_main, cv2.WINDOW_NORMAL)
            cv2.namedWindow(P.gui_window_warp, cv2.WINDOW_NORMAL)
        except Exception:
            self.enabled = False
            return

        period = 1.0 / max(1.0, P.gui_hz)

        while not rospy.is_shutdown() and not self._stop and self.enabled:
            t0 = time.time()

            with self._lock:
                mv = None if self._main is None else self._main.copy()
                wv = None if self._warp is None else self._warp.copy()

            if mv is not None:
                try:
                    cv2.imshow(P.gui_window_main, mv)
                    if wv is not None:
                        cv2.imshow(P.gui_window_warp, wv)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.close()
                        break
                except Exception:
                    self.enabled = False
                    break

            self._fps_count += 1
            now = time.time()
            if now - self._fps_last >= 1.0:
                self.fps = self._fps_count / (now - self._fps_last)
                self._fps_last = now
                self._fps_count = 0

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


def make_main_view(
    frame_bgr: np.ndarray,
    quad: Optional[List[List[int]]],
    status: str,
    cand_score: float,
    face_method: str,
    current_sign_label: str,
    top_raw: str,
    top_corr: str,
    bottom_raw: str,
    top_mean: float,
    bottom_mean: float,
    voted_top: str,
    voted_bottom: str,
    voted_score: float,
    voted_count: int,
    gui_fps: float,
    proc_fps: float,
    submitted_ids: List[int]
) -> np.ndarray:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    if quad is not None and len(quad) == 4:
        cv2.polylines(frame, [np.array(quad, dtype=np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

    out = np.zeros((h, w + P.gui_panel_w, 3), dtype=np.uint8)
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

    cv2.putText(out, f"cand={cand_score:.2f} face={face_method}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 24

    cv2.putText(out, f"sign={current_sign_label}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 24

    cv2.putText(out, f"PROC FPS: {proc_fps:.1f}  GUI FPS: {gui_fps:.1f}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26

    cv2.line(out, (px, y), (w + P.gui_panel_w - 18, y), (80, 80, 80), 1)
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

    cv2.line(out, (px, y), (w + P.gui_panel_w - 18, y), (80, 80, 80), 1)
    y += 28

    cv2.putText(out, "TEMPORAL VOTE", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 34

    cv2.putText(out, f"Type:  {safe_text(voted_top)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 120), 2, cv2.LINE_AA)
    y += 28

    cv2.putText(out, f"Value: {safe_text(voted_bottom)}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2, cv2.LINE_AA)
    y += 28

    cv2.putText(out, f"Score: {voted_score:.2f}  Count: {voted_count}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += 36

    cv2.line(out, (px, y), (w + P.gui_panel_w - 18, y), (80, 80, 80), 1)
    y += 28

    cv2.putText(out, "SUBMITTED IDS", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30
    cv2.putText(out, str(submitted_ids), (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA)

    cv2.putText(out, "Press q in GUI window to close windows", (px, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return out


def make_warp_view(
    warped_bgr: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    top_rect_w: Tuple[int, int, int, int],
    bot_rect_w: Tuple[int, int, int, int],
    top_boxes: List[Tuple[int, int, int, int]],
    bot_boxes: List[Tuple[int, int, int, int]],
    top_text: str,
    bottom_text: str
) -> np.ndarray:
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
# MAIN NODE
# =========================

class SignDetector:
    def __init__(self):
        ensure_dir(P.debug_dir)

        self.bridge = CvBridge()
        self.vote = TemporalVote(P.vote_window)

        self.classes = self._load_classes()
        self.interpreter, self.model_path = self._load_interpreter()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.gui = StableGUI()

        self.status_pub = rospy.Publisher(P.status_topic, StringMsg, queue_size=5)
        self.score_pub = rospy.Publisher(P.score_topic, StringMsg, queue_size=1)

        self.status_timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, P.status_rate_hz)),
            self._publish_status_timer
        )

        self.last_status: Dict = {}
        self.last_debug_write = 0.0

        self.proc_fps = 0.0
        self._fps_last = time.time()
        self._fps_count = 0

        # competition submission state
        self.started_timer = False
        self.stopped_timer = False
        self.node_start_wall_time = time.time()
        self.submitted_clue_ids: set[int] = set()
        self.last_submit_wall_time = 0.0

        # seen-sign state
        self.current_sign_index = 0
        self.anchor_center: Optional[np.ndarray] = None
        self.anchor_area: float = 0.0
        self.sign_miss_frames = 0

        rospy.loginfo(f"{P.node_name}: model={self.model_path}")
        rospy.loginfo(f"{P.node_name}: classes={P.classes_path} n={len(self.classes)}")

        rospy.Subscriber(P.image_topic, Image, self.callback, queue_size=1)

    # ---------- loading ----------

    def _load_classes(self) -> List[str]:
        if not os.path.exists(P.classes_path):
            raise FileNotFoundError(f"classes json not found: {P.classes_path}")
        with open(P.classes_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = sorted(data.keys(), key=lambda k: int(k))
            return [data[k] for k in keys]
        return list(data)

    def _load_interpreter(self) -> Tuple[tf.lite.Interpreter, str]:
        model_path = P.model_path_finetuned if os.path.exists(P.model_path_finetuned) else P.model_path_base
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"tflite model not found: {model_path}")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, model_path

    # ---------- ROS publish helpers ----------

    def _publish_status_timer(self, _evt) -> None:
        if not self.last_status:
            return
        try:
            self.status_pub.publish(StringMsg(data=json.dumps(self.last_status)))
        except Exception:
            pass

    def maybe_send_start(self) -> None:
        if not P.auto_timer_messages:
            return
        if self.started_timer:
            return
        if time.time() - self.node_start_wall_time < P.auto_start_delay_sec:
            return

        msg = f"{P.team_name},{P.team_password},0,NA"
        self.score_pub.publish(StringMsg(data=msg))
        self.started_timer = True
        rospy.loginfo(f"{P.node_name}: AUTO START -> {msg}")

    def maybe_send_stop(self) -> None:
        if not P.auto_timer_messages:
            return
        if not P.auto_stop_when_all_submitted:
            return
        if self.stopped_timer:
            return
        if len(self.submitted_clue_ids) != 8:
            return

        msg = f"{P.team_name},{P.team_password},-1,NA"
        self.score_pub.publish(StringMsg(data=msg))
        self.stopped_timer = True
        rospy.loginfo(f"{P.node_name}: AUTO STOP -> {msg}")

    def clue_type_to_id(self, clue_type: str) -> Optional[int]:
        return P.clue_type_to_id_map.get(str(clue_type).upper(), None)

    def publish_score_tracker(self, clue_type: str, clue_value: str) -> bool:
        clue_id = self.clue_type_to_id(clue_type)
        if clue_id is None:
            rospy.logwarn(f"{P.node_name}: unknown clue type, not publishing: {clue_type}")
            return False

        if clue_id in self.submitted_clue_ids:
            rospy.loginfo(f"{P.node_name}: clue {clue_id} already submitted, skipping")
            return False

        if time.time() - self.last_submit_wall_time < P.submit_cooldown_sec:
            return False

        # competition prediction field should have no spaces
        clue_value_clean = str(clue_value).upper().replace(" ", "")
        msg = f"{P.team_name},{P.team_password},{clue_id},{clue_value_clean}"
        self.score_pub.publish(StringMsg(data=msg))

        self.submitted_clue_ids.add(clue_id)
        self.last_submit_wall_time = time.time()

        rospy.loginfo(f"{P.node_name}: SUBMIT -> {msg}")
        self.maybe_send_stop()
        return True

    # ---------- fps ----------

    def update_proc_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_last >= 1.0:
            self.proc_fps = self._fps_count / (now - self._fps_last)
            self._fps_last = now
            self._fps_count = 0

    # ---------- inference ----------

    def normalize_pred_char(self, ch: str) -> str:
        if ch == "~":
            return " "
        if ch.upper() == "SPACE":
            return " "
        return ch

    def predict_char(self, crop_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        x, vis = preprocess_char_for_cnn(crop_bgr)

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        idx = int(np.argmax(out))
        conf = float(out[idx])
        ch = self.classes[idx] if 0 <= idx < len(self.classes) else "?"
        ch = self.normalize_pred_char(ch)
        return ch, conf, vis

    def predict_string(self, crops_bgr: List[np.ndarray]) -> Tuple[str, List[float], List[Tuple[str, float]], List[np.ndarray]]:
        chars = []
        confs = []
        details = []
        vis32 = []

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

        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < P.min_sign_area_frac * frame_area:
                continue
            if contour_touches_edge(c, W, H, P.edge_touch_margin_px):
                continue

            x, y, ww, hh = cv2.boundingRect(c)
            asp = float(ww) / float(hh + 1e-6)
            if not (P.sign_aspect_min <= asp <= P.sign_aspect_max):
                continue

            base = score_sign_contour(c, frame_area, sign_mask)
            candidates.append((base, c))

        if not candidates:
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

            downstream = 0.0
            try:
                face_rect, _, _, method = find_inner_face(warped)
                if method in ("border", "grey", "fallback") and face_rect is not None:
                    downstream += 1.0
                    face = crop_rect(warped, face_rect)
                    tr, br, _ = compute_line_rects(face)
                    if rect_area(tr) > 0 and rect_area(br) > 0:
                        downstream += 1.0
            except Exception:
                downstream += 0.0

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

    # ---------- seen sign logic ----------

    def update_seen_sign(self, quad_pts: Optional[List[List[int]]], frame_shape: Tuple[int, int, int]) -> bool:
        """
        Returns True if this appears to be a NEW sign.
        """
        H, W = frame_shape[:2]
        diag = float(np.hypot(W, H))

        center, area = quad_center_and_area(quad_pts)
        if center is None or area <= 0:
            self.sign_miss_frames += 1
            if self.sign_miss_frames >= P.sign_miss_reset_frames:
                self.anchor_center = None
                self.anchor_area = 0.0
            return False

        self.sign_miss_frames = 0

        if self.anchor_center is None:
            self.anchor_center = center.copy()
            self.anchor_area = area
            self.current_sign_index += 1
            return True

        dist_norm = float(np.linalg.norm(center - self.anchor_center)) / max(1.0, diag)
        area_ratio = float(area / max(1.0, self.anchor_area))

        is_new = (
            dist_norm > P.new_sign_center_dist_norm or
            area_ratio < P.new_sign_area_ratio_low or
            area_ratio > P.new_sign_area_ratio_high
        )

        if is_new:
            self.anchor_center = center.copy()
            self.anchor_area = area
            self.current_sign_index += 1
            return True

        # same sign -> smooth anchor
        self.anchor_center = (1.0 - P.sign_anchor_alpha) * self.anchor_center + P.sign_anchor_alpha * center
        self.anchor_area = (1.0 - P.sign_anchor_alpha) * self.anchor_area + P.sign_anchor_alpha * area
        return False

    # ---------- debug save ----------

    def save_debug_bundle(
        self,
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
        bottom_vis: Optional[np.ndarray]
    ) -> None:
        ensure_dir(P.debug_dir)

        cv2.imwrite(os.path.join(P.debug_dir, "frame.jpg"), frame)
        cv2.imwrite(os.path.join(P.debug_dir, "sign_mask.jpg"), sign_mask)
        cv2.imwrite(os.path.join(P.debug_dir, "sign_debug.jpg"), sign_debug)

        cv2.imwrite(
            os.path.join(P.debug_dir, "warped.jpg"),
            warped if warped is not None else np.zeros((P.warp_h, P.warp_w, 3), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "face_mask.jpg"),
            face_mask if face_mask is not None else np.zeros((P.warp_h, P.warp_w), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "face_debug.jpg"),
            face_debug if face_debug is not None else np.zeros((P.warp_h, P.warp_w, 3), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "face_crop.jpg"),
            face_crop if face_crop is not None else np.zeros((P.warp_h, P.warp_w, 3), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "top_line.jpg"),
            top_line if top_line is not None else np.zeros((80, 240, 3), dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(P.debug_dir, "bottom_line.jpg"),
            bottom_line if bottom_line is not None else np.zeros((80, 240, 3), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "top_char_mask.jpg"),
            top_char_mask if top_char_mask is not None else np.zeros((80, 240), dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(P.debug_dir, "bottom_char_mask.jpg"),
            bottom_char_mask if bottom_char_mask is not None else np.zeros((80, 240), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "top_char_debug.jpg"),
            top_char_debug if top_char_debug is not None else np.zeros((80, 240, 3), dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(P.debug_dir, "bottom_char_debug.jpg"),
            bottom_char_debug if bottom_char_debug is not None else np.zeros((80, 240, 3), dtype=np.uint8)
        )

        cv2.imwrite(
            os.path.join(P.debug_dir, "top_chars_vis.jpg"),
            top_vis if top_vis is not None else np.zeros((32, 32), dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(P.debug_dir, "bottom_chars_vis.jpg"),
            bottom_vis if bottom_vis is not None else np.zeros((32, 32), dtype=np.uint8)
        )

    # ---------- callback ----------

    def callback(self, msg: Image) -> None:
        self.update_proc_fps()
        self.maybe_send_start()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"{P.node_name}: CvBridge error: {e}")
            return

        warped, sign_mask, sign_dbg, quad_pts, cand_score = self.detect_best_sign(frame)

        status = "NO SIGN"
        face_method = ""
        top_raw = ""
        bottom_raw = ""
        top_corr = ""
        top_mean = 0.0
        bottom_mean = 0.0
        top_confs: List[float] = []
        bottom_confs: List[float] = []

        face_rect = (0, 0, P.warp_w, P.warp_h)
        top_r = (0, 0, 1, 1)
        bot_r = (0, 0, 1, 1)
        top_boxes: List[Tuple[int, int, int, int]] = []
        bot_boxes: List[Tuple[int, int, int, int]] = []

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

        # current vote state
        voted_top, voted_bottom, voted_score, voted_count = self.vote.best()

        if warped is None:
            self.sign_miss_frames += 1
            if self.sign_miss_frames >= P.sign_miss_reset_frames:
                self.vote.clear()
                self.anchor_center = None
                self.anchor_area = 0.0

            voted_top, voted_bottom, voted_score, voted_count = self.vote.best()

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

            if self.gui.enabled:
                main_view = make_main_view(
                    frame_bgr=frame,
                    quad=quad_pts,
                    status=status,
                    cand_score=cand_score,
                    face_method=face_method,
                    current_sign_label=f"{self.current_sign_index}",
                    top_raw=top_raw,
                    top_corr=top_corr,
                    bottom_raw=bottom_raw,
                    top_mean=top_mean,
                    bottom_mean=bottom_mean,
                    voted_top=voted_top,
                    voted_bottom=voted_bottom,
                    voted_score=voted_score,
                    voted_count=voted_count,
                    gui_fps=self.gui.fps,
                    proc_fps=self.proc_fps,
                    submitted_ids=sorted(self.submitted_clue_ids),
                )
                self.gui.update(main_view, None)

            self.last_status = {
                "ok": False,
                "reason": "no_sign",
                "t": rospy.Time.now().to_sec(),
                "cand_score": cand_score,
                "vote_top": voted_top,
                "vote_bottom": voted_bottom,
                "vote_score": voted_score,
                "vote_count": voted_count,
                "submitted_ids": sorted(self.submitted_clue_ids),
            }
            rospy.loginfo_throttle(1.0, f"{P.node_name}: no sign")
            return

        # Seen-sign logic
        is_new_sign = self.update_seen_sign(quad_pts, frame.shape)
        if is_new_sign:
            self.vote.clear()
            voted_top, voted_bottom, voted_score, voted_count = self.vote.best()
            rospy.loginfo(f"{P.node_name}: NEW SIGN detected -> sign_index={self.current_sign_index}")

        face_rect, face_mask, face_dbg, face_method = find_inner_face(warped)
        face_crop = crop_rect(warped, face_rect)

        top_r, bot_r, _ = compute_line_rects(face_crop)
        top_line = crop_rect(face_crop, top_r)
        bottom_line = crop_rect(face_crop, bot_r)

        top_crops_bgr, top_char_mask, top_char_dbg, top_boxes, top_processed = segment_and_preprocess_line(top_line)
        bot_crops_bgr, bottom_char_mask, bottom_char_dbg, bot_boxes, bot_processed = segment_and_preprocess_line(bottom_line)

        top_vis = build_vis_strip(top_processed)
        bottom_vis = build_vis_strip(bot_processed)

        rospy.loginfo_throttle(
            1.0,
            f"{P.node_name}: cand={cand_score:.2f} face={face_method} top_boxes={len(top_boxes)} bottom_boxes={len(bot_boxes)}"
        )

        if len(top_crops_bgr) == 0 or len(bot_crops_bgr) == 0:
            status = "NO CHARS"
            voted_top, voted_bottom, voted_score, voted_count = self.vote.best()
        else:
            top_raw, top_confs, top_details, _ = self.predict_string(top_crops_bgr)
            bottom_raw, bottom_confs, bottom_details, _ = self.predict_string(bot_crops_bgr)

            top_mean = float(np.mean(top_confs)) if top_confs else 0.0
            bottom_mean = float(np.mean(bottom_confs)) if bottom_confs else 0.0

            top_corr = closest_valid_top(top_raw)

            rospy.loginfo_throttle(
                0.8,
                f"{P.node_name}: RAW top={top_raw} mean={top_mean:.3f} chars={top_details} | "
                f"RAW bottom={bottom_raw} mean={bottom_mean:.3f} chars={bottom_details}"
            )
            rospy.loginfo_throttle(0.8, f"{P.node_name}: CORR top={top_corr}")

            valid_top_lengths = {len(w) for w in P.valid_top_words}
            counts_ok = (len(top_raw) in valid_top_lengths) and (3 <= len(bottom_raw) <= 20)

            hard_conf_ok = (
                top_mean >= P.word_mean_conf_threshold and
                bottom_mean >= P.word_mean_conf_threshold and
                all(c >= P.char_conf_threshold for c in top_confs) and
                all(c >= P.char_conf_threshold for c in bottom_confs)
            )

            doubt_conf_ok = (
                top_mean >= P.doubt_word_mean_threshold and
                bottom_mean >= P.doubt_word_mean_threshold and
                all(c >= P.doubt_char_conf_threshold for c in top_confs) and
                all(c >= P.doubt_char_conf_threshold for c in bottom_confs)
            )

            if counts_ok and hard_conf_ok:
                status = "VOTING"
                weight = float(sum(top_confs) + sum(bottom_confs))
                self.vote.add(top_corr, bottom_raw, weight)

            elif counts_ok and doubt_conf_ok:
                status = "REREAD"
                rospy.loginfo_throttle(1.0, f"{P.node_name}: reread-on-doubt")

            else:
                status = "LOW CONF"
                rospy.loginfo_throttle(1.0, f"{P.node_name}: rejected low-confidence frame before voting")

            voted_top, voted_bottom, voted_score, voted_count = self.vote.best()

            stable_vote = (
                voted_top != "" and
                voted_bottom != "" and
                voted_count >= P.vote_min_entries and
                voted_score >= P.vote_min_score
            )

            if stable_vote:
                published = self.publish_score_tracker(voted_top, voted_bottom)
                if published:
                    status = "SUBMITTED"
                else:
                    # already submitted / cooldown / duplicate
                    status = "LOCKED"

        fx0, fy0, _, _ = face_rect
        tx0, ty0, tx1, ty1 = top_r
        bx0, by0, bx1, by1 = bot_r

        top_w = (fx0 + tx0, fy0 + ty0, fx0 + tx1, fy0 + ty1)
        bot_w = (fx0 + bx0, fy0 + by0, fx0 + bx1, fy0 + by1)

        warp_view = make_warp_view(
            warped_bgr=warped,
            face_rect=face_rect,
            top_rect_w=top_w,
            bot_rect_w=bot_w,
            top_boxes=top_boxes,
            bot_boxes=bot_boxes,
            top_text=top_raw,
            bottom_text=bottom_raw
        )

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

        if self.gui.enabled:
            main_view = make_main_view(
                frame_bgr=frame,
                quad=quad_pts,
                status=status,
                cand_score=cand_score,
                face_method=face_method,
                current_sign_label=f"{self.current_sign_index}",
                top_raw=top_raw,
                top_corr=top_corr,
                bottom_raw=bottom_raw,
                top_mean=top_mean,
                bottom_mean=bottom_mean,
                voted_top=voted_top,
                voted_bottom=voted_bottom,
                voted_score=voted_score,
                voted_count=voted_count,
                gui_fps=self.gui.fps,
                proc_fps=self.proc_fps,
                submitted_ids=sorted(self.submitted_clue_ids),
            )
            self.gui.update(main_view, warp_view)

        self.last_status = {
            "ok": True,
            "t": rospy.Time.now().to_sec(),
            "status": status,
            "cand_score": cand_score,
            "face_method": face_method,
            "quad": quad_pts,
            "current_sign_index": self.current_sign_index,
            "top_raw": top_raw,
            "bottom_raw": bottom_raw,
            "top_mean": top_mean,
            "bottom_mean": bottom_mean,
            "top_corrected": top_corr,
            "voted_top": voted_top,
            "voted_bottom": voted_bottom,
            "voted_score": voted_score,
            "voted_count": voted_count,
            "top_boxes": len(top_boxes),
            "bottom_boxes": len(bot_boxes),
            "submitted_ids": sorted(self.submitted_clue_ids),
            "started_timer": self.started_timer,
            "stopped_timer": self.stopped_timer,
            "auto_timer_messages": P.auto_timer_messages,
            "model_path": self.model_path,
        }


if __name__ == "__main__":
    rospy.init_node(P.node_name, anonymous=False)
    ensure_dir(P.debug_dir)
    SignDetector()
    rospy.spin()
