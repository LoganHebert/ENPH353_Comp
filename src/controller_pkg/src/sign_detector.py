	#!/usr/bin/env python3

import os
import json
from collections import deque, defaultdict

import cv2
import rospy
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


DEBUG_DIR = "/tmp/sign_debug"

MODEL_PATH_FINETUNED = os.path.expanduser(
    "~/ros_ws/src/controller_pkg/models/sign_char_model_finetuned.tflite"
)
MODEL_PATH_BASE = os.path.expanduser(
    "~/ros_ws/src/controller_pkg/models/sign_char_model.tflite"
)
CLASSES_PATH = os.path.expanduser(
    "~/ros_ws/src/controller_pkg/models/sign_char_classes.json"
)

IMAGE_TOPIC = "/B1/rrbot/camera1/image_raw"

VALID_CLUE_TYPES = [
    "SIZE", "VICTIM", "CRIME", "TIME",
    "PLACE", "MOTIVE", "WEAPON", "BANDIT"
]


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def four_point_transform(image, pts, out_w=520, out_h=320):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
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


def nearest_clue_type(text):
    if not text:
        return ""
    best = None
    best_dist = 10**9
    for clue in VALID_CLUE_TYPES:
        d = levenshtein(text, clue)
        if d < best_dist:
            best_dist = d
            best = clue
    return best


def detect_sign_and_warp(frame):
    """
    Keep this close to your older working version.
    """
    debug = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 120, 30])
    upper_blue = np.array([135, 255, 190])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_area = 0

    H, W = frame.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue

        fill_ratio = area / float(rect_area)
        aspect_ratio = w / float(h)
        edge_count = cv2.countNonZero(edges[y:y + h, x:x + w])

        # reject stuff glued to image edges
        if x <= 2 or y <= 2 or (x + w) >= (W - 2) or (y + h) >= (H - 2):
            continue

        if fill_ratio < 0.30:
            continue
        if not (1.0 <= aspect_ratio <= 2.8):
            continue
        if edge_count < 40:
            continue

        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None, mask, edges, debug, None

    peri = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.03 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(best_cnt)
        quad = cv2.boxPoints(rect).astype(int)

    cv2.drawContours(debug, [quad.astype(int)], -1, (255, 0, 0), 2)
    warped = four_point_transform(frame, quad, out_w=520, out_h=320)

    return warped, mask, edges, debug, quad


def blue_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 60, 30])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def gray_face_mask(img_bgr):
    """
    Detect inner gray sign face dynamically.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 45])
    upper = np.array([180, 90, 245])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def find_face_rect(warped):
    """
    Find the inner gray face of the sign dynamically.
    """
    H, W = warped.shape[:2]
    mask = gray_face_mask(warped)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / float(max(h, 1))

        if area < 0.18 * H * W:
            continue
        if not (1.1 <= aspect <= 2.6):
            continue

        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    if best_rect is None:
        # conservative fallback
        x0 = int(0.12 * W)
        y0 = int(0.12 * H)
        x1 = int(0.88 * W)
        y1 = int(0.88 * H)
    else:
        x, y, w, h = best_rect
        pad_x = int(0.01 * w)
        pad_y = int(0.01 * h)
        x0 = max(0, x + pad_x)
        y0 = max(0, y + pad_y)
        x1 = min(W, x + w - pad_x)
        y1 = min(H, y + h - pad_y)

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 255), 2)

    return (x0, y0, x1, y1), mask, dbg


def find_text_bands(face_bgr):
    """
    Find top and bottom text rows dynamically from blue mask energy.
    """
    H, W = face_bgr.shape[:2]
    mask = blue_mask(face_bgr)

    row_sum = np.sum(mask > 0, axis=1).astype(np.float32) / float(max(W, 1))
    if len(row_sum) >= 9:
        kernel = np.ones(9, dtype=np.float32) / 9.0
        row_sum = np.convolve(row_sum, kernel, mode="same")

    peak = float(np.max(row_sum)) if row_sum.size else 0.0
    thresh = max(0.01, 0.20 * peak)

    bands = []
    in_band = False
    start = 0

    for y, v in enumerate(row_sum):
        if v > thresh and not in_band:
            start = y
            in_band = True
        elif v <= thresh and in_band:
            end = y
            if end - start >= 6:
                bands.append((start, end))
            in_band = False

    if in_band:
        end = H
        if end - start >= 6:
            bands.append((start, end))

    # fallback if detection weak
    if len(bands) < 2:
        top_band = (int(0.05 * H), int(0.40 * H))
        bottom_band = (int(0.52 * H), int(0.90 * H))
        return top_band, bottom_band, mask

    top_band = bands[0]
    bottom_band = bands[-1]
    return top_band, bottom_band, mask


def band_bbox(mask, band):
    """
    Tight bbox of mask inside a row band.
    """
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

    return (x0, yy0, x1, yy1)


def extract_line_regions(warped):
    """
    Dynamic line extraction:
    1) find gray face
    2) find blue-text row bands inside face
    3) crop tight top and bottom line images
    """
    fx0, fy0, fx1, fy1 = find_face_rect(warped)[0]
    face = warped[fy0:fy1, fx0:fx1]
    top_band, bottom_band, face_blue_mask = find_text_bands(face)

    top_bbox = band_bbox(face_blue_mask, top_band)
    bottom_bbox = band_bbox(face_blue_mask, bottom_band)

    FH, FW = face.shape[:2]

    if top_bbox is None:
        top_bbox = (int(0.38 * FW), int(0.05 * FH), int(0.82 * FW), int(0.35 * FH))
    if bottom_bbox is None:
        bottom_bbox = (int(0.04 * FW), int(0.55 * FH), int(0.55 * FW), int(0.88 * FH))

    def pad_bbox(b):
        x0, y0, x1, y1 = b
        pad_x = max(4, int(0.08 * (x1 - x0)))
        pad_y = max(4, int(0.22 * (y1 - y0)))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(FW, x1 + pad_x)
        y1 = min(FH, y1 + pad_y)
        return (x0, y0, x1, y1)

    top_bbox = pad_bbox(top_bbox)
    bottom_bbox = pad_bbox(bottom_bbox)

    tx0, ty0, tx1, ty1 = top_bbox
    bx0, by0, bx1, by1 = bottom_bbox

    top_img = face[ty0:ty1, tx0:tx1]
    bottom_img = face[by0:by1, bx0:bx1]

    top_mask = blue_mask(top_img)
    bottom_mask = blue_mask(bottom_img)

    # debug overlay on face
    face_dbg = face.copy()
    cv2.rectangle(face_dbg, (tx0, ty0), (tx1, ty1), (255, 0, 0), 2)
    cv2.rectangle(face_dbg, (bx0, by0), (bx1, by1), (0, 0, 255), 2)

    info = {
        "face": face,
        "face_dbg": face_dbg,
        "face_blue_mask": face_blue_mask
    }

    return top_img, bottom_img, top_mask, bottom_mask, info


def split_wide_box(mask, box, min_h_fraction=0.28):
    """
    Split merged boxes like 'AM' using vertical projection inside the contour box.
    """
    x, y, w, h = box
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0 or w < 8:
        return [box]

    col_sum = np.sum(roi > 0, axis=0).astype(np.float32)
    if np.max(col_sum) <= 0:
        return [box]

    if len(col_sum) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        col_sum = np.convolve(col_sum, kernel, mode="same")

    left_bound = max(2, int(0.22 * w))
    right_bound = min(w - 2, int(0.78 * w))
    if right_bound <= left_bound:
        return [box]

    best_split = None
    best_val = None

    for s in range(left_bound, right_bound):
        valley = col_sum[s]
        left_peak = np.max(col_sum[:s]) if s > 0 else 0.0
        right_peak = np.max(col_sum[s:]) if s < w else 0.0

        if left_peak < 1e-6 or right_peak < 1e-6:
            continue

        if valley <= 0.45 * min(left_peak, right_peak):
            if best_val is None or valley < best_val:
                best_val = valley
                best_split = s

    if best_split is None:
        return [box]

    s = int(best_split)
    left_roi = roi[:, :s]
    right_roi = roi[:, s:]

    def tight_box(sub_roi, x_off):
        ys, xs = np.where(sub_roi > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        return (x + x_off + x0, y + y0, x1 - x0 + 1, y1 - y0 + 1)

    b1 = tight_box(left_roi, 0)
    b2 = tight_box(right_roi, s)

    out = []
    for b in [b1, b2]:
        if b is None:
            continue
        bx, by, bw, bh = b
        if bh >= max(4, int(min_h_fraction * mask.shape[0])) and bw >= 2:
            out.append((bx, by, bw, bh))

    return out if len(out) >= 2 else [box]


def segment_chars_from_line(line_bgr, line_mask, debug=False):
    if line_bgr is None or line_mask is None:
        return [], None, None, []

    H, W = line_mask.shape[:2]
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 20:
            continue
        if h < 0.24 * H:
            continue
        if h > 0.98 * H:
            continue
        if w < max(2, int(0.01 * W)):
            continue
        if w > 0.30 * W:
            continue

        fill = cv2.contourArea(c) / float(max(w * h, 1))
        if fill < 0.08:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    if boxes:
        widths = np.array([b[2] for b in boxes], dtype=np.float32)
        median_w = float(np.median(widths))
        split_boxes = []
        for b in boxes:
            if median_w > 0 and b[2] > 1.45 * median_w:
                split_boxes.extend(split_wide_box(line_mask, b, min_h_fraction=0.24))
            else:
                split_boxes.append(b)
        boxes = sorted(split_boxes, key=lambda b: b[0])

    crops = []
    for x, y, w, h in boxes:
        pad = max(2, int(0.15 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        crop_bgr = line_bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            continue

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        ch, cw = bw.shape[:2]
        side = max(ch, cw) + 10
        canvas = np.zeros((side, side), dtype=np.uint8)

        y_off = (side - ch) // 2
        x_off = (side - cw) // 2
        canvas[y_off:y_off + ch, x_off:x_off + cw] = bw

        canvas = cv2.resize(canvas, (40, 40), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)

        crops.append(canvas)

    dbg = None
    if debug:
        dbg = line_bgr.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return crops, line_mask, dbg, boxes


class TemporalVote:
    def __init__(self, maxlen=12):
        self.top_history = deque(maxlen=maxlen)
        self.bottom_history = deque(maxlen=maxlen)

    def add(self, top_text, top_conf, bottom_text, bottom_conf):
        self.top_history.append((top_text, top_conf))
        self.bottom_history.append((bottom_text, bottom_conf))

    @staticmethod
    def weighted_best(history):
        scores = defaultdict(float)
        for text, conf in history:
            if text:
                scores[text] += conf
        if not scores:
            return "", 0.0
        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0], best[1]

    def get_best(self):
        best_top, top_score = self.weighted_best(self.top_history)
        best_bottom, bottom_score = self.weighted_best(self.bottom_history)
        return best_top, top_score, best_bottom, bottom_score

    def clear(self):
        self.top_history.clear()
        self.bottom_history.clear()


class SignDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_save_time = 0.0
        self.vote = TemporalVote(maxlen=12)

        os.makedirs(DEBUG_DIR, exist_ok=True)

        model_path = MODEL_PATH_FINETUNED if os.path.exists(MODEL_PATH_FINETUNED) else MODEL_PATH_BASE
        self.model_path = model_path

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(CLASSES_PATH, "r") as f:
            self.classes = json.load(f)

        rospy.loginfo("SignDetector ready")
        rospy.loginfo(f"Loaded model: {self.model_path}")
        rospy.loginfo(f"Loaded classes: {CLASSES_PATH}")

        rospy.Subscriber(
            IMAGE_TOPIC,
            Image,
            self.callback,
            queue_size=1
        )

    def predict_char(self, crop_gray):
        x = crop_gray.astype(np.float32) / 255.0
        x = x[None, ..., None]

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])

        idx = int(np.argmax(out))
        conf = float(np.max(out))
        return self.classes[idx], conf

    def predict_string(self, crops):
        chars = []
        confs = []
        details = []
        for crop in crops:
            ch, conf = self.predict_char(crop)
            chars.append(ch)
            confs.append(conf)
            details.append((ch, conf))
        return "".join(chars), confs, details

    def save_debug_images(
        self,
        frame=None,
        sign_mask=None,
        sign_edges=None,
        sign_debug=None,
        warped=None,
        face=None,
        face_debug=None,
        face_blue_mask=None,
        top_line=None,
        bottom_line=None,
        top_line_mask=None,
        bottom_line_mask=None,
        top_char_dbg=None,
        bottom_char_dbg=None,
        top_crops=None,
        bottom_crops=None
    ):
        if frame is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "frame.jpg"), frame)
        if sign_mask is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "sign_mask.jpg"), sign_mask)
        if sign_edges is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "sign_edges.jpg"), sign_edges)
        if sign_debug is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "sign_detected.jpg"), sign_debug)
        if warped is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "warped.jpg"), warped)
        if face is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "face.jpg"), face)
        if face_debug is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "face_debug.jpg"), face_debug)
        if face_blue_mask is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "face_blue_mask.jpg"), face_blue_mask)
        if top_line is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "top_line.jpg"), top_line)
        if bottom_line is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_line.jpg"), bottom_line)
        if top_line_mask is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "top_line_mask.jpg"), top_line_mask)
        if bottom_line_mask is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_line_mask.jpg"), bottom_line_mask)
        if top_char_dbg is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "top_char_debug.jpg"), top_char_dbg)
        if bottom_char_dbg is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_char_debug.jpg"), bottom_char_dbg)

        if top_crops is not None:
            for i, crop in enumerate(top_crops):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"top_crop_{i}.png"), crop)
        if bottom_crops is not None:
            for i, crop in enumerate(bottom_crops):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"bottom_crop_{i}.png"), crop)

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        warped, sign_mask, sign_edges, sign_debug, _ = detect_sign_and_warp(frame)
        if warped is None:
            self.vote.clear()
            self.save_debug_images(frame=frame, sign_mask=sign_mask, sign_edges=sign_edges, sign_debug=sign_debug)
            rospy.loginfo_throttle(1.0, "No sign candidate found")
            return

        top_line, bottom_line, top_line_mask, bottom_line_mask, info = extract_line_regions(warped)

        top_crops, _, top_char_dbg, top_boxes = segment_chars_from_line(top_line, top_line_mask, debug=True)
        bottom_crops, _, bottom_char_dbg, bottom_boxes = segment_chars_from_line(bottom_line, bottom_line_mask, debug=True)

        rospy.loginfo_throttle(1.0, f"top_boxes={len(top_boxes)} bottom_boxes={len(bottom_boxes)}")

        now = rospy.get_time()
        if now - self.last_save_time > 1.0:
            self.save_debug_images(
                frame=frame,
                sign_mask=sign_mask,
                sign_edges=sign_edges,
                sign_debug=sign_debug,
                warped=warped,
                face=info["face"],
                face_debug=info["face_dbg"],
                face_blue_mask=info["face_blue_mask"],
                top_line=top_line,
                bottom_line=bottom_line,
                top_line_mask=top_line_mask,
                bottom_line_mask=bottom_line_mask,
                top_char_dbg=top_char_dbg,
                bottom_char_dbg=bottom_char_dbg,
                top_crops=top_crops,
                bottom_crops=bottom_crops
            )
            self.last_save_time = now
            rospy.loginfo("Saved current debug images to /tmp/sign_debug")

        if len(top_crops) == 0 or len(bottom_crops) == 0:
            self.vote.clear()
            rospy.loginfo_throttle(1.0, "No characters segmented")
            return

        top_raw, top_confs, top_details = self.predict_string(top_crops)
        bottom_raw, bottom_confs, bottom_details = self.predict_string(bottom_crops)

        top_avg = float(np.mean(top_confs)) if top_confs else 0.0
        bottom_avg = float(np.mean(bottom_confs)) if bottom_confs else 0.0

        top_fixed = nearest_clue_type(top_raw)

        rospy.loginfo_throttle(
            0.5,
            f"RAW: top={top_raw} ({top_avg:.2f}) chars={top_details} | "
            f"bottom={bottom_raw} ({bottom_avg:.2f}) chars={bottom_details}"
        )

        if top_avg > 0.40 and bottom_avg > 0.40:
            self.vote.add(top_fixed, top_avg, bottom_raw, bottom_avg)

        best_top, best_top_score, best_bottom, best_bottom_score = self.vote.get_best()

        if best_top and best_bottom:
            rospy.loginfo_throttle(
                0.5,
                f"VOTED: TYPE={best_top} score={best_top_score:.2f} VALUE={best_bottom} score={best_bottom_score:.2f}"
            )
        else:
            rospy.loginfo_throttle(
                0.5,
                f"Unstable OCR: top={top_raw} ({top_avg:.2f}) bottom={bottom_raw} ({bottom_avg:.2f})"
            )


if __name__ == "__main__":
    rospy.init_node("sign_detector")
    SignDetector()
    rospy.spin()
