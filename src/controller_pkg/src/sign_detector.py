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
MODEL_PATH = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_model.tflite")
CLASSES_PATH = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_classes.json")

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


def fixed_line_crops(warped):
    h, w = warped.shape[:2]
    top = warped[int(0.08 * h):int(0.36 * h), int(0.40 * w):int(0.90 * w)]
    bottom = warped[int(0.52 * h):int(0.86 * h), int(0.12 * w):int(0.90 * w)]
    return top, bottom


def blue_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 60, 30])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def extract_line_regions(warped):
    """
    First use a broad fixed crop. Then compute masks inside those crops.
    This is simpler and more robust than the previous fully dynamic logic.
    """
    top_img, bottom_img = fixed_line_crops(warped)
    top_mask = blue_mask(top_img)
    bottom_mask = blue_mask(bottom_img)
    return top_img, bottom_img, top_mask, bottom_mask


def segment_chars_from_line(line_bgr, line_mask, debug=False):
    if line_bgr is None or line_mask is None:
        return [], None, None, []

    H, W = line_mask.shape[:2]
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 25:
            continue
        if h < 0.28 * H:
            continue
        if h > 0.98 * H:
            continue
        if w < 0.01 * W:
            continue
        if w > 0.22 * W:
            continue

        fill = cv2.contourArea(c) / float(max(w * h, 1))
        if fill < 0.08:
            continue

        # reject tall narrow border bars at edges
        is_tall = h > 0.72 * H
        is_narrow = w < 0.09 * W
        near_left = x < 0.06 * W
        near_right = (x + w) > 0.94 * W
        if is_tall and is_narrow and (near_left or near_right):
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    crops = []
    for x, y, w, h in boxes:
        pad = max(2, int(0.15 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        crop_mask = line_mask[y0:y1, x0:x1]

        ch, cw = crop_mask.shape[:2]
        side = max(ch, cw) + 10
        canvas = np.zeros((side, side), dtype=np.uint8)

        y_off = (side - ch) // 2
        x_off = (side - cw) // 2
        canvas[y_off:y_off + ch, x_off:x_off + cw] = crop_mask

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

        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(CLASSES_PATH, "r") as f:
            self.classes = json.load(f)

        rospy.loginfo("SignDetector ready")
        rospy.loginfo(f"Loaded model: {MODEL_PATH}")
        rospy.loginfo(f"Loaded classes: {CLASSES_PATH}")

        rospy.Subscriber(
            "/B1/rrbot/camera1/image_raw",
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
        for crop in crops:
            ch, conf = self.predict_char(crop)
            chars.append(ch)
            confs.append(conf)
        return "".join(chars), confs

    def save_debug_images(
        self,
        frame=None,
        sign_mask=None,
        sign_edges=None,
        sign_debug=None,
        warped=None,
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

        top_line, bottom_line, top_line_mask, bottom_line_mask = extract_line_regions(warped)

        top_crops, _, top_char_dbg, top_boxes = segment_chars_from_line(top_line, top_line_mask, debug=True)
        bottom_crops, _, bottom_char_dbg, bottom_boxes = segment_chars_from_line(bottom_line, bottom_line_mask, debug=True)

        rospy.loginfo_throttle(1.0, f"top_boxes={len(top_boxes)} bottom_boxes={len(bottom_boxes)}")

        # Always save current debug images, even on failure
        now = rospy.get_time()
        if now - self.last_save_time > 1.0:
            self.save_debug_images(
                frame=frame,
                sign_mask=sign_mask,
                sign_edges=sign_edges,
                sign_debug=sign_debug,
                warped=warped,
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

        top_raw, top_confs = self.predict_string(top_crops)
        bottom_raw, bottom_confs = self.predict_string(bottom_crops)

        top_avg = float(np.mean(top_confs)) if top_confs else 0.0
        bottom_avg = float(np.mean(bottom_confs)) if bottom_confs else 0.0

        top_fixed = nearest_clue_type(top_raw)

        if top_avg > 0.40 and bottom_avg > 0.40:
            self.vote.add(top_fixed, top_avg, bottom_raw, bottom_avg)

        best_top, best_top_score, best_bottom, best_bottom_score = self.vote.get_best()

        if best_top and best_bottom:
            rospy.loginfo_throttle(
                0.5,
                f"RAW: top={top_raw} ({top_avg:.2f}) bottom={bottom_raw} ({bottom_avg:.2f}) | "
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
