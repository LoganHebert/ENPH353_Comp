#!/usr/bin/env python3

import json
import os

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_model.tflite")
CLASSES_PATH = os.path.expanduser("~/ros_ws/src/controller_pkg/models/sign_char_classes.json")


def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)

    return interpreter, input_details, output_details, classes


def predict_char(interpreter, input_details, output_details, classes, crop_gray):
    crop_gray = cv2.resize(crop_gray, (32, 32), interpolation=cv2.INTER_AREA)
    x = crop_gray.astype(np.float32) / 255.0
    x = x[None, ..., None]  # (1, 32, 32, 1)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_idx = int(np.argmax(output))
    conf = float(np.max(output))
    pred_char = classes[pred_idx]

    return pred_char, conf


def main():
    interpreter, input_details, output_details, classes = load_model()

    # temporary synthetic test image
    test_img = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(test_img, "M", (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)

    pred_char, conf = predict_char(
        interpreter, input_details, output_details, classes, test_img
    )

    print("Predicted:", pred_char)
    print("Confidence:", conf)

    cv2.imwrite("/tmp/test_tflite_input.png", test_img)
    print("Saved /tmp/test_tflite_input.png")


if __name__ == "__main__":
    main()
