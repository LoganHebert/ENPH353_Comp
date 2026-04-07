#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SignDetector:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber("/B1/pi_camera/image_raw", Image, self.callback)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("sign_detector")
    SignDetector()
    rospy.spin()
