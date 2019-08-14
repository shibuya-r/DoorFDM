""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import numpy as np
import math
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from timeit import default_timer as timer

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

resize_prop = (640, 480)


class VideoCamera(object):
    def __init__(self, detections, no_v4l):

        self.input_stream = 0
        # NOTE need to check os, Linux, Windows or Mac
        if no_v4l:
            self.cap = cv2.VideoCapture(self.input_stream)
        else: #for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
            try:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            except:
                import traceback
                traceback.print_exc()
                print("\nPlease try to start with command line parameters using --no_v4l\n")
                os._exit(0)

        ret, self.frame = self.cap.read()
        cap_prop = self._get_cap_prop()
        logger.info("cap_pop:{}, frame_prop:{}".format(cap_prop, resize_prop))

        self.detections = detections

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, is_async_mode, is_face_detection,is_head_pose_detection):

        if is_async_mode:
            ret, next_frame = self.cap.read()
            if not ret:
                return None
            next_frame = cv2.resize(next_frame, resize_prop)
        else:
            ret, self.frame = self.cap.read()
            if not ret:
                return None
            self.frame = cv2.resize(self.frame, resize_prop)
            next_frame = None

        if is_face_detection:
            self.frame = self.detections.face_detection(
                self.frame, next_frame, is_async_mode,is_head_pose_detection)

        ret, jpeg = cv2.imencode('1.jpg', self.frame)

        if is_async_mode:
            self.frame = next_frame

        return jpeg.tostring()
