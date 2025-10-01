'''
cam = Camera(_RTSP)
cam.connect()
frame = cam.get_frame(cam_conn)
'''

import cv2
import time
import requests
import numpy as np
import os

class Camera:
    def __init__(self, rtsp):
        self.rtsp = rtsp
        self.connection = None
        self.is_jpg = '.jpg' in self.rtsp.lower()

    def connect(self):
        '''doc'''
        if not self.is_jpg:
            # Force RTSP to use TCP
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            self.connection = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)

    def get_frame(self):
        '''
        This method will return a frame with the original color.
        '''
        frame = None  # Initialize frame
        if self.is_jpg:
            try:
                # For JPG URLs, fetch the image using requests
                response = requests.get(self.rtsp, timeout=5)
                response.raise_for_status()  # Raise an exception for bad status codes
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except requests.exceptions.RequestException as e:
                # print(f'Error fetching JPG from {self.rtsp}: {e}')
                frame = None
            except Exception as e:
                # print(f'Error decoding JPG from {self.rtsp}: {e}')
                frame = None
        else:
            # For RTSP or other video streams
            if self.connection is None or not self.connection.isOpened():
                # print("RTSP connection is not open. Reconnecting...")
                self.connect()
                return None

            ret, frame = self.connection.read()
            cv2.waitKey(1)
            if not ret:
                # print('Can not read frame from RTSP stream. Reconnecting...')
                self.connect()
                return None

        # --- Shared Validation for both JPG and RTSP ---
        if frame is None:
            return None

        # Check if the frame is mostly grey/blank by calculating variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < 100:  # Threshold for variance, can be adjusted
            return None  # Skip grey/blank frame

        return frame