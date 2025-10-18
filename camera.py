'''
cam = Camera(_RTSP)
data = cam.get_data()
'''

import requests
import time
import logging as log

class Camera:
    def __init__(self, rtsp):
        self.rtsp = rtsp

    def get_data(self):

        try:
            response = requests.get(self.rtsp, timeout=2.0)
            end_time = time.time()

            if response.status_code != 200:
                log.error(f"CAM {self.rtsp} [ACQ]: HTTP Error: {response.status_code}")
                return None
            
            data = response.content
            if not data:
                log.error(f"CAM {self.rtsp} [ACQ]: Received empty content.")
                return None
            
            return data

        except requests.exceptions.RequestException as e:
            end_time = time.time()
            log.error(f"CAM {self.rtsp} [ACQ]: Error fetching data: {e}. Request took {(end_time - start_time) * 1000:.2f} ms.")
            return None