'''
This file is for `process_pass_frame`
'''

import time
import cv2
import os
import sys
import numpy as np
from screeninfo import get_monitors
from camera import Camera
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import logging as log
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def process_frame(rtsp, queue_frame, Process_frame_name, Hailey, stop_event, queue_alert=None):

    cam = Camera(rtsp)
    log.info(f"CAM {Process_frame_name}: Acquisition process started.")
    
    while not stop_event.is_set():
        t_loop_start = time.time()

        data = cam.get_data()

        t_after_get = time.time()

        if data is None:
            time.sleep(1) # Don't spin too fast on errors
            continue
        
        # This queue draining is redundant if the main loop also does it, but keep for safety
        while not queue_frame.empty():
            try:
                queue_frame.get_nowait()
            except Exception:
                break
        
        try:
            queue_frame.put_nowait(data)
        except Exception:
            pass
        
        t_after_put = time.time()

        # Dynamically sleep to aim for a ~15 FPS loop rate (67ms)
        loop_duration = t_after_put - t_loop_start
        sleep_time = max(0, 0.067 - loop_duration)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        t_end_loop = time.time()

        log.info(
            f"CAM {Process_frame_name} [ACQ TIMING]: "
            f"get_data: {(t_after_get - t_loop_start) * 1000:.2f} ms, "
            f"put_queue: {(t_after_put - t_after_get) * 1000:.2f} ms, "
            f"sleep: {sleep_time * 1000:.2f} ms, "
            f"Total Loop: {(t_end_loop - t_loop_start) * 1000:.2f} ms"
        )


