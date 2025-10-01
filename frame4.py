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


def process_frame(rtsp, queue_frame, Process_frame_name, Hailey, queue_alert=None):

    cam = Camera(rtsp)
    cam.connect()
    
    while True:
        frame = cam.get_frame()

        if type(frame) == type(None):
            continue

        
#        if Process_frame_name == 'under' and type(frame) != type(None):
          
#            frame = cv2.resize(frame, (1280,720))
            
        if queue_frame.empty() is True:
            queue_frame.put(frame)
   
        time.sleep(0.067)

        if cv2.waitKey(1) & 0xFF == ord('q') or 0xFF == ord('Q'):
            cv2.destroyAllWindows()
            sys.exit()
