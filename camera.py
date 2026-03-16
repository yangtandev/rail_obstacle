import cv2
import threading
import time
import logging as log
import os

class Camera:
    def __init__(self, rtsp):
        self.rtsp = rtsp
        
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
        
        self.stream = cv2.VideoCapture(self.rtsp)
        self.stopped = False
        
        if not self.stream.isOpened():
            log.error(f"CAM {self.rtsp} [ACQ]: 無法開啟 RTSP 串流。")
            self.ret = False
            self.frame = None
        else:
            self.ret, self.frame = self.stream.read()
            # 啟動背景執行緒持續讀取最新畫面，避免 OpenCV 緩衝區堆積舊畫面導致延遲
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
                break
            # 持續抓取最新畫面
            self.ret, self.frame = self.stream.read()
            time.sleep(0.01) # 略微休眠避免佔用過高 CPU

    def get_data(self):
        # 回傳直接可供 OpenCV/YOLO 使用的 numpy array (BGR 格式)
        if self.ret and self.frame is not None:
            return self.frame.copy()
        return None

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if self.stream.isOpened():
            self.stream.release()