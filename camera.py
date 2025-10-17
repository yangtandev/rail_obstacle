'''
cam = Camera(_RTSP)
frame = cam.get_frame()
'''

import cv2
import time
import numpy as np
import os
import requests
import sys
import io
import contextlib

@contextlib.contextmanager
def stderr_redirected(to=os.devnull):
    """
    Redirect stderr to a file-like object.
    """
    fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        sys.stderr.close()  # close sys.stderr
        os.dup2(to_fd, fd)  # redirect new sys.stderr to to_fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(file.fileno())
        try:
            yield
        finally:
            _redirect_stderr(old_stderr.fileno())

class Camera:
    def __init__(self, rtsp):
        self.rtsp = rtsp

    def get_frame(self):
        """
        This method fetches an image from a URL, verifies its integrity,
        and then runs quality checks.
        It combines proactive download verification with reactive quality filtering.
        """
        # --- Method 1: Proactive Download and Verification ---
        try:
            response = requests.get(self.rtsp, timeout=2.0) # 2-second timeout

            # Check 1: HTTP Status
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} from {self.rtsp}")
                return None

            data = response.content

            # Check 2: Empty content
            if not data:
                print(f"Error: Received empty content from {self.rtsp}")
                return None
            
            # --- New Check: Corrupt JPEG data (replaces old Check 3 and enhances Check 4) ---
            temp_stderr_file = "temp_stderr_camera.txt"
            frame = None
            with stderr_redirected(to=temp_stderr_file):
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            
            stderr_output = ""
            if os.path.exists(temp_stderr_file):
                with open(temp_stderr_file, 'r') as f:
                    stderr_output = f.read()
                os.remove(temp_stderr_file) # Clean up the temporary file

            if "Corrupt JPEG data" in stderr_output:
                print(f"Frame discarded: Corrupt JPEG data detected from {self.rtsp}.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching frame from {self.rtsp}: {e}")
            return None

        # Check 4 (modified): Decoding failure or empty image (if not already caught by corruption check)
        if frame is None or frame.size == 0:
            print("Error: Failed to decode image data (or empty after decoding).")
            return None

        # Check 5: Discard non-BGR images (e.g., grayscale)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print("Error: Received non-BGR image.")
            return None

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check 6: Solid color blocks anywhere in the image using a grid check
        h, w = gray.shape
        grid_size = 3  # Create a 3x3 grid
        cell_h, cell_w = h // grid_size, w // grid_size
        variance_threshold = 200

        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                if cell.size > 0:
                    variance = np.var(cell)
                    if variance < variance_threshold:
                        print(f"Frame discarded: Solid color block detected in grid cell ({i},{j}) with variance {variance:.2f} < {variance_threshold}.")
                        return None

        # Check 7: Blurry images (existing check)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            print("Frame discarded: Image is too blurry.")
            return None

        # If all checks pass, return the valid frame
        return frame
