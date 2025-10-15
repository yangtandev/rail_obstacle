'''
cam = Camera(_RTSP)
frame = cam.get_frame()
'''

import cv2
import time
import numpy as np
import os
import requests

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

            # Check 3: JPEG file integrity (SOI and EOI markers)
            if not (data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9')):
                print(f"Error: Incomplete JPEG file received from {self.rtsp}")
                return None
            
            # Decode the verified, complete image data
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching frame from {self.rtsp}: {e}")
            return None

        # --- Method 2: Reactive Quality Filtering ---

        # Check 4: Decoding failure or empty image
        if frame is None or frame.size == 0:
            print("Error: Failed to decode image data.")
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
