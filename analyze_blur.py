import cv2
import numpy as np

def analyze_image_blur(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Image: {image_path}, Laplacian Variance: {laplacian_var:.2f}")

image_paths = [
    "/home/gini-facetest/rail_obstacle/saved_images/20251029/detected_cam1921683120_2025-10-29_09-54-29.jpg",
    "/home/gini-facetest/rail_obstacle/saved_images/20251029/detected_cam1921683120_2025-10-29_09-56-44.jpg"
]

for path in image_paths:
    analyze_image_blur(path)
