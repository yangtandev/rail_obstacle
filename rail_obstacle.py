import numpy as np
import cv2
from ultralytics import YOLOv10
from pathlib import Path
import logging as log
import sys
import os
import glob
import time
import multiprocessing
from multiprocessing import Queue, Process, Event
import requests
from zoneinfo import ZoneInfo
from shapely.geometry import Polygon, box
import datetime
import contextlib
import io
import threading

from camera import Camera

api = "https://jenyi-xg.api.ginibio.com/api/v1"
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO, stream=sys.stdout)
models_dir = Path('./models')
model_name = "rail_obstacle"
int8_model_det_path = models_dir / 'int8' / f'{model_name}_openvino_model'

@contextlib.contextmanager
def stderr_redirected_to_file(filepath):
    """Redirects stderr to a given file path."""
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    
    with open(filepath, 'w') as f:
        os.dup2(f.fileno(), original_stderr_fd)
    
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

def save_image_with_limit(image, directory, folder_name, cam_id, limit=300):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, 'misclassification'))
    image_files = glob.glob(os.path.join(directory, "*.jpg"))
    if len(image_files) >= limit:
        oldest_image = min(image_files, key=os.path.getctime)
        os.remove(oldest_image)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    image_path = os.path.join(directory, f"{folder_name}_cam{cam_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, image)
    return image_path

def image2base64(image):
    image = cv2.resize(image, (250, 150))
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        return base64.b64encode(buffer).decode('utf-8')
    else:
        raise ValueError("Failed to encode image")

def read_areas(area_files):
    polygons = []
    for file_path in area_files:
        points = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                x, y = map(int, line.strip().split(','))
                points.append((x, y))
        polygons.append(Polygon(points))
    return polygons

def check_bboxes_in_danger_zone(danger_area_polygon, bboxes, iou_threshold=0.2):
    for bbox in bboxes:
        bbox_poly = box(*bbox)
        if danger_area_polygon.intersects(bbox_poly):
            intersection_area = danger_area_polygon.intersection(bbox_poly).area
            bbox_area = bbox_poly.area
            if bbox_area > 0:
                ratio = intersection_area / bbox_area
                if ratio > iou_threshold:
                    return True
    return False

def calculate_overlap_ratio(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    if bbox1_area == 0:
        return 0.0
    return intersection_area / bbox1_area

def draw_transparent_polygon(image, points, color=(0, 0, 255), opacity=0.3):
    overlay = image.copy()
    output = image.copy()
    if not points:
        return image
    if hasattr(points, 'coords'):
        points = list(points.coords)
    if points:
        cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], color)
        cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
    return output

def alert_api(image, api, location):
    url = api + '/alerts/intrusion_logs/'
    image = str(image)
    now = datetime.datetime.now(ZoneInfo('Asia/Taipei'))
    payload = {"image": image, "location": location, "timestamp": str(now), "status": 'not_success'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        log.info(f"API Status Code: {response.status_code}")
    except Exception as e:
        log.error(f"Error during API call: {e}")

def get_location_id_from_str(cam_id_str):
    cam_num = int(cam_id_str[-3:])
    if cam_num == 111:
        return 10026
    else:
        return 10037 + (cam_num - 112)

def handle_alert_in_background(annotated_frame, cam_id):
    """
    This function runs in a background thread to handle all blocking alert operations.
    """
    log.info(f"[{cam_id}] Background alert thread started.")
    
    # 1. Trigger physical alarm
    cam_num = int(cam_id[-3:])
    alert_ip = None
    if cam_num in range(111, 116):
        alert_ip = '192.168.3.181'
    elif cam_num in range(116, 121):
        alert_ip = '192.168.3.182'
    
    if alert_ip:
        try:
            requests.get(f'http://{alert_ip}:1880/gpio_out?pin=12&st=1', timeout=2)
            time.sleep(5)
            requests.get(f'http://{alert_ip}:1880/gpio_out?pin=12&st=0', timeout=2)
            log.info(f"[{cam_id}] Alarm cycle completed.")
        except requests.exceptions.RequestException as e:
            log.error(f"[{cam_id}] Failed to trigger alarm: {e}")

    # 2. Save image
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    directory = os.path.join('./saved_images', current_date)
    file_path = save_image_with_limit(annotated_frame, directory, 'detected', cam_id)
    
    # 3. Send API alert
    if file_path and os.path.exists(file_path):
        try:
            saved_image = cv2.imread(file_path)
            if saved_image is not None:
                base64_image = image2base64(saved_image)
                location_id = get_location_id_from_str(cam_id)
                alert_api(base64_image, api, location_id)
        except Exception as e:
            log.error(f"[{cam_id}] Error processing saved image for API: {e}")

def camera_process_worker(rtsp_link, cam_id, danger_zone, display_queue, stop_event):
    log.info(f"[{cam_id}] Process started.")
    cam = Camera(rtsp_link)
    model = YOLOv10(int8_model_det_path, task='detect')
    
    # State variables for cooldown
    last_alert_time = 0
    cooldown_period = 5
    
    temp_stderr_file = f"temp_stderr_{os.getpid()}.txt"

    while not stop_event.is_set():
        try:
            tz = ZoneInfo('Asia/Taipei')
            now = datetime.datetime.now(tz)
            if not (8 <= now.hour < 18):
                time.sleep(30)
                continue

            t_start = time.time()
            data = cam.get_data()
            if data is None:
                time.sleep(1)
                continue

            # Decode and validate frame
            is_corrupt = False
            frame = None
            try:
                with stderr_redirected_to_file(temp_stderr_file):
                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                stderr_output = ""
                if os.path.exists(temp_stderr_file):
                    with open(temp_stderr_file, 'r') as f:
                        stderr_output = f.read()
                if "Corrupt JPEG data" in stderr_output:
                    is_corrupt = True
                    log.warning(f"[{cam_id}] Frame discarded: Corrupt JPEG data detected.")
            finally:
                if os.path.exists(temp_stderr_file):
                    os.remove(temp_stderr_file)

            if frame is None or is_corrupt:
                continue

            # --- Start of additional validation checks ---

            # Check for blurry images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Threshold from handover notes
                log.warning(f"[{cam_id}] Frame discarded: Blurry image detected (Laplacian Var: {laplacian_var:.2f}).")
                continue
            
            # --- End of additional validation checks ---

            frame = cv2.resize(frame, (1280, 720))

            # Always run detection
            t_pre_infer = time.time()
            results = model(source=frame, iou=0.5, conf=0.55, verbose=False)[0]
            t_post_infer = time.time()
            log.info(f"[{cam_id}] Inference time: {(t_post_infer - t_pre_infer) * 1000:.2f} ms")

            # Check for intrusion
            bboxes = []
            train_bboxes = [result.xyxy[0] for result in results.boxes if int(result.cls[0]) == 1]
            for result in results.boxes:
                bbox = result.xyxy[0]
                cls = int(result.cls[0])
                if cls == 1 or any(calculate_overlap_ratio(bbox, train_bbox) > 0.8 for train_bbox in train_bboxes):
                    continue
                else:
                    bboxes.append(bbox)
            is_intrusion = bboxes and check_bboxes_in_danger_zone(danger_zone, bboxes)

            # Check cooldown status
            current_time = time.time()
            is_in_cooldown = (current_time - last_alert_time) <= cooldown_period

            # --- Alerting Logic ---
            if is_intrusion and not is_in_cooldown:
                last_alert_time = current_time
                annotated_frame_for_alert = results.plot()
                
                # Launch background thread for all blocking alert tasks
                alert_thread = threading.Thread(
                    target=handle_alert_in_background,
                    args=(annotated_frame_for_alert, cam_id),
                    daemon=True
                )
                alert_thread.start()

            # --- Display Logic ---
            display_frame = None
            if is_in_cooldown:
                # During cooldown, show the clean live frame
                display_frame = frame.copy()
            else:
                # After cooldown, show annotated frame if intrusion, else clean frame
                if is_intrusion:
                    display_frame = results.plot()
                else:
                    display_frame = frame.copy()
            
            final_display_frame = draw_transparent_polygon(display_frame, danger_zone.exterior)
            
            if not display_queue.full():
                display_queue.put((cam_id, final_display_frame))
            
            loop_duration = time.time() - t_start
            log.info(f"[{cam_id}] Total loop time: {loop_duration * 1000:.2f} ms")

        except Exception as e:
            log.error(f"[{cam_id}] Unhandled exception in worker process: {e}", exc_info=True)
            time.sleep(5)

def main():
    active_camera_ids = [
        "1921683111", "1921683113", "1921683115", "1921683118", "1921683120"
    ]

    rtsp_links = [f"http://192.168.3.201:9080/image/{cam_id}%2Ejpg" for cam_id in active_camera_ids]
    area_files = [f'./mask/{cam_id}.txt' for cam_id in active_camera_ids]
    
    danger_zones = read_areas(area_files)

    display_queue = Queue(maxsize=len(active_camera_ids) * 2)
    stop_event = Event()

    processes = []
    for i, cam_id in enumerate(active_camera_ids):
        process = Process(
            target=camera_process_worker,
            args=(rtsp_links[i], cam_id, danger_zones[i], display_queue, stop_event),
            daemon=True
        )
        processes.append(process)
        process.start()

    log.info("All camera processes started. Starting display loop.")

    latest_frames = {}
    window_names = {cam_id: f'Camera {cam_id}' for cam_id in active_camera_ids}

    try:
        while not stop_event.is_set():
            while not display_queue.empty():
                try:
                    cam_id, frame = display_queue.get_nowait()
                    latest_frames[cam_id] = frame
                except Exception:
                    break

            for cam_id, frame in latest_frames.items():
                window_name = window_names[cam_id]
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                log.info("Quit signal received. Shutting down.")
                stop_event.set()
                break
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received. Shutting down.")
        stop_event.set()

    finally:
        log.info("Cleaning up processes...")
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                log.warning(f"Process {process.pid} did not terminate gracefully. Terminating.")
                process.terminate()
        
        cv2.destroyAllWindows()
        log.info("Shutdown complete.")

if __name__ == '__main__':
    main()