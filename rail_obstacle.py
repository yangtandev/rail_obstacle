import numpy as np
import cv2
from ultralytics import YOLOv10
from pathlib import Path
import logging as log
import sys
import os
import glob
import time
from frame4 import process_frame
import multiprocessing
from multiprocessing import Queue, Process, Event

import base64
import requests
from zoneinfo import ZoneInfo
from shapely.geometry import Polygon, box
import datetime
import concurrent.futures
import threading
import contextlib
import queue # New import for threading.Queue

api = "https://jenyi-xg.api.ginibio.com/api/v1"
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
models_dir = Path('./models')
model_name = "rail_obstacle"
int8_model_det_path = models_dir / 'int8' / f'{model_name}_openvino_model'

# Thread-local storage to hold a model instance for each thread
thread_local_data = threading.local()

@contextlib.contextmanager
def stderr_redirected(lock, to=os.devnull):
    lock.acquire()
    fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        sys.stderr.close()
        os.dup2(to_fd, fd)

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(file.fileno())
        try:
            yield
        finally:
            _redirect_stderr(old_stderr.fileno())
            lock.release()

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
    # Ensure points is a list of tuples/lists for fillPoly
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
        response = requests.post(url, json=payload)
        print(f"API Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error during API call: {e}")

def get_location_id_from_str(cam_id_str):
    cam_num = int(cam_id_str[-3:])
    if cam_num == 111:
        return 10026
    else:
        return 10037 + (cam_num - 112)



stderr_lock = threading.Lock()

def handle_alert_and_save(cam_id_str, alert_lock, last_alert_times, camera_queue, model_path, danger_zones_all, active_camera_ids):
    log.info(f"CAM {cam_id_str}: Intrusion detected. Starting alert/save thread.")
    
    cooldown_period = 5
    current_time = time.time()
    
    with alert_lock:
        last_alert_time = last_alert_times.get(cam_id_str, 0)
        if (current_time - last_alert_time) < cooldown_period:
            log.info(f"CAM {cam_id_str}: Alert cooldown active. Skipping this alert.")
            return
        last_alert_times[cam_id_str] = current_time

    latest_data = None
    while not camera_queue.empty():
        try:
            latest_data = camera_queue.get_nowait()
        except Exception:
            break 
    
    if latest_data is None:
        log.warning(f"CAM {cam_id_str}: No latest frame available after cooldown. Skipping alert.")
        return

    if not hasattr(thread_local_data, 'model'):
        log.info(f"Initializing model for alert thread {threading.get_ident()}...")
        try:
            thread_local_data.model = YOLOv10(model_path, task='detect')
        except Exception as e:
            log.error(f"Failed to load model in alert thread {threading.get_ident()}: {e}")
            return

    frame = cv2.imdecode(np.frombuffer(latest_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        log.warning(f"CAM {cam_id_str}: Failed to decode latest frame for alert. Skipping.")
        return
    
    frame = cv2.resize(frame, (1280, 720))
    
    results = thread_local_data.model(source=frame, iou=0.5, conf=0.55, verbose=False)[0]
    annotated_frame = results.plot()

    bboxes = []
    train_bboxes = [result.xyxy[0] for result in results.boxes if int(result.cls[0]) == 1]
    for result in results.boxes:
        bbox = result.xyxy[0]
        cls = int(result.cls[0])
        if cls == 1 or any(calculate_overlap_ratio(bbox, train_bbox) > 0.8 for train_bbox in train_bboxes):
            continue
        else:
            bboxes.append(bbox)

    danger_zone = None
    try:
        cam_index = active_camera_ids.index(cam_id_str)
        danger_zone = danger_zones_all[cam_index]
    except (ValueError, IndexError):
        log.error(f"CAM {cam_id_str}: Danger zone not found for alert thread. Skipping.")
        return

    if not (bboxes and check_bboxes_in_danger_zone(danger_zone, bboxes)):
        log.info(f"CAM {cam_id_str}: No intrusion in latest frame after cooldown. Skipping alert.")
        return 

    is_corrupt = False
    temp_stderr_file = f"temp_stderr_{threading.get_ident()}.txt"
    try:
        with stderr_redirected(stderr_lock, to=temp_stderr_file):
            _ = cv2.imdecode(np.frombuffer(latest_data, np.uint8), cv2.IMREAD_COLOR)
        
        stderr_output = ""
        if os.path.exists(temp_stderr_file):
            with open(temp_stderr_file, 'r') as f:
                stderr_output = f.read()
            os.remove(temp_stderr_file)

        if "Corrupt JPEG data" in stderr_output:
            is_corrupt = True
    except Exception as e:
        log.error(f"CAM {cam_id_str}: Exception during corruption check in alert thread: {e}")
        is_corrupt = True

    if is_corrupt:
        log.warning(f"CAM {cam_id_str}: Alert thread: Intrusion detected, but alert suppressed due to corrupt frame.")
    else:
        cam_num = int(cam_id_str[-3:])
        if cam_num in range(111, 116):
            alert_ip = '192.168.3.181'
        elif cam_num in range(116, 121):
            alert_ip = '192.168.3.182'
        else:
            alert_ip = None
        
        if alert_ip:
            try:
                response_on = requests.get(f'http://{alert_ip}:1880/gpio_out?pin=12&st=1', timeout=2)
                if response_on.status_code == 200:
                    print(f"CAM {cam_id_str}: 警報器已啟動")
                    time.sleep(5)
                    response_off = requests.get(f'http://{alert_ip}:1880/gpio_out?pin=12&st=0', timeout=2)
                    if response_off.status_code == 200:
                        print(f"CAM {cam_id_str}: 警報器已關閉")
            except requests.exceptions.RequestException as e:
                print(f"CAM {cam_id_str}: 未能輸出警報: {e}")

        current_date = datetime.datetime.now().strftime("%Y%m%d")
        directory = os.path.join('./saved_images', current_date)
        file_path = save_image_with_limit(annotated_frame, directory, 'detected', cam_id_str)
        
        if file_path and os.path.exists(file_path):
            try:
                image_data = np.fromfile(file_path, dtype=np.uint8)
                save_frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if save_frame is not None:
                    base64_image = image2base64(save_frame)
                    location_id = get_location_id_from_str(cam_id_str)
                    alert_api(base64_image, api, location_id)
            except Exception as e:
                print(f"Error processing saved image for API: {e}")

def process_detection_task(args):
    t0 = time.time()
    data, cam_id_str, danger_zone, alert_lock, last_alert_times, slow_path_executor, camera_queue, model_path, danger_zones_all, active_camera_ids = args

    if not hasattr(thread_local_data, 'model'):
        log.info(f"Initializing model for thread {threading.get_ident()}...")
        try:
            thread_local_data.model = YOLOv10(model_path, task='detect')
        except Exception as e:
            log.error(f"Failed to load model in thread {threading.get_ident()}: {e}", exc_info=True)
            return cam_id_str, np.zeros((720, 1280, 3), dtype=np.uint8)
    
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    if frame is None:
        log.warning(f"CAM {cam_id_str}: Failed to decode frame, skipping task.")
        return cam_id_str, np.zeros((720, 1280, 3), dtype=np.uint8)

    frame = cv2.resize(frame, (1280, 720))
    t_pre = time.time()
    
    results = thread_local_data.model(source=frame, iou=0.5, conf=0.55, verbose=False)[0]
    t_post = time.time()
    log.info(f"CAM {cam_id_str} [TIMING]: Model inference took {(t_post - t_pre) * 1000:.2f} ms")

    annotated_frame = results.plot()
    
    bboxes = []
    train_bboxes = [result.xyxy[0] for result in results.boxes if int(result.cls[0]) == 1]
    for result in results.boxes:
        bbox = result.xyxy[0]
        cls = int(result.cls[0])
        if cls == 1 or any(calculate_overlap_ratio(bbox, train_bbox) > 0.8 for train_bbox in train_bboxes):
            continue
        else:
            bboxes.append(bbox)

    if bboxes and check_bboxes_in_danger_zone(danger_zone, bboxes):
        slow_path_executor.submit(
            handle_alert_and_save,
            cam_id_str, alert_lock, last_alert_times, camera_queue, model_path, danger_zones_all, active_camera_ids
        )

    final_frame = draw_transparent_polygon(annotated_frame, danger_zone.exterior)
    
    return cam_id_str, final_frame

def display_worker(display_queue, stop_event):
    log.info("Display thread started.")
    windows = {}
    while not stop_event.is_set():
        try:
            latest_frames = {}
            while not display_queue.empty():
                cam_id, frame = display_queue.get_nowait()
                latest_frames[cam_id] = frame

            for cam_id, frame in latest_frames.items():
                window_name = f'Camera {cam_id}'
                if window_name not in windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(window_name, 1280, 720)
                    windows[window_name] = True
                
                cv2.imshow(window_name, frame)
        except queue.Empty:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            log.info("Quit signal received in display thread. Stopping all processes.")
            stop_event.set()
            break
    
    cv2.destroyAllWindows()
    log.info("Display thread finished.")

def main_loop(main_processes, main_stop_events, queues, danger_zones, active_camera_ids, rtsp_links, Hailey, display_queue, display_stop_event):
    num_workers = os.cpu_count()
    log.info(f"Initializing detection pool with {num_workers} workers.")

    alert_lock = threading.Lock()
    last_alert_times = {}
    is_paused = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as fast_path_executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_camera_ids)) as slow_path_executor:
            future_to_cam = {}
            
            while not display_stop_event.is_set():
                tz = ZoneInfo('Asia/Taipei')
                now = datetime.datetime.now(tz)

                if 8 <= now.hour < 18:
                    if is_paused in [True, None]:
                        log.info("Operating hours (08:00-18:00) started. Resuming detection.")
                        is_paused = False
                        if not main_processes:
                            log.info("Camera processes are not running. Starting them now...")
                            main_stop_events.clear()
                            main_stop_events.extend([Event() for _ in range(len(rtsp_links))])
                            for i, rtsp_link in enumerate(rtsp_links):
                                p = Process(target=process_frame, daemon=True, args=(rtsp_link, queues[i], active_camera_ids[i], Hailey, main_stop_events[i]))
                                main_processes.append(p)
                                p.start()
                            log.info(f"{len(main_processes)} camera processes started.")

                    # Process completed futures to free up camera slots
                    for future in list(future_to_cam):
                        if future.done():
                            cam_id_str_future = future_to_cam.pop(future)
                            try:
                                cam_id_result, processed_frame = future.result()
                                if processed_frame is not None:
                                    try:
                                        display_queue.put_nowait((cam_id_result, processed_frame))
                                    except queue.Full:
                                        pass
                            except Exception as exc:
                                log.error(f'Camera {cam_id_str_future} generated an exception: {exc}', exc_info=True)

                    # Submit new tasks for cameras that are not currently being processed
                    for idx, cam_id_str in enumerate(active_camera_ids):
                        if cam_id_str in future_to_cam.values():
                            continue  # Skip if a task for this camera is already in the pipeline

                        latest_data = None
                        while not queues[idx].empty():
                            try:
                                latest_data = queues[idx].get_nowait()
                            except queue.Empty:
                                break
                        
                        if latest_data is not None:
                            data = latest_data
                            args = (data, cam_id_str, danger_zones[idx], alert_lock, last_alert_times, slow_path_executor, queues[idx], int8_model_det_path, danger_zones, active_camera_ids)
                            future = fast_path_executor.submit(process_detection_task, args)
                            future_to_cam[future] = cam_id_str

                else:
                    if is_paused in [False, None]:
                        log.info(f"Operating hours ended. Pausing detection until 08:00. Current time: {now.strftime('%H:%M:%S')}")
                        is_paused = True
                    
                    if main_processes:
                        log.info("Stopping camera processes...")
                        for event in main_stop_events:
                            event.set()
                        for p in main_processes:
                            p.join(timeout=5)
                        main_processes.clear()
                        log.info("All camera processes have been stopped.")

                    for future in future_to_cam:
                        future.cancel()
                    future_to_cam.clear()
                    
                    for q in queues:
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                    
                    time.sleep(30)

                time.sleep(0.01) # Prevent busy-waiting

if __name__ == '__main__':
    active_camera_ids = [
        "1921683111", "1921683113", "1921683115", "1921683118", "1921683120"
    ]

    rtsp_links = [f"http://192.168.3.201:9080/image/{cam_id}%2Ejpg" for cam_id in active_camera_ids]
    area_files = [f'./mask/{cam_id}.txt' for cam_id in active_camera_ids]
    
    danger_zones = read_areas(area_files)

    manager = multiprocessing.Manager()
    Hailey = manager.Namespace()
    Hailey.coordinate = None

    queues = [Queue() for _ in range(len(rtsp_links))]
    display_queue = queue.Queue(maxsize=len(active_camera_ids) * 2)

    main_stop_events = [Event() for _ in range(len(rtsp_links))]
    display_stop_event = threading.Event()

    display_thread = threading.Thread(target=display_worker, args=(display_queue, display_stop_event))
    display_thread.start()

    processes = []
    
    try:
        main_loop(processes, main_stop_events, queues, danger_zones, active_camera_ids, rtsp_links, Hailey, display_queue, display_stop_event)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received. Shutting down.")
    finally:
        log.info("Ensuring all processes and threads are stopped before exiting.")
        display_stop_event.set()
        for event in main_stop_events:
            event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
        display_thread.join(timeout=5)
        log.info("Shutdown complete.")
