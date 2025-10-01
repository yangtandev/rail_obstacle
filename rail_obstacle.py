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
from multiprocessing import Queue, Process
import base64
import requests
from zoneinfo import ZoneInfo
from shapely.geometry import Polygon, box
import datetime
import concurrent.futures
import threading

api = "https://jenyi-xg.api.ginibio.com/api/v1"
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
models_dir = Path('./models')
model_name = "rail_obstacle"
int8_model_det_path = models_dir / 'int8' / f'{model_name}_openvino_model/{model_name}.xml'

# 載入模型
try:
    ov_yolo_int8_model = YOLOv10(int8_model_det_path.parent, task='detect')
    model = ov_yolo_int8_model
    log.info(f'Model loaded successfully: {model}')
except Exception as e:
    log.error(f"Failed to load model: {e}")
    sys.exit(1)

#  設置圖片儲存的限額並防止檔名重複
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
    # Assuming cam_id_str is like '...111', '...112'
    cam_num = int(cam_id_str[-3:])
    if cam_num == 111:
        return 10026
    else:
        # This logic might need adjustment based on full range of camera IDs
        return 10037 + (cam_num - 112)

def process_detection_task(args):
    frame, cam_id_str, danger_zone, model, lock, last_alert_times = args
    
    results = model(source=frame, iou=0.5, conf=0.35, verbose=False)[0]
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
        cooldown_period = 10
        current_time = time.time()
        
        with lock:
            last_alert_time = last_alert_times.get(cam_id_str, 0)
            if (current_time - last_alert_time) > cooldown_period:
                last_alert_times[cam_id_str] = current_time
                
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

    final_frame = draw_transparent_polygon(annotated_frame, danger_zone.exterior.coords)
    return cam_id_str, final_frame

def main_loop(queues, danger_zones, model, active_camera_ids):
    num_workers = os.cpu_count()
    log.info(f"Initializing detection pool with {num_workers} workers.")

    alert_lock = threading.Lock()
    last_alert_times = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_cam = {}
        
        while True:
            for idx, cam_id_str in enumerate(active_camera_ids):
                if not queues[idx].empty():
                    frame = queues[idx].get()
                    if frame is not None:
                        args = (frame, cam_id_str, danger_zones[idx], model, alert_lock, last_alert_times)
                        future = executor.submit(process_detection_task, args)
                        future_to_cam[future] = cam_id_str

            try:
                done_futures = concurrent.futures.as_completed(future_to_cam, timeout=0.01)
                for future in done_futures:
                    cam_id_str_future = future_to_cam.pop(future)
                    try:
                        cam_id_result, processed_frame = future.result()
                        cv2.imshow(f'Camera {cam_id_result}', processed_frame)
                    except Exception as exc:
                        log.error(f'Camera {cam_id_str_future} generated an exception: {exc}')
            except concurrent.futures.TimeoutError:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    active_camera_ids = [
        "1921683111", "1921683112", "1921683113", "1921683114", "1921683115",
        "1921683116", "1921683117", "1921683118", "1921683119", "1921683120"
    ]
    # Comment out cameras you want to disable
    # active_camera_ids = ["1921683111", "1921683120"]

    rtsp_links = [f"http://111.70.11.75:9080/image/{cam_id}%2Ejpg" for cam_id in active_camera_ids]
    area_files = [f'./mask/{cam_id}.txt' for cam_id in active_camera_ids]
    
    danger_zones = read_areas(area_files)

    manager = multiprocessing.Manager()
    Hailey = manager.Namespace()
    Hailey.coordinate = None

    queues = [Queue() for _ in range(len(rtsp_links))]

    processes = []
    for i, rtsp_link in enumerate(rtsp_links):
        process_frame_instance = Process(target=process_frame, daemon=True, args=(rtsp_link, queues[i], active_camera_ids[i], Hailey))
        processes.append(process_frame_instance)

    for p in processes:
        p.start()

    main_loop(queues, danger_zones, model, active_camera_ids)