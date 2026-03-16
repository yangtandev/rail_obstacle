import numpy as np
import cv2
from ultralytics import YOLOv10
from pathlib import Path
import logging as log
import sys
import os
import glob
import time
from multiprocessing import Queue, Process, Event
import requests
from zoneinfo import ZoneInfo
from shapely.geometry import Polygon, box
import datetime
import base64
import threading
from camera import Camera

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# ================= 系統全域設定 =================
ENABLE_RECORDING = True  # 控制是否啟用自動錄影 (True: 啟用, False: 停用)
# ================================================

api = "https://jenyi-xg.api.ginibio.com/api/v1"
log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S', 
    level=log.INFO, 
    stream=sys.stdout
)
models_dir = Path('./models')
model_name = "rail_obstacle"
int8_model_det_path = models_dir / 'int8' / f'{model_name}_openvino_model'

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

def camera_process_worker(rtsp_link, cam_id, danger_zone, display_queue, stop_event, enable_recording):
    log.info(f"[{cam_id}] Process started. 準備連線 RTSP...")
    cam = Camera(rtsp_link)
    
    log.info(f"[{cam_id}] RTSP 連線完成. 準備載入模型...")
    model = YOLOv10(int8_model_det_path, task='detect')
    
    log.info(f"[{cam_id}] 模型載入完成. 進入影像處理迴圈.")
    
    last_alert_time = 0
    cooldown_period = 5
    
    no_frame_counter = 0
    reconnect_threshold = 10 

    # 錄影相關變數
    video_writer = None
    current_record_hour = None 
    if enable_recording:
        record_dir = "./records"
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

    try:
        while not stop_event.is_set():
            try:
                tz = ZoneInfo('Asia/Taipei')
                now = datetime.datetime.now(tz)
                
                # 下班時間安全收尾
                if not (8 <= now.hour < 18):
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        current_record_hour = None
                        log.info(f"[{cam_id}] ⏹️ 進入非辨識時段，自動停止錄影並封裝存檔。")
                    time.sleep(30)
                    continue

                t_start = time.time()
                
                frame = cam.get_data()
                
                if frame is None:
                    no_frame_counter += 1
                    log.warning(f"[{cam_id}] 警告: 無法取得影像 ({no_frame_counter}/{reconnect_threshold})...")
                    
                    if no_frame_counter >= reconnect_threshold:
                        log.error(f"[{cam_id}] 影像中斷過久，釋放資源並嘗試重新連線...")
                        cam.release()
                        time.sleep(2)
                        cam = Camera(rtsp_link)
                        no_frame_counter = 0
                        
                    time.sleep(1)
                    continue
                
                no_frame_counter = 0

                frame = cv2.resize(frame, (1280, 720))

                results = model(source=frame, iou=0.5, conf=0.55, verbose=False)[0]

                bboxes = []
                train_bboxes = [result.xyxy[0] for result in results.boxes if int(result.cls[0]) == 1]
                for result in results.boxes:
                    bbox = result.xyxy[0]
                    cls = int(result.cls[0])

                    box_width = bbox[2] - bbox[0]
                    box_height = bbox[3] - bbox[1]
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    if box_width > frame_width * 0.5 or box_height > frame_height * 0.5:
                        continue

                    if cls == 1 or any(calculate_overlap_ratio(bbox, train_bbox) > 0.8 for train_bbox in train_bboxes):
                        continue
                    else:
                        bboxes.append(bbox)
                is_intrusion = bboxes and check_bboxes_in_danger_zone(danger_zone, bboxes)

                current_time = time.time()
                is_in_cooldown = (current_time - last_alert_time) <= cooldown_period

                # Alerting Logic
                if is_intrusion and not is_in_cooldown:
                    last_alert_time = current_time
                    annotated_frame_for_alert = results.plot()
                    
                    annotated_frame_for_alert = draw_transparent_polygon(annotated_frame_for_alert, danger_zone.exterior)
                    
                    alert_thread = threading.Thread(
                        target=handle_alert_in_background,
                        args=(annotated_frame_for_alert, cam_id),
                        daemon=True
                    )
                    alert_thread.start()

                # --- 修改後的 Display Logic：永遠顯示 YOLO 的辨識框 ---
                display_frame = results.plot()
                final_display_frame = draw_transparent_polygon(display_frame, danger_zone.exterior)
                
                if not display_queue.full():
                    display_queue.put((cam_id, final_display_frame))
                
                # 自動換檔與寫入錄影
                if enable_recording:
                    current_hour = now.hour
                    
                    # 發現跨越到下一個小時了，先關閉目前的寫入器
                    if video_writer is not None and current_record_hour != current_hour:
                        video_writer.release()
                        video_writer = None
                        log.info(f"[{cam_id}] 🕛 跨小時換檔，前一段影片已自動存檔。")

                    # 若沒有寫入器，就建一個新的
                    if video_writer is None:
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        record_path = os.path.join(record_dir, f"record_cam{cam_id}_{timestamp}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(record_path, fourcc, 15.0, (1920, 1080))
                        current_record_hour = current_hour
                        log.info(f"[{cam_id}] 🔴 開始錄製此小時區段影片: {record_path}")
                    
                    record_frame = cv2.resize(final_display_frame, (1920, 1080))
                    video_writer.write(record_frame)

            except Exception as e:
                log.error(f"[{cam_id}] Unhandled exception in worker process: {e}", exc_info=True)
                time.sleep(5)
    finally:
        if video_writer is not None:
            video_writer.release()
        cam.release()

def main():
    active_camera_ids = [
        "1921683111", 
        #"1921683113", 
        #"1921683115", 
        #"1921683118", 
        "1921683120"
    ]

    rtsp_links = [
        "rtsp://192.168.3.201:9554/live/192.168.3.111",
        #"rtsp://192.168.3.201:9554/live/192.168.3.113",
        #"rtsp://192.168.3.201:9554/live/192.168.3.115",
        #"rtsp://192.168.3.201:9554/live/192.168.3.118",
        "rtsp://192.168.3.201:9554/live/192.168.3.120"
    ]
    area_files = [f'./mask/{cam_id}.txt' for cam_id in active_camera_ids]
    
    danger_zones = read_areas(area_files)

    display_queue = Queue(maxsize=len(active_camera_ids) * 2)
    stop_event = Event()
    
    if ENABLE_RECORDING:
        log.info("系統設定：自動錄影功能已啟用 (8~18點間將自動分段錄影)")
    else:
        log.info("系統設定：自動錄影功能已停用")

    processes = []
    for i, cam_id in enumerate(active_camera_ids):
        process = Process(
            target=camera_process_worker,
            args=(rtsp_links[i], cam_id, danger_zones[i], display_queue, stop_event, ENABLE_RECORDING),
            daemon=True
        )
        processes.append(process)
        process.start()
        time.sleep(2)
        
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
            
            # 保留手動截圖快捷鍵 's'
            elif key in [ord('s'), ord('S')]:
                save_dir = "./exhibition_shots"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                for c_id, f in latest_frames.items():
                    filename = os.path.join(save_dir, f"exhibition_cam{c_id}_{timestamp}.jpg")
                    cv2.imwrite(filename, f)
                log.info(f"📸 參展截圖已儲存至 {save_dir} 資料夾！")
            
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