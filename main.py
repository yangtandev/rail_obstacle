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
from shapely.errors import GEOSException
from shapely.ops import unary_union
try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None
import datetime
import base64
import threading
import json
import signal
from camera import Camera

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

def load_config(config_path=None):
    """從 config.json 讀取部署設定"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.json'
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        log.error(f"找不到設定檔: {config_path}")
        log.error("請先執行 sudo ./install.sh 或手動編輯 config.json")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
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
    image_files = glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png"))
    if len(image_files) >= limit:
        oldest_image = min(image_files, key=os.path.getctime)
        os.remove(oldest_image)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    image_path = os.path.join(directory, f"{folder_name}_cam{cam_id}_{timestamp}.png")
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
                point = (x, y)
                if not points or points[-1] != point:
                    points.append(point)

        polygon = Polygon(points)
        if not polygon.is_valid:
            fixed = make_valid(polygon) if make_valid else polygon.buffer(0)
            if fixed.geom_type == 'GeometryCollection':
                polygons_only = [geom for geom in fixed.geoms if geom.geom_type in ('Polygon', 'MultiPolygon')]
                fixed = unary_union(polygons_only) if polygons_only else fixed
            if not fixed.is_empty and fixed.is_valid and fixed.geom_type in ('Polygon', 'MultiPolygon'):
                polygon = fixed
                log.warning(f"{file_path} polygon invalid; repaired for runtime use.")
            else:
                log.error(f"{file_path} polygon invalid and cannot be repaired.")
        polygons.append(polygon)
    return polygons

def check_bboxes_in_danger_zone(danger_area_polygon, bboxes, iou_threshold=0.2):
    for bbox in bboxes:
        bbox_poly = box(*bbox)
        try:
            if not danger_area_polygon.intersects(bbox_poly):
                continue
            intersection = danger_area_polygon.intersection(bbox_poly)
        except GEOSException as e:
            log.warning(f"Invalid danger zone geometry skipped for bbox: {e}")
            continue
        intersection_area = intersection.area
        bbox_area = bbox_poly.area
        if bbox_area > 0:
            ratio = intersection_area / bbox_area
            if ratio > iou_threshold:
                x1, y1, x2, y2 = bbox
                bbox_height = y2 - y1
                try:
                    inter_minx, inter_miny, inter_maxx, inter_maxy = intersection.bounds
                    # 只要腳底部距離交集區端點大於身高 10% (代表物件在紅色區域外下方的面積達整體的 10% 以上)，就濾除
                    if inter_maxy < y2 - (bbox_height * 0.1):
                        continue
                except Exception as e:
                    pass
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
    if points is None:
        return image
    if hasattr(points, 'geoms'):
        for geom in points.geoms:
            if geom.geom_type == 'Polygon':
                cv2.fillPoly(overlay, [np.array(geom.exterior.coords, dtype=np.int32)], color)
        cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
        return output
    if hasattr(points, 'exterior'):
        points = points.exterior
    if hasattr(points, 'coords'):
        points = list(points.coords)
    if len(points) > 0:
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



def handle_alert_in_background(annotated_frame, cam_id, api_url, alert_device_ip, location_id, raw_frame=None, debug_info=None):
    """
    This function runs in a background thread to handle all blocking alert operations.
    """
    log.info(f"[{cam_id}] Background alert thread started.")

    # 1. Trigger physical alarm
    if alert_device_ip:
        try:
            requests.get(f'http://{alert_device_ip}:1880/gpio_out?pin=12&st=1', timeout=2)
            time.sleep(5)
            requests.get(f'http://{alert_device_ip}:1880/gpio_out?pin=12&st=0', timeout=2)
            log.info(f"[{cam_id}] Alarm cycle completed.")
        except requests.exceptions.RequestException as e:
            log.error(f"[{cam_id}] Failed to trigger alarm: {e}")

    # 2. Save image
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    directory = os.path.join('./saved_images', current_date)
    file_path = save_image_with_limit(annotated_frame, directory, 'detected', cam_id)

    # 2.5 Save debug info
    if file_path and raw_frame is not None and debug_info is not None:
        try:
            debug_dir = os.path.join(directory, 'debug')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            basename = os.path.splitext(os.path.basename(file_path))[0]
            raw_image_path = os.path.join(debug_dir, f"{basename}_raw.png")
            cv2.imwrite(raw_image_path, raw_frame)

            json_path = os.path.join(debug_dir, f"{basename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, ensure_ascii=False, indent=4)
        except Exception as e:
            log.error(f"[{cam_id}] Error saving debug info: {e}")

    # 3. Send API alert
    if file_path and os.path.exists(file_path):
        try:
            saved_image = cv2.imread(file_path)
            if saved_image is not None:
                base64_image = image2base64(saved_image)
                alert_api(base64_image, api_url, location_id)
        except Exception as e:
            log.error(f"[{cam_id}] Error processing saved image for API: {e}")

def camera_process_worker(rtsp_link, cam_id, danger_zone, display_queue, stop_event, enable_recording, api_url, alert_device_ip, location_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log.info(f"[{cam_id}] Process started. 準備連線 RTSP...")
    transports = ('tcp', 'udp')
    transport_index = 0
    cam = Camera(rtsp_link, transports[transport_index])

    preview_deadline = time.time() + 5
    while not stop_event.is_set() and time.time() < preview_deadline:
        frame = cam.get_data()
        if frame is not None:
            preview_frame = cv2.resize(frame, (1280, 720))
            preview_frame = draw_transparent_polygon(preview_frame, danger_zone)
            if not display_queue.full():
                display_queue.put((cam_id, preview_frame))
            break
        time.sleep(0.1)

    log.info(f"[{cam_id}] RTSP 連線完成. 準備載入模型...")
    model = YOLOv10(int8_model_det_path, task='detect')

    log.info(f"[{cam_id}] 模型載入完成. 進入影像處理迴圈.")

    last_alert_time = 0
    cooldown_period = 5

    no_frame_counter = 0
    no_frame_sleep = 0.2
    reconnect_after_seconds = 60
    first_no_frame_time = None
    last_no_frame_log = 0

    # 錄影相關變數
    video_writer = None
    current_record_hour = None
    if enable_recording:
        record_dir = "./records"
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

    # 歷史遮罩追蹤參數 (Temporal Tracking Mask)
    train_history_bboxes = []
    train_history_ttl = 0
    MAX_TTL = 15  # 記憶存活幀數 (假設約為 1 秒)

    try:
        while not stop_event.is_set():
            try:
                tz = ZoneInfo('Asia/Taipei')
                now = datetime.datetime.now(tz)

                # 下班時間安全收尾
                # if not (8 <= now.hour < 18):
                #     if video_writer is not None:
                #         video_writer.release()
                #         video_writer = None
                #         current_record_hour = None
                #         log.info(f"[{cam_id}] ⏹️ 進入非辨識時段，自動停止錄影並封裝存檔。")
                #     time.sleep(30)
                #     continue

                t_start = time.time()

                frame = cam.get_data()

                if frame is None:
                    if not cam.is_opened():
                        transport_index = (transport_index + 1) % len(transports)
                        log.warning(f"[{cam_id}] RTSP 尚未開啟，5 秒後改用 {transports[transport_index]} 重連")
                        cam.release()
                        time.sleep(5)
                        cam = Camera(rtsp_link, transports[transport_index])
                        no_frame_counter = 0
                        first_no_frame_time = None
                        last_no_frame_log = 0
                        continue

                    no_frame_counter += 1
                    now_ts = time.time()
                    if first_no_frame_time is None:
                        first_no_frame_time = now_ts
                    elapsed_no_frame = now_ts - first_no_frame_time
                    remaining = max(0, int(reconnect_after_seconds - elapsed_no_frame))
                    if no_frame_counter == 1 or now_ts - last_no_frame_log >= 15:
                        log.warning(f"[{cam_id}] 等待 RTSP 影像中，{remaining} 秒後仍無影像才重連")
                        last_no_frame_log = now_ts

                    if elapsed_no_frame >= reconnect_after_seconds:
                        transport_index = (transport_index + 1) % len(transports)
                        log.error(f"[{cam_id}] 超過 {reconnect_after_seconds} 秒無影像，改用 {transports[transport_index]} 重連...")
                        cam.release()
                        time.sleep(0.5)
                        cam = Camera(rtsp_link, transports[transport_index])
                        no_frame_counter = 0
                        first_no_frame_time = None
                        last_no_frame_log = 0

                    time.sleep(no_frame_sleep)
                    continue

                if no_frame_counter:
                    log.info(f"[{cam_id}] RTSP 影像已恢復 ({transports[transport_index]})")
                no_frame_counter = 0
                first_no_frame_time = None
                last_no_frame_log = 0

                frame = cv2.resize(frame, (1280, 720))

                # 恢復全域預設門檻 0.45
                results = model(source=frame, iou=0.5, conf=0.45, verbose=False)[0]

                # 建立列車遮罩 (此處已全是大於 0.45 的結果)
                current_train_bboxes = [result.xyxy[0] for result in results.boxes if int(result.cls[0]) == 1]

                # --- 時序追蹤 (Temporal Tracking) 邏輯 ---
                if current_train_bboxes:
                    # 如果當前幀有抓到輕軌，更新記憶庫並重置 TTL
                    train_history_bboxes = current_train_bboxes
                    train_history_ttl = MAX_TTL
                else:
                    # 如果當前幀沒抓到，但還有 TTL 壽命，則扣除壽命並延用上一幀的遮罩
                    if train_history_ttl > 0:
                        train_history_ttl -= 1
                    else:
                        train_history_bboxes = []

                # 最終有效遮罩 (來自當前或歷史記憶)
                active_train_bboxes = train_history_bboxes
                bboxes = []

                for result in results.boxes:
                    bbox = result.xyxy[0]
                    cls = int(result.cls[0])
                    conf = float(result.conf[0])

                    # 過濾邏輯
                    if cls == 1:
                        # 列車(Train)本身不為入侵告警目標
                        continue
                    else:
                        box_width = bbox[2] - bbox[0]
                        box_height = bbox[3] - bbox[1]
                        frame_height = frame.shape[0]
                        frame_width = frame.shape[1]

                        # 尺寸過濾限制
                        if box_width > frame_width * 0.5 or box_height > frame_height * 0.5:
                            continue

                        # 邊緣防禦過濾 (針對畫面左右外側 10% 範圍，需要更高的信心度 > 0.75)
                        center_x = (bbox[0] + bbox[2]) / 2.0
                        if center_x <= frame_width * 0.1 or center_x >= frame_width * 0.9:
                            if conf < 0.75:
                                continue # 在邊緣且信心度不足，視為邊緣雜訊誤判拋棄

                        # 是否與任何被捕捉到的輕軌高度重疊 (包含正在記憶體存活的歷史車影)
                        if any(calculate_overlap_ratio(bbox, train_bbox) > 0.8 for train_bbox in active_train_bboxes):
                            continue

                        # 通過所有嚴格檢驗，加入有效入侵檢測框
                        bboxes.append(bbox)
                is_intrusion = bboxes and check_bboxes_in_danger_zone(danger_zone, bboxes)

                current_time = time.time()
                is_in_cooldown = (current_time - last_alert_time) <= cooldown_period

                # Alerting Logic
                if is_intrusion and not is_in_cooldown:
                    last_alert_time = current_time
                    annotated_frame_for_alert = results.plot()

                    annotated_frame_for_alert = draw_transparent_polygon(annotated_frame_for_alert, danger_zone)

                    try:
                        debug_info = {
                            "cam_id": cam_id,
                            "timestamp": now.isoformat(),
                            "bboxes": [[float(x) for x in box.xyxy[0]] for box in results.boxes],
                            "confidences": [float(box.conf[0]) for box in results.boxes],
                            "classes": [int(box.cls[0]) for box in results.boxes],
                            "final_intrusion_bboxes": [[float(x) for x in bbox] for bbox in bboxes]
                        }
                    except Exception as e:
                        log.error(f"[{cam_id}] Error preparing debug info: {e}")
                        debug_info = {}

                    alert_thread = threading.Thread(
                        target=handle_alert_in_background,
                        args=(annotated_frame_for_alert, cam_id, api_url, alert_device_ip, location_id, frame.copy(), debug_info),
                        daemon=True
                    )
                    alert_thread.start()

                # --- 修改後的 Display Logic：永遠顯示 YOLO 的辨識框 ---
                display_frame = results.plot()
                final_display_frame = draw_transparent_polygon(display_frame, danger_zone)

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
    config = load_config()
    api_url = config['api_url']
    enable_recording = config.get('enable_recording', False)
    cameras = config['cameras']

    active_camera_ids = [cam['id'] for cam in cameras]
    area_files = [f'./mask/{cam_id}.txt' for cam_id in active_camera_ids]

    danger_zones = read_areas(area_files)

    display_queue = Queue(maxsize=len(cameras) * 2)
    stop_event = Event()

    if enable_recording:
        log.info("系統設定：自動錄影功能已啟用 (8~18點間將自動分段錄影)")
    else:
        log.info("系統設定：自動錄影功能已停用")

    processes = []
    for i, cam in enumerate(cameras):
        process = Process(
            target=camera_process_worker,
            args=(cam['rtsp_url'], cam['id'], danger_zones[i], display_queue, stop_event, enable_recording,
                  api_url, cam.get('alert_device_ip'), cam.get('location_id')),
            daemon=True
        )
        processes.append(process)
        process.start()
        time.sleep(2)

    log.info("All camera processes started. Starting display loop.")

    latest_frames = {}
    for cam_id in active_camera_ids:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"Waiting for {cam_id}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        latest_frames[cam_id] = frame
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
        deadline = time.time() + 5
        for process in processes:
            process.join(timeout=max(0, deadline - time.time()))

        for process in processes:
            if process.is_alive():
                log.warning(f"Process {process.pid} did not terminate gracefully. Terminating.")
                process.terminate()

        for process in processes:
            if process.is_alive():
                process.join(timeout=2)

        cv2.destroyAllWindows()
        log.info("Shutdown complete.")

if __name__ == '__main__':
    main()
