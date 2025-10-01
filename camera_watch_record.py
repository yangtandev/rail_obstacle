

import cv2
import time
import threading
from datetime import datetime

# 攝影機的RTSP串流連結列表
camera_urls = [
#     'rtsp://admin:!QAZ87518499@192.168.3.104',
     'rtsp://admin:!QAZ87518499@192.168.3.118',
]

# 每台攝影機執行串流播放和錄影的函數
def process_camera_stream(url, camera_id):
    connection = cv2.VideoCapture(url)
    
    # 取得當前時間並格式化作為檔案名稱的一部分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'camera_{camera_id}_{timestamp}.avi'
    
    # 影片寫入器初始化
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 15, (1920, 1080))
    
    while True:
        # 如果camera的連接成功
        if connection.isOpened():
            ret, frame = connection.read()
        else:
            print(f'Camera {camera_id} disconnected...')
            connection.release()
            time.sleep(10)
            print(f'Reconnecting camera {camera_id}...')
            connection = cv2.VideoCapture(url)
            continue

        # 如果接收到影像資料
        if ret:
            frame = cv2.resize(frame, (1920, 1080))  # 調整尺寸，視需求而定
            
            # 顯示每幀影像
            cv2.imshow(f'Camera {camera_id}', frame)
            
            # 將影像寫入影片檔
            out.write(frame)

            # 若按下 'q' 鍵，則退出顯示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f'Cannot get frame from camera {camera_id}')
            connection.release()
            time.sleep(10)
            print(f'Reconnecting camera {camera_id}...')
            connection = cv2.VideoCapture(url)

    # 釋放資源
    connection.release()
    out.release()
    cv2.destroyAllWindows()

# 為每台攝影機創建一個執行緒
threads = []
for idx, url in enumerate(camera_urls):
    thread = threading.Thread(target=process_camera_stream, args=(url, idx + 1))
    thread.start()
    threads.append(thread)

# 等待所有執行緒結束
for thread in threads:
    thread.join()
