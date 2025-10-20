import cv2
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from time import sleep
from PIL import ImageTk, Image
import numpy as np
from functools import partial
from socket import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import configparser
config = configparser.ConfigParser()
folder_path = "/home/gini-facetest/rail_obstacle"
#folder_path = '/home/gini/ftp/intrusion_detect_v102/' # 這個資料夾的路徑
#camera_ip = '192.168.1.101' # 攝影機的ip
#camera_ip = '192.168.1.104'
camera_ip = '1921683120'
pictures_path = f"{folder_path}/image/{camera_ip}.jpg" # 從攝影機擷取下來的圖片
resolution = (1280,720) # 修改圖片的size
mask_folder_path = f"{folder_path}/mask" # 圖片遮罩位置
coordinate_txt_path = f"{mask_folder_path}/{camera_ip}.txt" # 遮罩座標點位置

try:
    frame = cv2.imread(pictures_path)
    frame = cv2.resize(frame, resolution)
except:
    print('無法讀取圖片')
    exit()

ROI_name = "draw area of site - Double click to finish/Right click to clear Trajectory"
crop = None
copy = None
lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
count = 0
pointsMax = 6
ROI_bymouse_flag = 1
keep_point = []
x_arr = []
y_arr = []
def Electronic_fence_conf():
    global frame, crop, copy, ROI_name, ROI_name
    crop = frame
    copy = crop.copy()
    cv2.namedWindow(ROI_name)
    cv2.setMouseCallback(ROI_name, on_mouse)
    cv2.imshow(ROI_name, crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def on_mouse(event, x, y, flags, param):
    global point1, point2, pointsCount, lsPointsChoose, tpPointsChoose, crop, copy
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsCount = pointsCount + 1
        point1 = (x, y)
        cv2.circle(crop, point1, 10, (0, 255, 0), 2)
        # print(point1)
        keep_point.append(point1)
        lsPointsChoose.append([x, y])
        tpPointsChoose.append((x, y))
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(crop, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        cv2.imshow(ROI_name, crop)

    if event == cv2.EVENT_RBUTTONDOWN:
        print("clean all!")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(crop, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        crop = copy.copy()
        cv2.imshow(ROI_name, crop)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        ROI_byMouse()
        ROI_bymouse_flag = 1
        lsPointsChoose = []
        
def ROI_byMouse():
    global ROI, ROI_flag, mask2, x_arr, y_arr
    mask = np.zeros(crop.shape, np.uint8)
    pts = np.array([lsPointsChoose], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.polylines(mask, [pts], True, (0, 255, 255), thickness=10)
    mask2 = cv2.fillPoly(mask, [pts], ( 28, 36, 235))
    f = open(coordinate_txt_path, 'w')
    for l in lsPointsChoose:
        f.write(str(l[0]) + ',' +str(l[1]) + '\n')
        x_arr.append(l[0])
        y_arr.append(l[1])
    f.close()

    x_arr = sorted(x_arr)
    print(x_arr, x_arr[0], x_arr[-1])
    y_arr = sorted(y_arr)
    print(y_arr, y_arr[0], y_arr[-1])

    text = 'Danger Area'
    cv2.putText(mask2, text, (int((x_arr[-1] - x_arr[0])/2)-10, int((y_arr[-1] - y_arr[0])/2)), cv2.FONT_HERSHEY_DUPLEX,
        1, (0, 255, 255), 2, cv2.LINE_AA)
    #print(lsPointsChoose)

    # cv2.imwrite(folder_path + "/dangerous_area_mask.jpg", mask2)
    cv2.imwrite(f"{mask_folder_path}/{camera_ip}.jpg", mask2)
    cv2.destroyAllWindows()

Electronic_fence_conf()
