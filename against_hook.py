import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import uuid

def get_transform(image_height, image_width):
    # 根據圖片尺寸定義裁剪大小
    crop_height = min(640, image_height)
    crop_width = min(640, image_width)

    transform = A.Compose([
    A.HorizontalFlip(p=0.5), # 僅水平翻轉
    #A.VerticalFlip(p=0.5), 
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=1),  # 減小旋轉範圍
    #A.RandomBrightnessContrast(p=0.1), # 吊掛都都是黃色
    A.HueSaturationValue(p=0.5),
    A.RandomCrop(width=640, height=640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    
    return transform

# 定義類別ID，假設吊掛鉤的類別ID為2
target_class_id = 6

# 加載圖片並進行數據增強
data_dir = r'C:\Users\Gini-AI\Desktop\pile_driver_20241224'
output_dir = r'C:\Users\Gini-AI\Desktop\pile_driver_20241224_augment'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(data_dir):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        img_path = os.path.join(data_dir, file_name)
        label_path = os.path.join(data_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 構建標註列表
            bboxes = []
            class_labels = []
            for line in lines:
                class_id, x_center, y_center, box_width, box_height = map(float, line.split())
                bboxes.append([x_center, y_center, box_width, box_height])
                class_labels.append(class_id)

            # 檢查是否包含目標類別
            has_target_class = any(int(line.split()[0]) == target_class_id for line in lines)

            if has_target_class:
                # 讀取圖像
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"錯誤：無法讀取圖片 {img_path}")
                    continue

                height, width, _ = image.shape

                # 根據圖片尺寸獲取轉換管道
                transform = get_transform(height, width)
                
                # 應用增強
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']

                # 反歸一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                augmented_image = augmented_image.permute(1, 2, 0).numpy()  # 轉換為HWC
                augmented_image = std * augmented_image + mean  # 反歸一化
                augmented_image = np.clip(augmented_image * 255, 0, 255).astype(np.uint8)  # 轉換回0-255範圍

                # 生成新的文件名
                new_file_name = f"{uuid.uuid4().hex}"
                new_img_name = f"{new_file_name}.jpg"
                new_label_name = f"{new_file_name}.txt"

                # 保存增強後的圖像
                cv2.imwrite(os.path.join(output_dir, new_img_name), augmented_image)

                # 保存更新後的標籤文件
                with open(os.path.join(output_dir, new_label_name), 'w') as f:
                    for bbox, class_id in zip(augmented_bboxes, class_labels):
                        x_center, y_center, box_width, box_height = bbox
                        f.write(f"{int(class_id)} {x_center} {y_center} {box_width} {box_height}\n")

print("完成數據增強和文件保存")
