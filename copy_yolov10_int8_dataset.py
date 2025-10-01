import os
import shutil

# 源資料夾的絕對路徑
source_path = r"/home/gini-jyxg/annotations/dataset"
# 目標資料夾的絕對路徑
target_path = r"/home/gini-jyxg/recognition/datasets/rail_obstacle"

# 創建目標資料夾結構
images_train_path = os.path.join(target_path, 'images', 'train')
images_val_path = os.path.join(target_path, 'images', 'val')
labels_train_path = os.path.join(target_path, 'labels', 'train')
labels_val_path = os.path.join(target_path, 'labels', 'val')

os.makedirs(images_train_path, exist_ok=True)
os.makedirs(images_val_path, exist_ok=True)
os.makedirs(labels_train_path, exist_ok=True)
os.makedirs(labels_val_path, exist_ok=True)

# 生成 train.txt 和 val.txt 路徑
train_txt_path = os.path.join(target_path, 'train.txt')
val_txt_path = os.path.join(target_path, 'val.txt')

# 複製圖片和標註檔，並在 txt 文件中寫入絕對路徑
def copy_files_and_generate_txt(subfolder, img_target_folder, lbl_target_folder, txt_file):
    image_source_folder = os.path.join(source_path, subfolder, 'images')
    label_source_folder = os.path.join(source_path, subfolder, 'labels')

    with open(txt_file, 'w') as txt:
        for img_file in os.listdir(image_source_folder):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):  # 根據圖片格式調整
                # 複製圖片
                img_source_path = os.path.join(image_source_folder, img_file)
                img_target_path = os.path.join(img_target_folder, img_file)
                shutil.copy(img_source_path, img_target_path)

                # 複製標註檔
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                label_source_path = os.path.join(label_source_folder, label_file)
                label_target_path = os.path.join(lbl_target_folder, label_file)
                if os.path.exists(label_source_path):
                    shutil.copy(label_source_path, label_target_path)

                # 將圖片的絕對路徑寫入 txt
                absolute_img_path = os.path.abspath(img_target_path)
                txt.write(absolute_img_path + '\n')

# 將 train 資料夾內容複製到目標資料夾
copy_files_and_generate_txt('train', images_train_path, labels_train_path, train_txt_path)

# 將 valid 資料夾內容複製到目標資料夾
copy_files_and_generate_txt('valid', images_val_path, labels_val_path, val_txt_path)

print("資料複製完成，train.txt 和 val.txt 已生成（包含絕對路徑）。")
