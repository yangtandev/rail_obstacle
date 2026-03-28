# Rail Obstacle Detection System

## Description
This project implements a real-time rail obstacle detection system designed to monitor camera feeds for potential hazards on railway tracks. It leverages YOLOv10 for object detection, optimized with OpenVINO for efficient inference, and integrates with an alerting mechanism and an API for logging intrusions.

## Features
*   **Real-time Object Detection**: Utilizes YOLOv10 for accurate and efficient obstacle detection.
*   **OpenVINO Optimization**: Model inference is optimized with OpenVINO for high performance on Intel hardware.
*   **Multi-Camera Support**: Capable of monitoring multiple camera feeds concurrently.
*   **Danger Zone Monitoring**: Defines and monitors specific polygonal "danger zones" within camera views.
*   **Alerting Mechanism**: Triggers external alerts (e.g., GPIO signals via HTTP requests) when obstacles are detected in danger zones.
*   **Intrusion Logging API**: Integrates with an external API to log detection events and associated images.
*   **Image Saving**: Automatically saves detected frames with bounding boxes, with a configurable limit.
*   **Flexible Input Sources**: Supports both HTTP-based JPG image URLs and RTSP video streams.
*   **Robust Processing**: Employs multi-processing and multi-threading for concurrent frame acquisition and detection, with graceful handling of processing timeouts.

## Project Structure
```
rail_obstacle/
├── main.py          # Main application entry point
├── camera.py                 # Camera module (RTSP stream handler)
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── .gitattributes
├── models/                   # Trained model files
│   ├── rail_obstacle.pt      # Original PyTorch weights
│   └── int8/                 # INT8 quantized OpenVINO model (used at runtime)
│       └── rail_obstacle_openvino_model/
├── mask/                     # Danger zone coordinates per camera
│   ├── {cam_id}.jpg          # Danger zone visualization
│   └── {cam_id}.txt          # Polygon coordinates (x,y per line)
├── image/                    # Camera snapshots (used by tools/ele_test.py)
├── datasets/                 # Training dataset configuration
│   └── rail_obstacle.yaml
├── saved_images/             # Detection result screenshots (runtime output)
└── tools/                    # Data processing & setup utilities
    ├── ele_test.py            # Interactive danger zone drawing tool
    ├── crawl_pic.py           # Training image crawler
    ├── json_to_txt.py         # COCO JSON → YOLO TXT converter
    ├── change_multi_cls.py    # Batch class ID remapping
    ├── delete_and_relabel_classes.py
    ├── delete_blank_pic_N_txt.py
    ├── delete_unmatched_jpgs.py
    ├── generate_white_txt.py
    ├── move_dateset.py
    ├── train_val_shuffle.py
    └── copy_yolov10_int8_dataset.py
```

## Installation

### Prerequisites
*   Python 3.x
*   OpenCV
*   `ultralytics` (for YOLOv10)
*   `openvino`
*   `requests`
*   `shapely`
*   `numpy`
*   Git LFS (for model files)

### Steps
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yangtandev/rail_obstacle.git
    cd rail_obstacle
    ```
2.  **Install Git LFS**:
    Ensure Git LFS is installed on your system. Follow instructions from [git-lfs.com](https://git-lfs.com/).
    Then, initialize Git LFS in your repository and track model files:
    ```bash
    git lfs install
    git lfs track "models/*.pt"
    git lfs track "models/**/*.xml"
    git lfs track "models/**/*.bin"
    git add .gitattributes
    git commit -m "feat: Configure Git LFS for models" # Or add to an existing commit
    ```
3.  **Create a Python virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage & Deployment

### Configuration
*   **Camera IDs and URLs**: Modify the `active_camera_ids` list and `rtsp_links` generation in `main.py` to match your camera setup.
    *   For HTTP JPG sources: `rtsp_links = [f"http://your.ip.address/image/{cam_id}.jpg" for cam_id in active_camera_ids]`
    *   For RTSP streams: `rtsp_links = ["rtsp://your.rtsp.stream/url1", "rtsp://your.rtsp.stream/url2"]`
*   **Danger Zones**: Define polygonal danger zones for each camera in `mask/{cam_id}.txt` files. Each line in the file should contain `x,y` coordinates. You can use `tools/ele_test.py` to interactively draw danger zones on camera snapshots.
*   **Model Path**: The system expects the OpenVINO model to be located at `models/int8/rail_obstacle_openvino_model/`. Ensure your model files (`.xml`, `.bin`) are present there.
*   **Alert API**: The `api` variable in `main.py` (`https://jenyi-xg.api.ginibio.com/api/v1`) is used for intrusion logging. Adjust if necessary.
*   **Alert Device IPs**: The `handle_alert_in_background` function contains logic for triggering external alerts based on camera ID ranges (`192.168.3.181`, `192.168.3.182`). Modify this logic to suit your alert hardware and network configuration.

### Running the Application
To start the detection system:
```bash
source venv/bin/activate
python main.py
```

### Systemd Service (Example)
For production deployment, it's recommended to run the application as a systemd service. An example service file (`rail_obstacle.service`) might look like this:
```ini
[Unit]
Description=Rail Obstacle Detection Service
After=network.target

[Service]
User=gini-facetest
WorkingDirectory=/home/gini-facetest/rail_obstacle
ExecStart=/home/gini-facetest/rail_obstacle/venv/bin/python /home/gini-facetest/rail_obstacle/main.py
Restart=always
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
*(Note: You would need to create this file, place it in `/etc/systemd/system/`, and then enable and start it: `sudo systemctl enable rail_obstacle.service && sudo systemctl start rail_obstacle.service`)*

## Tools

The `tools/` directory contains standalone utility scripts for data preparation and system setup. These are **not** required for running the main detection system.

### Danger Zone Setup
*   **`tools/ele_test.py`** — Interactive tool to draw polygonal danger zones on camera snapshots. Left-click to add points, double-click to confirm, right-click to clear.

### Image Labeling
For manually labeling training images, we recommend using [LabelImg](https://github.com/HumanSignal/labelImg), an open-source graphical image annotation tool that supports YOLO format output.

Install and run via pip:
```bash
pip install labelImg
labelImg
```

### Data Processing
*   `tools/json_to_txt.py` — Convert COCO JSON annotations to YOLO TXT format.
*   `tools/change_multi_cls.py` — Batch remap class IDs in annotation files.
*   `tools/delete_and_relabel_classes.py` — Delete specific classes and renumber remaining ones.
*   `tools/delete_blank_pic_N_txt.py` — Remove empty annotation files and their corresponding images.
*   `tools/delete_unmatched_jpgs.py` — Remove images without matching annotation files.
*   `tools/generate_white_txt.py` — Generate empty annotation files for unlabeled images.
*   `tools/move_dateset.py` — Move paired image/annotation files between directories.
*   `tools/train_val_shuffle.py` — Randomly split dataset into train/validation sets.
*   `tools/copy_yolov10_int8_dataset.py` — Copy and organize dataset into YOLOv10 format structure.
*   `tools/crawl_pic.py` — Crawl training images from the web using Selenium.

## Configuration Notes
*   **Path Adjustments**: Ensure all hardcoded paths in `main.py` (e.g., `models/`, `mask/`, `saved_images/`) are correct relative to the project root or are absolute paths.
*   **OpenCV FFMPEG Warnings**: The `VIDEOIO(FFMPEG)` warnings in the logs often indicate issues with OpenCV's ability to capture video by name or specific backend configurations. Ensure your OpenCV installation has proper FFMPEG support and that camera URLs are correct.

## Contributing
(Optional: Add guidelines for contributing to the project here.)

## License
(Optional: Specify the project's license here, e.g., MIT, Apache 2.0.)
