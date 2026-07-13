# Rail Obstacle Detection System

## Description
A real-time rail obstacle detection system that monitors camera feeds for hazards on railway tracks. It uses YOLOv10 for object detection, optimized with OpenVINO for efficient inference on Intel hardware, and integrates with an alerting mechanism and an API for logging intrusion events.

## Features
*   **Real-time Object Detection**: YOLOv10 for accurate obstacle detection.
*   **OpenVINO Optimization**: High-performance inference on Intel hardware.
*   **Multi-Camera Support**: Concurrent monitoring of multiple camera feeds.
*   **Danger Zone Monitoring**: Polygonal danger zone definitions per camera.
*   **Alerting Mechanism**: Triggers external alerts (GPIO via HTTP) on detection.
*   **Intrusion Logging**: Logs detection events and images to an external API.
*   **Flexible Input Sources**: Supports HTTP JPG image URLs and RTSP streams.
*   **Robust Processing**: Multi-processing / multi-threading with graceful timeout handling.

## Project Structure
```
rail_obstacle/
├── main.py                   # Main application entry point
├── camera.py                 # Camera module (RTSP stream handler)
├── install.sh                # One-click deployment script
├── requirements.txt          # Python dependencies
├── config.json               # Runtime configuration
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

## Deployment

The entire setup process is automated via `install.sh`, which handles:
- System dependencies
- Python virtual environment (via `uv`)
- Git LFS model retrieval
- `config.json` configuration
- systemd user service registration and startup

```bash
git clone https://github.com/yangtandev/rail_obstacle.git
cd rail_obstacle
sudo bash install.sh
```

The service will be running as `rail_obstacle.service` upon completion.
It is managed by the installing user via `systemctl --user`.
The installer enables user lingering so the service can start at boot.

## Configuration

The following items require manual adjustment before or after running `install.sh`:

*   **Camera IDs and URLs**: Edit the `active_camera_ids` list and `rtsp_links` in `main.py` to match your camera setup.
    *   HTTP JPG: `rtsp_links = [f"http://your.ip/{cam_id}.jpg" for cam_id in active_camera_ids]`
    *   RTSP: `rtsp_links = ["rtsp://your.stream/url1", ...]`
*   **Danger Zones**: Define polygonal zones per camera in `mask/{cam_id}.txt` (one `x,y` coordinate per line). Use `tools/ele_test.py` to draw zones interactively.
*   **Alert API**: The `api` variable in `main.py` points to the intrusion logging endpoint. Update it if your backend URL changes.
*   **Alert Device IPs**: `handle_alert_in_background()` in `main.py` maps camera ID ranges to external alert device IPs (`192.168.3.181`, `192.168.3.182`). Adjust to match your hardware.
*   **Model Path**: OpenVINO model (`.xml`, `.bin`) must be at `models/int8/rail_obstacle_openvino_model/`.

After changing configuration, restart the service:
```bash
systemctl --user restart rail_obstacle.service
```

## Running Manually

If you need to run the application outside of systemd:
```bash
source venv/bin/activate
python main.py
```

## Tools

The `tools/` directory contains standalone utilities for data preparation. These are **not** required for the main detection system.

### Danger Zone Setup
*   **`tools/ele_test.py`** — Interactively draw polygonal danger zones on camera snapshots. Left-click to add points, double-click to confirm, right-click to clear.

### Image Labeling
We recommend [LabelImg](https://github.com/HumanSignal/labelImg) for annotating training images (YOLO format output):
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

## License
(Optional: Specify the project's license here, e.g., MIT, Apache 2.0.)
