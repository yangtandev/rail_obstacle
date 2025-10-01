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

## Installation

### Prerequisites
*   Python 3.x
*   OpenCV
*   `ultralytics` (for YOLOv10)
*   `openvino`
*   `requests`
*   `shapely`
*   `numpy`
*   `screeninfo`
*   `Pillow`
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
    *(Note: Ensure `requirements.txt` contains all necessary packages. If not, you may need to add them manually or generate a new one.)*

## Usage & Deployment

### Configuration
*   **Camera IDs and URLs**: Modify the `active_camera_ids` list and `rtsp_links` generation in `rail_obstacle.py` to match your camera setup.
    *   For HTTP JPG sources: `rtsp_links = [f"http://your.ip.address/image/{cam_id}.jpg" for cam_id in active_camera_ids]`
    *   For RTSP streams: `rtsp_links = ["rtsp://your.rtsp.stream/url1", "rtsp://your.rtsp.stream/url2"]`
*   **Danger Zones**: Define polygonal danger zones for each camera in `mask/{cam_id}.txt` files. Each line in the file should contain `x,y` coordinates.
*   **Model Path**: The system expects the OpenVINO model to be located at `models/int8/rail_obstacle_openvino_model/`. Ensure your model files (`.xml`, `.bin`) are present there.
*   **Alert API**: The `api` variable in `rail_obstacle.py` (`https://jenyi-xg.api.ginibio.com/api/v1`) is used for intrusion logging. Adjust if necessary.
*   **Alert Device IPs**: The `process_detection_task` function contains logic for triggering external alerts based on camera ID ranges (`192.168.3.181`, `192.168.3.182`). Modify this logic to suit your alert hardware and network configuration.

### Running the Application
To start the detection system:
```bash
source venv/bin/activate
python rail_obstacle.py
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
ExecStart=/home/gini-facetest/rail_obstacle/venv/bin/python /home/gini-facetest/rail_obstacle/rail_obstacle.py
Restart=always
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
*(Note: You would need to create this file, place it in `/etc/systemd/system/`, and then enable and start it: `sudo systemctl enable rail_obstacle.service && sudo systemctl start rail_obstacle.service`)*

## Configuration Notes
*   **Path Adjustments**: Ensure all hardcoded paths in `rail_obstacle.py` (e.g., `models/`, `mask/`, `saved_images/`) are correct relative to the project root or are absolute paths.
*   **Performance**: The `timeout=0.01` in `rail_obstacle.py`'s `main_loop` is set for responsiveness. If you experience high CPU usage or missed frames, you might need to adjust this value or optimize your model further.
*   **OpenCV FFMPEG Warnings**: The `VIDEOIO(FFMPEG)` warnings in the logs often indicate issues with OpenCV's ability to capture video by name or specific backend configurations. Ensure your OpenCV installation has proper FFMPEG support and that camera URLs are correct.

## Contributing
(Optional: Add guidelines for contributing to the project here.)

## License
(Optional: Specify the project's license here, e.g., MIT, Apache 2.0.)
