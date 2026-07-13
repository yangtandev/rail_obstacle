# Release Notes

## v1.1.3 - 2026-07-13

- 新增攝影機推論 FPS 上限設定，預設每路 8 FPS，降低多路 RTSP 推論造成的 CPU load。

## v1.1.2 - 2026-07-13

- 補上 `huggingface-hub` Python 依賴，修正全新安裝後缺少 `huggingface_hub` 模組的問題。

## v1.1.1 - 2026-07-13

- 修正部分 HEVC RTSP 串流在 OpenCV 解碼失敗，導致長時間無法取得畫面的問題。
- RTSP 擷取改用 ffmpeg raw frame，並保留 TCP/UDP 重連 fallback。
- 新增 RTSP 診斷工具，方便檢查各攝影機連線與取幀狀態。

## v1.1.0 - 2026-07-13

- 新增預設攝影機設定，包含 1921683113、1921683115、1921683118。
- 移除 README 末尾尚未填寫的 License placeholder。

## v1.0.0 - 2026-07-13

- 將部署服務改為 systemd user service，改用 `systemctl --user` 管理。
- 啟用 user lingering，讓服務可在開機後自動啟動。
- 更新 README 與架構文件中的服務管理指令。
