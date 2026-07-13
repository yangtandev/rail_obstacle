# Release Notes

## v1.0.0 - 2026-07-13

- 將部署服務改為 systemd user service，改用 `systemctl --user` 管理。
- 啟用 user lingering，讓服務可在開機後自動啟動。
- 更新 README 與架構文件中的服務管理指令。
