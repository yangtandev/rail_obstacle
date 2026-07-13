# Release Notes

## v1.1.0 - 2026-07-13

- 新增預設攝影機設定，包含 1921683113、1921683115、1921683118。
- 移除 README 末尾尚未填寫的 License placeholder。

## v1.0.0 - 2026-07-13

- 將部署服務改為 systemd user service，改用 `systemctl --user` 管理。
- 啟用 user lingering，讓服務可在開機後自動啟動。
- 更新 README 與架構文件中的服務管理指令。
