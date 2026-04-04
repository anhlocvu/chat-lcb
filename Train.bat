@echo off
title Chat LCB Data Trainer
echo ==========================================
echo       STARTING AI TRAINING PROCESS
echo ==========================================
echo.
echo [1/3] Checking for existing database...

:: Sửa lỗi dấu ngoặc bằng cách xóa bỏ chúng trong lệnh echo
if exist chroma_db (
    echo [INFO] Deleting old database chroma_db...
    rmdir /s /q "chroma_db"
) else (
    echo [INFO] No old database found.
)

echo.
echo [2/3] Preparing training data from /data...
uv run python src/ingest.py

if %errorlevel% equ 0 (
    echo.
    echo [3/3] Training completed successfully!
    echo [SUCCESS] Your AI now has new knowledge.
) else (
    echo.
    echo [ERROR] Training failed! Please check your data files.
)

pause
