@echo off
chcp 65001 > nul
echo ====================================
echo IDS_Project 运行测试
echo ====================================
echo.

echo 1. 测试 Python 环境...
"D:\Users\zxacd\Miniconda3\python.exe" -c "import sys; print('Python 版本:', sys.version); print('Python 路径:', sys.executable)"

echo.
echo 2. 运行测试脚本...
"D:\Users\zxacd\Miniconda3\python.exe" test_run.py

echo.
echo ====================================
echo 测试完成
echo ====================================
pause
