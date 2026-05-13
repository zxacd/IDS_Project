@echo off
chcp 65001 > nul
echo ====================================
echo IDS_Project 环境测试
echo ====================================
echo.

echo 正在测试 Python 环境...
"D:\Users\zxacd\Miniconda3\python.exe" "D:\Users\zxacd\IDS_Project\quick_test.py"

echo.
echo ====================================
echo 如果看到上面的测试信息，说明环境正常
echo 可以按任意键关闭窗口
echo ====================================
pause > nul
