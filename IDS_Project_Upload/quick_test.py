#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试脚本 - 验证 Python 环境"""
import sys
import os

print("=" * 60)
print("Python 环境测试")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"当前目录: {os.getcwd()}")
print()

# 测试 TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow 版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   GPU 数量: {len(gpus)}")
except ImportError as e:
    print(f"❌ TensorFlow 导入失败: {e}")
except Exception as e:
    print(f"❌ TensorFlow 错误: {e}")

# 测试 NumPy
try:
    import numpy as np
    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")

# 测试 Pandas
try:
    import pandas as pd
    print(f"✅ Pandas 版本: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas 导入失败: {e}")

# 测试数据文件
print()
print("检查数据文件...")
data_files = [
    "data/processed/X_train.npy",
    "data/processed/X_val.npy",
    "data/processed/X_test.npy"
]
for f in data_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"✅ {f}: {size_mb:.1f} MB")
    else:
        print(f"❌ {f}: 不存在")

# 测试模型文件
print()
print("检查模型文件...")
model_files = [
    "models/final_model.h5",
    "models/student_model_optimized.h5"
]
for f in model_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"✅ {f}: {size_mb:.1f} MB")
    else:
        print(f"⚠️  {f}: 不存在")

print()
print("=" * 60)
print("测试完成")
print("=" * 60)

input("按 Enter 键退出...")
