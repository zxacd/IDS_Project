"""
推理时延测试脚本 - 完全绕过 mixed_float16 加载问题
运行前确保 fix_model_loading.py 在同一目录下
"""

import os
import sys
import numpy as np
import pandas as pd
import time

# 第一步：先应用所有 patch，再导入 tensorflow
print("=" * 60)
print("推理时延测试（修复版）")
print("=" * 60)

print("\n[0] 应用 mixed_float16 加载修复补丁...")
import tensorflow as tf
import h5py
import json

# ===== Patch 1: Operation.from_config =====
try:
    from tensorflow.keras.src.ops.operation import Operation
    _orig_op = Operation.from_config

    @classmethod
    def _patched_op_from_config(cls, config):
        if 'dtype' in config:
            d = config['dtype']
            if isinstance(d, str):
                config['dtype'] = 'float32'
            elif isinstance(d, dict):
                config['dtype'] = 'float32'
        return _orig_op.__func__(cls, config)

    Operation.from_config = _patched_op_from_config
    print("  ✅ Patch Operation.from_config")
except Exception as e:
    print(f"  ⚠️  Patch Operation 失败: {e}")

# ===== Patch 2: 序列化反序列化 =====
try:
    from tensorflow.keras.src.legacy.saving import serialization as _ser

    _orig_deser = _ser.deserialize_keras_object

    def _patched_deser(config, custom_objects=None):
        def _fix(d):
            if isinstance(d, dict):
                if 'dtype' in d:
                    d['dtype'] = 'float32'
                for k in d:
                    d[k] = _fix(d[k])
            elif isinstance(d, list):
                return [_fix(i) for i in d]
            return d
        config = _fix(config)
        return _orig_deser(config, custom_objects)

    _ser.deserialize_keras_object = _patched_deser
    print("  ✅ Patch deserialize_keras_object")
except Exception as e:
    print(f"  ⚠️  Patch deserialize 失败: {e}")

# ===== Patch 3: 全局精度策略 =====
try:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("  ✅ 设置全局精度为 float32")
except Exception as e:
    print(f"  ⚠️  设置精度失败: {e}")


# ==================== 模型加载函数 ====================
def load_model_via_config(model_path):
    """
    通过读取 H5 配置重建模型，彻底绕过 mixed_float16 问题
    """
    print(f"    方式1：通过 config 重建 {model_path} ...")
    with h5py.File(model_path, 'r') as f:
        raw = f.attrs['model_config']
        if isinstance(raw, bytes):
            cfg_str = raw.decode('utf-8')
        else:
            cfg_str = str(raw)
        config = json.loads(cfg_str)

    # 递归修复所有 dtype
    def _fix(d):
        if isinstance(d, dict):
            if 'dtype' in d:
                d['dtype'] = 'float32'
            for k in d:
                d[k] = _fix(d[k])
        elif isinstance(d, list):
            return [_fix(i) for i in d]
        return d

    config = _fix(config)

    model = tf.keras.models.model_from_config(config)
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_model_simple(model_path):
    """
    方法2：直接 load_model（在 patch 后应该能工作）
    """
    print(f"    方式2：直接 tf.keras.models.load_model {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ==================== 1. 加载数据 ====================
print("\n[1] 加载测试数据...")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
num_classes = int(np.max(y_test)) + 1
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
print(f"    测试集: {X_test.shape}, 类别数: {num_classes}")


# ==================== 2. 加载模型 ====================
print("\n[2] 加载模型...")
teacher, student = None, None

# --- 教师模型 ---
t_path = "models/final_model.h5"
if os.path.exists(t_path):
    for loader in [load_model_simple, load_model_via_config]:
        try:
            teacher = loader(t_path)
            print(f"    ✅ 教师模型加载成功")
            break
        except Exception as e:
            print(f"    ⚠️  {loader.__name__} 失败: {e}")
            continue
    if teacher is None:
        print("    ❌ 教师模型所有加载方式均失败")
        sys.exit(1)
    t_params = teacher.count_params()
    t_size = os.path.getsize(t_path) / 1024 / 1024
    print(f"    参数量: {t_params:,}  大小: {t_size:.2f} MB")
else:
    print(f"    ❌ 文件不存在: {t_path}")
    sys.exit(1)

# --- 学生模型 ---
s_path = "models/student_model_optimized.h5"
has_student = False
if os.path.exists(s_path):
    for loader in [load_model_simple, load_model_via_config]:
        try:
            student = loader(s_path)
            print(f"    ✅ 学生模型加载成功")
            has_student = True
            break
        except Exception as e:
            print(f"    ⚠️  {loader.__name__} 失败: {e}")
            continue
    if not has_student:
        print("    ⚠️  学生模型加载失败，跳过")
    else:
        s_params = student.count_params()
        s_size = os.path.getsize(s_path) / 1024 / 1024
        print(f"    参数量: {s_params:,}  大小: {s_size:.2f} MB")
else:
    print(f"    ⚠️  学生模型不存在: {s_path}，跳过")


# ==================== 3. 预热 ====================
print("\n[3] 模型预热...")
_ = teacher.predict(X_test[:100], verbose=0)
if has_student:
    _ = student.predict(X_test[:100], verbose=0)
print("    完成")


# ==================== 4. 推理时延测试 ====================
print("\n[4] 推理时延测试（每批 5 次取平均）...")
print("-" * 60)

batch_sizes = [1, 16, 32, 64, 128, 256]
results = []

for bs in batch_sizes:
    # 教师
    t_times = []
    for _ in range(5):
        st = time.time()
        teacher.predict(X_test[:1000], batch_size=bs, verbose=0)
        t_times.append(time.time() - st)
    t_avg = float(np.mean(t_times))
    t_std = float(np.std(t_times))
    t_fps = 1000 / t_avg

    if has_student:
        s_times = []
        for _ in range(5):
            st = time.time()
            student.predict(X_test[:1000], batch_size=bs, verbose=0)
            s_times.append(time.time() - st)
        s_avg = float(np.mean(s_times))
        s_std = float(np.std(s_times))
        s_fps = 1000 / s_avg
        speedup = t_avg / s_avg

        print(f"  Batch {bs:>3}: "
              f"教师 {t_avg:.4f}±{t_std:.4f}s | "
              f"学生 {s_avg:.4f}±{s_std:.4f}s | "
              f"加速 {speedup:.2f}x")
    else:
        print(f"  Batch {bs:>3}: 教师 {t_avg:.4f}±{t_std:.4f}s | {t_fps:.1f} FPS")

    row = {'batch': bs, 't_time': t_avg, 't_std': t_std, 't_fps': t_fps}
    if has_student:
        row.update({'s_time': s_avg, 's_std': s_std, 's_fps': s_fps, 'speedup': speedup})
    results.append(row)

print("-" * 60)


# ==================== 5. 准确率 ====================
print("\n[5] 准确率测试...")
t_acc = float(teacher.evaluate(X_test, y_test_cat, verbose=0)[1])
print(f"    教师模型: {t_acc:.4f} ({t_acc*100:.2f}%)")
if has_student:
    s_acc = float(student.evaluate(X_test, y_test_cat, verbose=0)[1])
    print(f"    学生模型: {s_acc:.4f} ({s_acc*100:.2f}%)")
    print(f"    差距: {(t_acc - s_acc)*100:.2f}%")


# ==================== 6. 保存结果 ====================
print("\n[6] 保存结果...")
os.makedirs("results", exist_ok=True)

# inference_speed_results.csv
rows = []
for r in results:
    row = {'Batch_Size': r['batch'],
            'Teacher_Time (s)': round(r['t_time'], 4),
            'Teacher_Std (s)': round(r['t_std'], 4),
            'Teacher_FPS': round(r['t_fps'], 1)}
    if has_student:
        row.update({
            'Student_Time (s)': round(r['s_time'], 4),
            'Student_Std (s)': round(r['s_std'], 4),
            'Student_FPS': round(r['s_fps'], 1),
            'Speedup': round(r['speedup'], 2)
        })
    rows.append(row)
pd.DataFrame(rows).to_csv("results/inference_speed_results.csv", index=False, encoding="utf-8-sig")
print("    ✅ results/inference_speed_results.csv")

# model_comparison.csv
bs32 = next(r for r in results if r['batch'] == 32)
t_params = teacher.count_params()
t_size = os.path.getsize("models/final_model.h5") / 1024 / 1024

if has_student:
    s_params = student.count_params()
    s_size = os.path.getsize(s_path) / 1024 / 1024
    cmp_data = {
        'Metric': ['参数量', '模型大小(MB)', '准确率', f'推理时间@batch=32(s)', '吞吐量(FPS)'],
        'Teacher': [f"{t_params:,}", f"{t_size:.2f}", f"{t_acc:.4f}",
                    f"{bs32['t_time']:.4f}", f"{bs32['t_fps']:.1f}"],
        'Student': [f"{s_params:,}", f"{s_size:.2f}", f"{s_acc:.4f}",
                    f"{bs32['s_time']:.4f}", f"{bs32['s_fps']:.1f}"],
        'Reduction': [
            f"{(1 - s_params/t_params)*100:.1f}%",
            f"{(1 - s_size/t_size)*100:.1f}%",
            f"{(t_acc - s_acc)*100:.2f}%",
            f"{bs32['speedup']:.2f}x",
            f"{bs32['speedup']:.2f}x"
        ]
    }
else:
    cmp_data = {
        'Metric': ['参数量', '模型大小(MB)', '准确率', f'推理时间@batch=32(s)', '吞吐量(FPS)'],
        'Teacher': [f"{t_params:,}", f"{t_size:.2f}", f"{t_acc:.4f}",
                    f"{bs32['t_time']:.4f}", f"{bs32['t_fps']:.1f}"]
    }

pd.DataFrame(cmp_data).to_csv("results/model_comparison.csv", index=False, encoding="utf-8-sig")
print("    ✅ results/model_comparison.csv")


# ==================== 7. 打印汇总 ====================
print("\n" + "=" * 60)
print("汇总结果（batch=32）")
print("=" * 60)
print(f"  教师推理时间: {bs32['t_time']:.4f}s / 1000样本")
print(f"  教师单样本:  {bs32['t_time']/1000*1000:.3f}ms")
print(f"  教师吞吐量:   {bs32['t_fps']:.1f} FPS")
if has_student:
    print(f"  学生推理时间: {bs32['s_time']:.4f}s / 1000样本")
    print(f"  学生单样本:  {bs32['s_time']/1000*1000:.3f}ms")
    print(f"  学生吞吐量:   {bs32['s_fps']:.1f} FPS")
    print(f"  加速比:       {bs32['speedup']:.2f}x")
print("=" * 60)
print("✅ 全部完成！")
