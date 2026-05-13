"""
实验 3：模型压缩效果分析
对比教师模型和学生模型的：
- 参数量
- 模型大小
- 推理速度
- 内存占用
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("实验 3：模型压缩效果分析")
print("=" * 60)

# 加载测试数据
print("\n[1] 加载数据和模型...")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
num_classes = int(np.max(y_test)) + 1
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# 加载教师模型
teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 加载学生模型
student_path = "models/student_model_optimized.h5"
if os.path.exists(student_path):
    student = tf.keras.models.load_model(student_path, compile=False)
    student.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    has_student = True
    print("✅ 学生模型已加载")
else:
    print("⚠️  学生模型不存在，跳过学生模型评估")
    has_student = False

# ==================== 1. 参数量对比 ====================
print("\n[2] 参数量对比...")

teacher_params = teacher.count_params()
if has_student:
    student_params = student.count_params()
    param_reduction = (1 - student_params / teacher_params) * 100
else:
    student_params = 0
    param_reduction = 0

print(f"教师模型参数量: {teacher_params:,}")
if has_student:
    print(f"学生模型参数量: {student_params:,}")
    print(f"参数量减少: {param_reduction:.1f}%")

# ==================== 2. 模型大小对比 ====================
print("\n[3] 模型大小对比...")

teacher_size = os.path.getsize("models/final_model.h5") / (1024 * 1024)
if has_student:
    student_size = os.path.getsize(student_path) / (1024 * 1024)
    size_reduction = (1 - student_size / teacher_size) * 100
else:
    student_size = 0
    size_reduction = 0

print(f"教师模型大小: {teacher_size:.2f} MB")
if has_student:
    print(f"学生模型大小: {student_size:.2f} MB")
    print(f"大小减少: {size_reduction:.1f}%")

# ==================== 3. 推理速度对比 ====================
print("\n[4] 推理速度测试...")

# 预热
_ = teacher.predict(X_test[:100], verbose=0)
if has_student:
    _ = student.predict(X_test[:100], verbose=0)

# 测试不同 batch size 下的推理速度
batch_sizes = [1, 16, 32, 64, 128, 256]
speed_results = []

for bs in batch_sizes:
    # 教师模型
    n_runs = 5
    times_teacher = []
    for _ in range(n_runs):
        start = time.time()
        _ = teacher.predict(X_test[:1000], batch_size=bs, verbose=0)
        times_teacher.append(time.time() - start)
    avg_time_teacher = np.mean(times_teacher)
    fps_teacher = 1000 / avg_time_teacher
    
    if has_student:
        # 学生模型
        times_student = []
        for _ in range(n_runs):
            start = time.time()
            _ = student.predict(X_test[:1000], batch_size=bs, verbose=0)
            times_student.append(time.time() - start)
        avg_time_student = np.mean(times_student)
        fps_student = 1000 / avg_time_student
        speedup = avg_time_teacher / avg_time_student
    else:
        avg_time_student = 0
        fps_student = 0
        speedup = 1.0
    
    speed_results.append({
        'Batch_Size': bs,
        'Teacher_Time (s)': avg_time_teacher,
        'Student_Time (s)': avg_time_student,
        'Teacher_FPS': fps_teacher,
        'Student_FPS': fps_student,
        'Speedup': speedup if has_student else 1.0
    })
    
    print(f"  Batch Size {bs}: Teacher {avg_time_teacher:.3f}s, Student {avg_time_student:.3f}s, 加速比 {speedup:.2f}x")

df_speed = pd.DataFrame(speed_results)
df_speed.to_csv("results/inference_speed_results.csv", index=False, encoding="utf-8-sig")
print(f"✅ 推理速度结果已保存: results/inference_speed_results.csv")

# ==================== 4. 准确率对比 ====================
print("\n[5] 准确率对比...")

teacher_acc = teacher.evaluate(X_test, y_test_cat, verbose=0)[1]
if has_student:
    student_acc = student.evaluate(X_test, y_test_cat, verbose=0)[1]
    acc_diff = teacher_acc - student_acc
else:
    student_acc = 0
    acc_diff = 0

print(f"教师模型准确率: {teacher_acc:.4f}")
if has_student:
    print(f"学生模型准确率: {student_acc:.4f}")
    print(f"准确率差距: {acc_diff:.4f} ({acc_diff*100:.2f}%)")

# ==================== 5. 生成对比表格 ====================
print("\n[6] 生成对比表格...")

comparison_data = {
    'Metric': ['参数量', '模型大小 (MB)', '准确率', '推理时间 (batch=32)', '吞吐量 (FPS)'],
    'Teacher': [
        f"{teacher_params:,}",
        f"{teacher_size:.2f}",
        f"{teacher_acc:.4f}",
        f"{df_speed[df_speed['Batch_Size']==32]['Teacher_Time (s)'].values[0]:.3f}s",
        f"{df_speed[df_speed['Batch_Size']==32]['Teacher_FPS'].values[0]:.1f}"
    ]
}

if has_student:
    comparison_data['Student'] = [
        f"{student_params:,}",
        f"{student_size:.2f}",
        f"{student_acc:.4f}",
        f"{df_speed[df_speed['Batch_Size']==32]['Student_Time (s)'].values[0]:.3f}s",
        f"{df_speed[df_speed['Batch_Size']==32]['Student_FPS'].values[0]:.1f}"
    ]
    comparison_data['Reduction'] = [
        f"{param_reduction:.1f}%",
        f"{size_reduction:.1f}%",
        f"{acc_diff*100:.2f}% (差距)",
        f"{df_speed[df_speed['Batch_Size']==32]['Speedup'].values[0]:.2f}x 加速",
        f"{df_speed[df_speed['Batch_Size']==32]['Speedup'].values[0]:.2f}x 加速"
    ]

df_comparison = pd.DataFrame(comparison_data)
print("\n" + "=" * 60)
print("模型对比汇总")
print("=" * 60)
print(df_comparison.to_string(index=False))

df_comparison.to_csv("results/model_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 模型对比结果已保存: results/model_comparison.csv")

# ==================== 6. 可视化 ====================
print("\n[7] 生成可视化图表...")

# 图1：参数量和模型大小对比
if has_student:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 参数量对比
    axes[0].bar(['Teacher', 'Student'], [teacher_params/1e6, student_params/1e6], 
                 color=['#1f77b4', '#2ca02c'])
    axes[0].set_ylabel('Parameters (Million)')
    axes[0].set_title('Model Parameters Comparison')
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate([teacher_params/1e6, student_params/1e6]):
        axes[0].text(i, v + 0.5, f'{v:.2f}M', ha='center')
    
    # 模型大小对比
    axes[1].bar(['Teacher', 'Student'], [teacher_size, student_size], 
                 color=['#1f77b4', '#2ca02c'])
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('Model Size Comparison')
    axes[1].grid(True, alpha=0.3)
    for i, v in enumerate([teacher_size, student_size]):
        axes[1].text(i, v + 0.2, f'{v:.2f}MB', ha='center')
    
    plt.tight_layout()
    plt.savefig("results/model_size_comparison.png", dpi=300)
    plt.close()
    print("✅ 模型大小对比图已保存: results/model_size_comparison.png")

# 图2：推理速度对比
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(batch_sizes))
width = 0.35

if has_student:
    teacher_times = df_speed['Teacher_Time (s)'].values
    student_times = df_speed['Student_Time (s)'].values
    
    ax.bar(x - width/2, teacher_times, width, label='Teacher', color='#1f77b4')
    ax.bar(x + width/2, student_times, width, label='Student', color='#2ca02c')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Inference Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/inference_time_comparison.png", dpi=300)
    plt.close()
    print("✅ 推理速度对比图已保存: results/inference_time_comparison.png")

print("\n" + "=" * 60)
print("实验 3 完成！")
print("=" * 60)
print("\n生成的结果文件:")
print("  - results/model_comparison.csv")
print("  - results/inference_speed_results.csv")
print("  - results/model_size_comparison.png")
print("  - results/inference_time_comparison.png")
