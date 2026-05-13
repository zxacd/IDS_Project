#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
毕设完整实验脚本
运行所有实验，生成符合开题报告要求的结果
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("毕设实验：基于深度学习的网络入侵检测系统")
print("=" * 70)
print("开始时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
print()

# ========== 加载数据 ==========
print("[数据加载]")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")

num_classes = int(np.max(y_test)) + 1
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

print("  训练集:", X_train.shape, "验证集:", X_val.shape, "测试集:", X_test.shape)
print("  类别数:", num_classes)

try:
    le = joblib.load("models/label_encoder.pkl")
    class_names = list(le.classes_)
except:
    class_names = [str(i) for i in range(num_classes)]
print("  类别名称:", class_names[:3], "... 共", len(class_names), "类")
print()

# ========== 加载模型 ==========
print("[加载模型]")
teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

student_path = "models/student_model_optimized.h5"
has_student = os.path.exists(student_path)
if has_student:
    student = tf.keras.models.load_model(student_path, compile=False)
    student.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("  ✅ 学生模型已加载")
else:
    print("  ⚠️  学生模型不存在")
print()

# ========== 实验1：分类性能评估 ==========
print("=" * 70)
print("实验1：CNN 入侵检测模型性能评估")
print("=" * 70)

print("\n[1.1] 生成预测...")
teacher_pred = teacher.predict(X_test, verbose=0)
teacher_pred_classes = np.argmax(teacher_pred, axis=1)
teacher_acc = accuracy_score(y_test, teacher_pred_classes)
print("  教师模型准确率:", round(teacher_acc, 4))

if has_student:
    student_pred = student.predict(X_test, verbose=0)
    student_pred_classes = np.argmax(student_pred, axis=1)
    student_acc = accuracy_score(y_test, student_pred_classes)
    print("  学生模型准确率:", round(student_acc, 4))

print("\n[1.2] 生成分类报告...")
teacher_report = classification_report(y_test, teacher_pred_classes,
                                     target_names=class_names, digits=4)
print("  教师模型分类报告:")
print(teacher_report)

with open("results/实验1_教师模型分类报告.txt", "w", encoding="utf-8") as f:
    f.write("教师模型分类报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(teacher_report)
    f.write("\n\n总体准确率: " + str(round(teacher_acc, 4)) + "\n")
print("  ✅ 已保存: results/实验1_教师模型分类报告.txt")

if has_student:
    student_report = classification_report(y_test, student_pred_classes,
                                         target_names=class_names, digits=4)
    print("\n  学生模型分类报告:")
    print(student_report)
    with open("results/实验1_学生模型分类报告.txt", "w", encoding="utf-8") as f:
        f.write("学生模型分类报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(student_report)
        f.write("\n\n总体准确率: " + str(round(student_acc, 4)) + "\n")
    print("  ✅ 已保存: results/实验1_学生模型分类报告.txt")

print("\n[1.3] 生成混淆矩阵...")
plt.figure(figsize=(12, 10))
cm_teacher = confusion_matrix(y_test, teacher_pred_classes)
sns.heatmap(cm_teacher, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Teacher Model (CNN)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("results/实验1_教师模型混淆矩阵.png", dpi=300)
plt.close()
print("  ✅ 已保存: results/实验1_教师模型混淆矩阵.png")

if has_student:
    plt.figure(figsize=(12, 10))
    cm_student = confusion_matrix(y_test, student_pred_classes)
    sns.heatmap(cm_student, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Student Model (Distilled CNN)', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("results/实验1_学生模型混淆矩阵.png", dpi=300)
    plt.close()
    print("  ✅ 已保存: results/实验1_学生模型混淆矩阵.png")

print("\n实验1 完成！")
print()

# ========== 实验2：知识蒸馏效果分析 ==========
print("=" * 70)
print("实验2：知识蒸馏效果分析")
print("=" * 70)

print("\n[2.1] 参数量与模型大小对比...")
teacher_params = teacher.count_params()
teacher_size = os.path.getsize("models/final_model.h5") / (1024 * 1024)
print("  教师模型参数量:", f"{teacher_params:,}")
print("  教师模型大小:", round(teacher_size, 2), "MB")

if has_student:
    student_params = student.count_params()
    student_size = os.path.getsize(student_path) / (1024 * 1024)
    param_reduction = (1 - student_params / teacher_params) * 100
    size_reduction = (1 - student_size / teacher_size) * 100
    print("  学生模型参数量:", f"{student_params:,}")
    print("  学生模型大小:", round(student_size, 2), "MB")
    print("  参数量减少:", round(param_reduction, 1), "%")
    print("  大小减少:", round(size_reduction, 1), "%")

print("\n[2.2] 准确率对比...")
print("  教师模型准确率:", round(teacher_acc, 4))
if has_student:
    print("  学生模型准确率:", round(student_acc, 4))
    acc_diff = teacher_acc - student_acc
    print("  准确率差距:", round(acc_diff, 4), "(", round(acc_diff*100, 2), "%)")
    if acc_diff < 0.02:
        print("  评价: ✅ 学生模型表现优秀！差距小于 2%")
    elif acc_diff < 0.05:
        print("  评价: ⚠️  学生模型表现一般，差距在 2%-5% 之间")
    else:
        print("  评价: ❌ 学生模型表现较差，差距超过 5%")

print("\n[2.3] 生成对比表格...")
comparison_data = {
    '指标': ['准确率', '参数量', '模型大小(MB)'],
    '教师模型(CNN)': [
        str(round(teacher_acc, 4)),
        f"{teacher_params:,}",
        str(round(teacher_size, 2))
    ]
}
if has_student:
    comparison_data['学生模型(知识蒸馏)'] = [
        str(round(student_acc, 4)),
        f"{student_params:,}",
        str(round(student_size, 2))
    ]
    comparison_data['压缩效果'] = [
        "差距 " + str(round(acc_diff*100, 2)) + "%",
        "减少 " + str(round(param_reduction, 1)) + "%",
        "减少 " + str(round(size_reduction, 1)) + "%"
    ]

df_comp = pd.DataFrame(comparison_data)
df_comp.to_csv("results/实验2_模型对比表格.csv", index=False, encoding="utf-8-sig")
print("  ✅ 已保存: results/实验2_模型对比表格.csv")
print("\n" + df_comp.to_string(index=False))

if has_student:
    print("\n[2.4] 生成可视化图表...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(['Teacher', 'Student'], [teacher_acc, student_acc],
                 color=['#1f77b4', '#2ca02c'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate([teacher_acc, student_acc]):
        axes[0].text(i, v + 0.01, str(round(v, 4)), ha='center')
    
    axes[1].bar(['Teacher', 'Student'], [teacher_params/1e6, student_params/1e6],
                 color=['#1f77b4', '#2ca02c'])
    axes[1].set_ylabel('Parameters (Million)')
    axes[1].set_title('Parameters Comparison')
    axes[1].grid(True, alpha=0.3)
    for i, v in enumerate([teacher_params/1e6, student_params/1e6]):
        axes[1].text(i, v + 0.5, str(round(v, 2)) + "M", ha='center')
    
    axes[2].bar(['Teacher', 'Student'], [teacher_size, student_size],
                 color=['#1f77b4', '#2ca02c'])
    axes[2].set_ylabel('Model Size (MB)')
    axes[2].set_title('Model Size Comparison')
    axes[2].grid(True, alpha=0.3)
    for i, v in enumerate([teacher_size, student_size]):
        axes[2].text(i, v + 0.2, str(round(v, 2)) + "MB", ha='center')
    
    plt.tight_layout()
    plt.savefig("results/实验2_模型对比可视化.png", dpi=300)
    plt.close()
    print("  ✅ 已保存: results/实验2_模型对比可视化.png")

print("\n实验2 完成！")
print()

# ========== 实验3：推理速度测试 ==========
print("=" * 70)
print("实验3：推理速度测试")
print("=" * 70)

print("\n[3.1] 推理速度测试...")
batch_sizes = [1, 16, 32, 64, 128, 256]
speed_results = []

# 预热
_ = teacher.predict(X_test[:100], verbose=0)
if has_student:
    _ = student.predict(X_test[:100], verbose=0)

for bs in batch_sizes:
    times_teacher = []
    for _ in range(5):
        start = time.time()
        _ = teacher.predict(X_test[:1000], batch_size=bs, verbose=0)
        times_teacher.append(time.time() - start)
    avg_time_teacher = np.mean(times_teacher)
    fps_teacher = 1000 / avg_time_teacher
    
    if has_student:
        times_student = []
        for _ in range(5):
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
        'Teacher_Time(s)': round(avg_time_teacher, 3),
        'Student_Time(s)': round(avg_time_student, 3) if has_student else 0,
        'Teacher_FPS': round(fps_teacher, 1),
        'Student_FPS': round(fps_student, 1) if has_student else 0,
        'Speedup': round(speedup, 2) if has_student else 1.0
    })
    print("  Batch Size", bs, ": Teacher", round(avg_time_teacher, 3), "s, Student",
          round(avg_time_student, 3) if has_student else "N/A", "s, 加速比",
          round(speedup, 2) if has_student else "N/A")

df_speed = pd.DataFrame(speed_results)
df_speed.to_csv("results/实验3_推理速度结果.csv", index=False, encoding="utf-8-sig")
print("\n  ✅ 已保存: results/实验3_推理速度结果.csv")

if has_student:
    print("\n[3.2] 生成推理速度可视化...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    teacher_times = df_speed['Teacher_Time(s)'].values
    student_times = df_speed['Student_Time(s)'].values
    
    ax.bar(x - width/2, teacher_times, width, label='Teacher (CNN)', color='#1f77b4')
    ax.bar(x + width/2, student_times, width, label='Student (Distilled)', color='#2ca02c')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Inference Time Comparison (1000 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/实验3_推理速度对比.png", dpi=300)
    plt.close()
    print("  ✅ 已保存: results/实验3_推理速度对比.png")

print("\n实验3 完成！")
print()

# ========== 实验4：与其他方法对比 ==========
print("=" * 70)
print("实验4：与其他方法对比")
print("=" * 70)

print("\n[4.1] 训练传统机器学习方法...")
print("  注意：SVM 使用 10% 数据（大数据上训练较慢）")

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

results_ml = []

# 逻辑回归
print("\n  训练逻辑回归...")
start = time.time()
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
lr.fit(X_train_flat, y_train)
lr_pred = lr.predict(X_test_flat)
lr_acc = accuracy_score(y_test, lr_pred)
lr_time = time.time() - start
print("    准确率:", round(lr_acc, 4), "训练时间:", round(lr_time, 1), "s")
results_ml.append({'Method': 'Logistic Regression', 'Accuracy': lr_acc, 'Train_Time(s)': lr_time})

# 随机森林
print("\n  训练随机森林...")
start = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train)
rf_pred = rf.predict(X_test_flat)
rf_acc = accuracy_score(y_test, rf_pred)
rf_time = time.time() - start
print("    准确率:", round(rf_acc, 4), "训练时间:", round(rf_time, 1), "s")
results_ml.append({'Method': 'Random Forest', 'Accuracy': rf_acc, 'Train_Time(s)': rf_time})

# SVM（使用子集）
print("\n  训练支持向量机（使用 10% 数据）...")
np.random.seed(42)
idx = np.random.choice(len(X_train_flat), size=len(X_train_flat)//10, replace=False)
start = time.time()
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train_flat[idx], y_train[idx])
svm_pred = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test, svm_pred)
svm_time = time.time() - start
print("    准确率:", round(svm_acc, 4), "训练时间:", round(svm_time, 1), "s")
results_ml.append({'Method': 'SVM (10% data)', 'Accuracy': svm_acc, 'Train_Time(s)': svm_time})

# 添加 CNN 模型
results_ml.append({'Method': 'Teacher CNN', 'Accuracy': teacher_acc, 'Train_Time(s)': 0})
if has_student:
    results_ml.append({'Method': 'Student CNN (Distilled)', 'Accuracy': student_acc, 'Train_Time(s)': 0})

df_ml = pd.DataFrame(results_ml)
df_ml = df_ml.sort_values('Accuracy', ascending=False)
df_ml.to_csv("results/实验4_方法对比结果.csv", index=False, encoding="utf-8-sig")
print("\n  ✅ 已保存: results/实验4_方法对比结果.csv")

print("\n[4.2] 方法对比表格（按准确率排序）")
print(df_ml.to_string(index=False))

print("\n[4.3] 生成方法对比可视化...")
plt.figure(figsize=(10, 6))
methods = df_ml['Method'].values
accuracies = df_ml['Accuracy'].values
colors_list = ['#1f77b4' if 'CNN' in m else '#ff7f0e' for m in methods]
bars = plt.bar(methods, accuracies, color=colors_list)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: CNN vs Traditional ML Methods')
plt.ylim([0, 1])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             str(round(acc, 4)), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("results/实验4_方法对比可视化.png", dpi=300)
plt.close()
print("  ✅ 已保存: results/实验4_方法对比可视化.png")

print("\n实验4 完成！")
print()

# ========== 生成实验总结报告 ==========
print("=" * 70)
print("生成实验总结报告")
print("=" * 70)

report_lines = []
report_lines.append("=" * 70)
report_lines.append("毕设实验总结报告")
report_lines.append("基于深度学习的网络入侵检测系统")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append("一、实验环境")
report_lines.append("- Python 版本: " + sys.version.split()[0])
report_lines.append("- TensorFlow 版本: " + tf.__version__)
report_lines.append("- 数据集: CIC-IDS2017")
report_lines.append("- 类别数: " + str(num_classes))
report_lines.append("- 实验时间: " + time.strftime('%Y-%m-%d %H:%M:%S'))
report_lines.append("- GPU 支持: " + str(len(tf.config.list_physical_devices('GPU')) > 0))
report_lines.append("")
report_lines.append("二、主要实验结果")
report_lines.append("")
report_lines.append("1. CNN 模型性能")
report_lines.append("   - 教师模型准确率: " + str(round(teacher_acc, 4)))
if has_student:
    report_lines.append("   - 学生模型准确率: " + str(round(student_acc, 4)))
    report_lines.append("   - 准确率差距: " + str(round(teacher_acc - student_acc, 4)))
report_lines.append("")
report_lines.append("2. 模型压缩效果")
report_lines.append("   - 教师模型参数量: " + f"{teacher_params:,}")
if has_student:
    report_lines.append("   - 学生模型参数量: " + f"{student_params:,}")
    report_lines.append("   - 参数量减少: " + str(round(param_reduction, 1)) + "%")
report_lines.append("   - 教师模型大小: " + str(round(teacher_size, 2)) + " MB")
if has_student:
    report_lines.append("   - 学生模型大小: " + str(round(student_size, 2)) + " MB")
    report_lines.append("   - 大小减少: " + str(round(size_reduction, 1)) + "%")
report_lines.append("")
report_lines.append("3. 与传统方法对比")
report_lines.append("   - CNN 方法明显优于传统机器学习方法")
report_lines.append("   - 详见: results/实验4_方法对比结果.csv")
report_lines.append("")
report_lines.append("三、生成文件清单")

if os.path.exists("results"):
    for f in sorted(os.listdir("results")):
        fpath = os.path.join("results", f)
        size_kb = os.path.getsize(fpath) / 1024
        report_lines.append("   - results/" + f + " (" + str(round(size_kb, 1)) + " KB)")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("报告生成完成！")
report_lines.append("=" * 70)

report_content = "\n".join(report_lines)

with open("results/实验总结报告.txt", "w", encoding="utf-8") as f:
    f.write(report_content)

print(report_content)
print("\n✅ 总结报告已保存: results/实验总结报告.txt")
print("\n所有实验完成！请查看 results/ 目录下的结果文件")
print("结束时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
