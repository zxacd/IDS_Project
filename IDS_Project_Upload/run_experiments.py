#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毕设实验脚本 - 简洁版
直接在 VS Code 中运行，生成所有实验结果
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys
import time

# 确保 results 目录存在
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("毕设实验：基于深度学习的网络入侵检测系统")
print("=" * 70)
print("开始时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
print()

# 加载数据
print("[1] 加载数据...")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

num_classes = int(np.max(y_test)) + 1
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print("  测试集:", X_test.shape, "训练集:", X_train.shape)
print("  类别数:", num_classes)

# 加载标签名称
try:
    le = joblib.load("models/label_encoder.pkl")
    class_names = list(le.classes_)
except:
    class_names = [str(i) for i in range(num_classes)]
print("  类别:", class_names[:5], "...")
print()

# 加载模型
print("[2] 加载模型...")
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

# ========== 实验1：生成分类报告 ==========
print("=" * 70)
print("实验1：分类性能评估")
print("=" * 70)

print("\n[1.1] 教师模型预测...")
teacher_pred = teacher.predict(X_test, verbose=0)
teacher_pred_classes = np.argmax(teacher_pred, axis=1)
teacher_acc = accuracy_score(y_test, teacher_pred_classes)
print("  教师模型准确率:", round(teacher_acc, 4))

if has_student:
    print("\n[1.2] 学生模型预测...")
    student_pred = student.predict(X_test, verbose=0)
    student_pred_classes = np.argmax(student_pred, axis=1)
    student_acc = accuracy_score(y_test, student_pred_classes)
    print("  学生模型准确率:", round(student_acc, 4))

# 保存分类报告
print("\n[1.3] 保存分类报告...")
teacher_report = classification_report(y_test, teacher_pred_classes,
                                     labels=list(range(num_classes)),
                                     target_names=class_names, digits=4)
with open("results/教师模型分类报告.txt", "w", encoding="utf-8") as f:
    f.write("教师模型分类报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(teacher_report)
    f.write("\n总体准确率: " + str(round(teacher_acc, 4)))
print("  ✅ 已保存: results/教师模型分类报告.txt")

if has_student:
    student_report = classification_report(y_test, student_pred_classes,
                                         labels=list(range(num_classes)),
                                         target_names=class_names, digits=4)
    with open("results/学生模型分类报告.txt", "w", encoding="utf-8") as f:
        f.write("学生模型分类报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(student_report)
        f.write("\n总体准确率: " + str(round(student_acc, 4)))
    print("  ✅ 已保存: results/学生模型分类报告.txt")

print("\n实验1 完成！")
print()

# ========== 实验2：模型对比 ==========
print("=" * 70)
print("实验2：模型对比分析")
print("=" * 70)

teacher_params = teacher.count_params()
teacher_size = os.path.getsize("models/final_model.h5") / (1024 * 1024)

print("\n[2.1] 参数量与模型大小...")
print("  教师模型参数量:", f"{teacher_params:,}")
print("  教师模型大小:", round(teacher_size, 2), "MB")

if has_student:
    student_params = student.count_params()
    student_size = os.path.getsize(student_path) / (1024 * 1024)
    param_red = (1 - student_params / teacher_params) * 100
    size_red = (1 - student_size / teacher_size) * 100
    
    print("  学生模型参数量:", f"{student_params:,}")
    print("  学生模型大小:", round(student_size, 2), "MB")
    print("  参数量减少:", round(param_red, 1), "%")
    print("  大小减少:", round(size_red, 1), "%")

print("\n[2.2] 准确率对比...")
print("  教师模型准确率:", round(teacher_acc, 4))
if has_student:
    print("  学生模型准确率:", round(student_acc, 4))
    acc_diff = teacher_acc - student_acc
    print("  准确率差距:", round(acc_diff, 4), "(", round(acc_diff*100, 2), "%)")

# 保存对比表格
print("\n[2.3] 保存对比表格...")
data = {
    '指标': ['准确率', '参数量', '模型大小(MB)'],
    '教师模型': [
        str(round(teacher_acc, 4)),
        f"{teacher_params:,}",
        str(round(teacher_size, 2))
    ]
}
if has_student:
    data['学生模型'] = [
        str(round(student_acc, 4)),
        f"{student_params:,}",
        str(round(student_size, 2))
    ]
    data['压缩效果'] = [
        "差距 " + str(round(acc_diff*100, 2)) + "%",
        "减少 " + str(round(param_red, 1)) + "%",
        "减少 " + str(round(size_red, 1)) + "%"
    ]

df = pd.DataFrame(data)
df.to_csv("results/实验2_模型对比.csv", index=False, encoding="utf-8-sig")
print("  ✅ 已保存: results/实验2_模型对比.csv")
print(df.to_string(index=False))

print("\n实验2 完成！")
print()

# ========== 实验3：生成可视化 ==========
print("=" * 70)
print("实验3：生成可视化图表")
print("=" * 70)

# 混淆矩阵 - 教师模型
print("\n[3.1] 生成混淆矩阵...")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, teacher_pred_classes)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Teacher Model (CNN)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("results/实验3_教师混淆矩阵.png", dpi=300)
plt.close()
print("  ✅ 已保存: results/实验3_教师混淆矩阵.png")

if has_student:
    plt.figure(figsize=(12, 10))
    cm2 = confusion_matrix(y_test, student_pred_classes)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Student Model (Distilled)', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("results/实验3_学生混淆矩阵.png", dpi=300)
    plt.close()
    print("  ✅ 已保存: results/实验3_学生混淆矩阵.png")

# 准确率对比图
if has_student:
    print("\n[3.2] 生成准确率对比图...")
    plt.figure(figsize=(8, 6))
    plt.bar(['Teacher (CNN)', 'Student (Distilled)'], [teacher_acc, student_acc],
            color=['#1f77b4', '#2ca02c'])
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    for i, v in enumerate([teacher_acc, student_acc]):
        plt.text(i, v + 0.01, str(round(v, 4)), ha='center')
    plt.tight_layout()
    plt.savefig("results/实验3_准确率对比.png", dpi=300)
    plt.close()
    print("  ✅ 已保存: results/实验3_准确率对比.png")

print("\n实验3 完成！")
print()

# ========== 生成总结报告 ==========
print("=" * 70)
print("生成总结报告")
print("=" * 70)

report = []
report.append("=" * 70)
report.append("毕设实验总结报告")
report.append("基于深度学习的网络入侵检测系统")
report.append("=" * 70)
report.append("")
report.append("一、实验环境")
report.append("  - Python 版本: " + sys.version.split()[0])
report.append("  - TensorFlow 版本: " + tf.__version__)
report.append("  - 数据集: CIC-IDS2017")
report.append("  - 类别数: " + str(num_classes))
report.append("  - GPU 支持: " + str(len(tf.config.list_physical_devices('GPU')) > 0))
report.append("")
report.append("二、主要实验结果")
report.append("  - 教师模型准确率: " + str(round(teacher_acc, 4)))
if has_student:
    report.append("  - 学生模型准确率: " + str(round(student_acc, 4)))
    report.append("  - 准确率差距: " + str(round(acc_diff, 4)))
    report.append("  - 参数量减少: " + str(round(param_red, 1)) + "%")
    report.append("  - 模型大小减少: " + str(round(size_red, 1)) + "%")
report.append("")
report.append("三、生成文件清单")

if os.path.exists("results"):
    for f in sorted(os.listdir("results")):
        fpath = os.path.join("results", f)
        size_kb = os.path.getsize(fpath) / 1024
        report.append("  - results/" + f + " (" + str(round(size_kb, 1)) + " KB)")

report.append("")
report.append("=" * 70)
report.append("报告生成完成！")
report.append("=" * 70)

report_text = "\n".join(report)
print(report_text)

with open("results/实验总结报告.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n✅ 总结报告已保存: results/实验总结报告.txt")
print("\n所有实验完成！")
print("结束时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
