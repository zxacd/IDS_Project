"""
实验 1：CNN 入侵检测模型性能评估
生成分类报告、混淆矩阵等实验结果
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("实验 1：CNN 入侵检测模型性能评估")
print("=" * 60)

# 加载测试数据
print("\n[1] 加载数据和模型...")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
num_classes = int(np.max(y_test)) + 1
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"测试集大小: {X_test.shape[0]}")
print(f"类别数: {num_classes}")

# 加载标签名称
try:
    le = joblib.load("models/label_encoder.pkl")
    class_names = list(le.classes_)
    print(f"类别名称: {class_names}")
except:
    class_names = [str(i) for i in range(num_classes)]
    print(f"使用数字标签: 0 ~ {num_classes-1}")

# 加载教师模型
teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 加载学生模型
student_path = "models/student_model_optimized.h5"
has_student = os.path.exists(student_path)
if has_student:
    student = tf.keras.models.load_model(student_path, compile=False)
    student.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ 学生模型已加载")
else:
    print("⚠️  学生模型不存在，仅评估教师模型")

# 生成预测
print("\n[2] 生成预测结果...")
teacher_pred = teacher.predict(X_test, verbose=0)
teacher_pred_classes = np.argmax(teacher_pred, axis=1)
teacher_acc = np.mean(teacher_pred_classes == y_test)
print(f"教师模型准确率: {teacher_acc:.4f}")

if has_student:
    student_pred = student.predict(X_test, verbose=0)
    student_pred_classes = np.argmax(student_pred, axis=1)
    student_acc = np.mean(student_pred_classes == y_test)
    print(f"学生模型准确率: {student_acc:.4f}")

# 生成分类报告
print("\n[3] 生成分类报告...\n")

# 教师模型
print("=" * 60)
print("教师模型分类报告")
print("=" * 60)
teacher_report = classification_report(y_test, teacher_pred_classes, 
                                     target_names=class_names, digits=4)
print(teacher_report)

with open("results/teacher_classification_report.txt", "w", encoding="utf-8") as f:
    f.write("教师模型分类报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(teacher_report)
    f.write(f"\n\n总体准确率: {teacher_acc:.4f}\n")
print("✅ 教师模型报告已保存: results/teacher_classification_report.txt")

# 学生模型
if has_student:
    print("\n" + "=" * 60)
    print("学生模型分类报告")
    print("=" * 60)
    student_report = classification_report(y_test, student_pred_classes,
                                         target_names=class_names, digits=4)
    print(student_report)
    
    with open("results/student_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("学生模型分类报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(student_report)
        f.write(f"\n\n总体准确率: {student_acc:.4f}\n")
    print("✅ 学生模型报告已保存: results/student_classification_report.txt")

# 生成混淆矩阵
print("\n[4] 生成混淆矩阵...")

plt.figure(figsize=(12, 10))
cm_teacher = confusion_matrix(y_test, teacher_pred_classes)
sns.heatmap(cm_teacher, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Teacher Model', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("results/confusion_matrix_teacher.png", dpi=300)
plt.close()
print("✅ 教师模型混淆矩阵已保存: results/confusion_matrix_teacher.png")

if has_student:
    plt.figure(figsize=(12, 10))
    cm_student = confusion_matrix(y_test, student_pred_classes)
    sns.heatmap(cm_student, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Student Model', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_student.png", dpi=300)
    plt.close()
    print("✅ 学生模型混淆矩阵已保存: results/confusion_matrix_student.png")

# 汇总表格
print("\n[5] 生成汇总表格...")
from sklearn.metrics import precision_recall_fscore_support

teacher_precision, teacher_recall, teacher_f1, _ = precision_recall_fscore_support(
    y_test, teacher_pred_classes, average=None
)

summary_data = {
    'Class': class_names,
    'Teacher_Precision': [f"{x:.4f}" for x in teacher_precision],
    'Teacher_Recall': [f"{x:.4f}" for x in teacher_recall],
    'Teacher_F1': [f"{x:.4f}" for x in teacher_f1],
}

if has_student:
    student_precision, student_recall, student_f1, _ = precision_recall_fscore_support(
        y_test, student_pred_classes, average=None
    )
    summary_data['Student_Precision'] = [f"{x:.4f}" for x in student_precision]
    summary_data['Student_Recall'] = [f"{x:.4f}" for x in student_recall]
    summary_data['Student_F1'] = [f"{x:.4f}" for x in student_f1]

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("results/classification_results.csv", index=False, encoding="utf-8-sig")
print("✅ 分类结果已保存: results/classification_results.csv")
print("\n" + df_summary.to_string(index=False))

print("\n" + "=" * 60)
print("实验 1 完成！")
print("=" * 60)
