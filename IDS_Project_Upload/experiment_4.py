"""
实验 4：与其他方法对比
对比 CNN 知识蒸馏模型与传统机器学习方法
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("实验 4：与其他方法对比")
print("=" * 60)

# 加载数据
print("\n[1] 加载数据...")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

# 将 2D 特征图展平为 1D（用于传统 ML 方法）
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"训练集: {X_train_flat.shape}")
print(f"测试集: {X_test_flat.shape}")

# 加载 CNN 模型结果
print("\n[2] 评估 CNN 模型...")

# 教师模型
teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y_test_cat = tf.keras.utils.to_categorical(y_test, int(np.max(y_test))+1)
teacher_acc = teacher.evaluate(X_test, y_test_cat, verbose=0)[1]

# 学生模型
student_path = "models/student_model_optimized.h5"
if os.path.exists(student_path):
    student = tf.keras.models.load_model(student_path, compile=False)
    student.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    student_acc = student.evaluate(X_test, y_test_cat, verbose=0)[1]
else:
    student_acc = 0.0

print(f"教师模型 (CNN) 准确率: {teacher_acc:.4f}")
print(f"学生模型 (Distilled CNN) 准确率: {student_acc:.4f}")

# 传统机器学习方法
print("\n[3] 训练传统机器学习方法（用于对比）...")
print("注意：由于数据量较大，这会需要一些时间...")

results = []

# 1. 逻辑回归
print("\n  训练逻辑回归...")
start = time.time()
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
lr.fit(X_train_flat, y_train)
lr_pred = lr.predict(X_test_flat)
lr_acc = accuracy_score(y_test, lr_pred)
lr_time = time.time() - start
print(f"    准确率: {lr_acc:.4f}, 训练时间: {lr_time:.1f}s")
results.append({
    'Method': 'Logistic Regression',
    'Accuracy': lr_acc,
    'Train_Time (s)': lr_time,
    'Model_Size (MB)': sys.getsizeof(lr) / (1024*1024)
})

# 2. 支持向量机（使用子集，因为 SVM 在大数据上很慢）
print("\n  训练支持向量机（使用 10% 数据）...")
idx = np.random.choice(len(X_train_flat), size=len(X_train_flat)//10, random_state=42)
start = time.time()
svm = SVC(kernel='rbf', gamma='scale', probability=True)
svm.fit(X_train_flat[idx], y_train[idx])
svm_pred = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test, svm_pred)
svm_time = time.time() - start
print(f"    准确率: {svm_acc:.4f}, 训练时间: {svm_time:.1f}s")
results.append({
    'Method': 'SVM (10% data)',
    'Accuracy': svm_acc,
    'Train_Time (s)': svm_time,
    'Model_Size (MB)': sys.getsizeof(svm) / (1024*1024)
})

# 3. 随机森林
print("\n  训练随机森林...")
start = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train)
rf_pred = rf.predict(X_test_flat)
rf_acc = accuracy_score(y_test, rf_pred)
rf_time = time.time() - start
print(f"    准确率: {rf_acc:.4f}, 训练时间: {rf_time:.1f}s")
results.append({
    'Method': 'Random Forest',
    'Accuracy': rf_acc,
    'Train_Time (s)': rf_time,
    'Model_Size (MB)': sys.getsizeof(rf) / (1024*1024)
})

# 添加 CNN 模型结果
results.append({
    'Method': 'Teacher CNN',
    'Accuracy': teacher_acc,
    'Train_Time (s)': 0,  # 已训练好
    'Model_Size (MB)': os.path.getsize("models/final_model.h5") / (1024*1024)
})

if student_acc > 0:
    results.append({
        'Method': 'Student CNN (Distilled)',
        'Accuracy': student_acc,
        'Train_Time (s)': 0,  # 已训练好
        'Model_Size (MB)': os.path.getsize(student_path) / (1024*1024)
    })

# 保存结果
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Accuracy', ascending=False)
df_results.to_csv("results/comparison_with_ml_methods.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 对比结果已保存: results/comparison_with_ml_methods.csv")

# 打印对比表格
print("\n" + "=" * 60)
print("方法对比（按准确率排序）")
print("=" * 60)
print(df_results.to_string(index=False))

# 可视化
print("\n[4] 生成可视化图表...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 准确率对比
methods = df_results['Method'].values
accuracies = df_results['Accuracy'].values
colors = ['#1f77b4' if 'CNN' in m else '#ff7f0e' for m in methods]
bars1 = axes[0].bar(methods, accuracies, color=colors)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylim([0, 1])
axes[0].tick_params(axis='x', rotation=45)
axes[0]..grid(True, alpha=0.3)
for bar, acc in zip(bars1, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                  f'{acc:.4f}', ha='center', fontsize=9)

# 模型大小对比（仅 CNN 模型）
cnn_methods = [m for m in methods if 'CNN' in m]
cnn_sizes = [df_results[df_results['Method']==m]['Model_Size (MB)'].values[0] for m in cnn_methods]
colors2 = ['#1f77b4', '#2ca02c']
bars2 = axes[1].bar(cnn_methods, cnn_sizes, color=colors2)
axes[1].set_ylabel('Model Size (MB)')
axes[1].set_title('CNN Model Size Comparison')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)
for bar, size in zip(bars2, cnn_sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                  f'{size:.2f}MB', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("results/method_comparison.png", dpi=300)
plt.close()
print("✅ 方法对比图已保存: results/method_comparison.png")

print("\n" + "=" * 60)
print("实验 4 完成！")
print("=" * 60)
print("\n生成的结果文件:")
print("  - results/comparison_with_ml_methods.csv")
print("  - results/method_comparison.png")
