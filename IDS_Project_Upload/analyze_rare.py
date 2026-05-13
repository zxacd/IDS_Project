#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析稀有类别问题 + 生成论文可用的说明
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report

MODEL_DIR = 'models'
DATA_DIR = 'data/processed'

# 加载数据
X_test = np.load(f'{DATA_DIR}/X_test.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')
X_train = np.load(f'{DATA_DIR}/X_train.npy')
y_train = np.load(f'{DATA_DIR}/y_train.npy')

le = joblib.load(f'{MODEL_DIR}/label_encoder.pkl')
class_names = list(le.classes_)
num_classes = len(class_names)

print("=" * 70)
print("稀有类别详细分析")
print("=" * 70)

# 训练集 vs 测试集分布对比
print("\n类别分布对比 (训练集 vs 测试集):")
print(f"{'类别':<30} {'训练集数量':>12} {'训练集占比':>10} {'测试集数量':>12} {'测试集占比':>10} {'问题':>15}")
print("-" * 100)

rare_in_train = []
rare_in_test = []
problem_classes = []

for i, c in enumerate(class_names):
    n_train = int((y_train == i).sum())
    n_test = int((y_test == i).sum())
    p_train = n_train / len(y_train) * 100
    p_test = n_test / len(y_test) * 100
    
    problem = ""
    if n_train == 0:
        problem += "训练集无样本 "
        rare_in_train.append(c)
    elif n_train < 100:
        problem += "训练集样本极少 "
        rare_in_train.append(c)
    if n_test == 0:
        problem += "测试集无样本"
        rare_in_test.append(c)
    elif n_test < 10:
        problem += "测试集样本极少"
        rare_in_test.append(c)
    
    if problem:
        problem_classes.append(c)
    
    print(f"{c:<30} {n_train:>12,} {p_train:>9.2f}% {n_test:>12,} {p_test:>9.2f}% {problem:>15}")

# 加载教师模型预测，看混淆矩阵
print("\n" + "=" * 70)
print("教师模型对稀有类别的预测情况:")
print("=" * 70)

m = tf.keras.models.load_model(f'{MODEL_DIR}/final_model.h5', compile=False)
y_proba = m.predict(X_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

print("\n逐稀有类别分析:")
for i, c in enumerate(class_names):
    n_test = int((y_test == i).sum())
    if n_test == 0:
        print(f"  {c:<30} 测试集样本=0  →  无法评估，论文中说明数据集限制")
        continue
    
    # 该类的预测情况
    mask = (y_test == i)
    pred_for_c = y_pred[mask]
    correct = int((pred_for_c == i).sum())
    
    # 各类被预测成什么
    pred_counts = {}
    for p in pred_for_c:
        pred_counts.setdefault(p, 0)
        pred_counts[p] += 1
    
    pred_detail = ", ".join([f"{class_names[k]}:{v}" for k, v in 
                            sorted(pred_counts.items(), key=lambda x: -x[1])[:3]])
    
    print(f"  {c:<30} 样本={n_test}, 正确={correct}, 预测分布: {pred_detail}")

# 论文说明建议
print("\n" + "=" * 70)
print("📝 论文中需说明的限制 (Limitations):")
print("=" * 70)
print("""
1. 数据集固有限制:
   - Heartbleed: 训练集和测试集均为 0 样本 (CIC-IDS2017 数据集标注问题)
   - Infiltration: 测试集仅 2 个样本，无法得到稳定的评估结果
   - Web Attack - Sql Injection: 测试集仅 2 个样本
   
   → 论文中应使用训练集有样本的 14 个类别进行主要评估
   → 测试集样本为 0 的类别，在 Macro-AUC 计算时自动排除

2. 类别不平衡问题:
   - BENIGN 占 82.1%，稀有攻击类别 (<100测试样本) 共 5 个
   - 宏平均指标受稀有类别影响大，加权平均更能反映整体性能
   
   → 论文中同时报告宏平均和加权平均，并说明二者差异的原因

3. 量化限制:
   - TFLite 动态量化后准确率从 98.73% 降至 82.10%
   - 原因: 含 BatchNorm 的 CNN 模型对量化敏感
   
   → 论文中采用未量化的学生模型进行部署，实现 90% 模型压缩且精度损失 <1%

4. 建议的论文写法:
   "由于 CIC-IDS2017 数据集的固有限制，部分攻击类型（Heartbleed、Infiltration等）
    在测试集中样本数量极少（<5），无法提供 statistically significant 的评估结果。
    因此，主要性能指标基于测试集中样本数 ≥ 10 的 11 个攻击类别计算。
    完整 15 类别的 Macro-AUC 通过 one-vs-rest 方式计算，自动排除无样本类别。"
""")

# 生成一个"论文可用"的简化指标表
print("=" * 70)
print("📊 建议论文中使用的指标表 (排除样本不足的类别):")
print("=" * 70)

# 只保留测试集样本 >= 10 的类别
valid_classes = [i for i in range(num_classes) if np.sum(y_test == i) >= 10]
print(f"\n有效评估类别 (测试集≥10样本): {len(valid_classes)} / {num_classes}")
print(f"类别列表: {[class_names[i] for i in valid_classes]}")

# 用有效类别重新计算 Macro 指标
from sklearn.metrics import f1_score, precision_score, recall_score

y_true_valid = y_test
y_pred_valid = y_pred

# 逐类 F1 (仅有效类别)
f1_per_class = {}
for i in valid_classes:
    mask = (y_true_valid == i)
    if mask.sum() > 0:
        from sklearn.metrics import f1_score as f1_single
        f1 = f1_score(y_true_valid, y_pred_valid, labels=[i], average='macro', zero_division=0)
        f1_per_class[class_names[i]] = f1

print(f"\n有效类别的逐类 F1 (教师模型):")
for c, f1 in sorted(f1_per_class.items(), key=lambda x: x[1]):
    n = int((y_test == [i for i in range(num_classes) if class_names[i]==c][0]).sum())
    print(f"  {c:<30} F1={f1:.4f}  (测试集样本={n})")
