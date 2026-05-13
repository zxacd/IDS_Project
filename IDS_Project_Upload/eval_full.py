#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整评估脚本 v5 - 修复所有语法和逻辑错误
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import joblib
import json
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

# ========== 配置 ==========
MODEL_DIR = 'models'
DATA_DIR = 'data/processed'
RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 加载数据 ==========
print('=' * 60)
print('加载数据...')
X_test = np.load(f'{DATA_DIR}/X_test.npy')
y_true = np.load(f'{DATA_DIR}/y_test.npy')

le = joblib.load(f'{MODEL_DIR}/label_encoder.pkl')
class_names = list(le.classes_)
num_classes = len(class_names)

print(f'  类别数: {num_classes}')
print(f'  测试集: {X_test.shape}')

# 哪些类别在测试集中有样本
present_classes = [i for i in range(num_classes) if np.sum(y_true == i) > 0]
missing_classes = [i for i in range(num_classes) if np.sum(y_true == i) == 0]
rare_classes = [i for i in present_classes if np.sum(y_true == i) < 100]

print(f'  有样本的类别: {len(present_classes)} / {num_classes}')
if missing_classes:
    print(f'  测试集无样本的类别: {[class_names[i] for i in missing_classes]}')
if rare_classes:
    print(f'  稀有类别 (<100样本): {[class_names[i] for i in rare_classes]}')

# ========== AUC 计算 ==========
def compute_auc(y_true, y_proba):
    """
    逐类计算 one-vs-rest AUC，再取平均
    y_true: 1D整数标签
    y_proba: (n_samples, n_classes) 概率矩阵
    """
    if len(present_classes) < 2:
        return None, None
    try:
        aucs = []
        weights = []
        for c in present_classes:
            y_bin = (y_true == c).astype(int)
            n_pos = np.sum(y_bin)
            n_neg = len(y_true) - n_pos
            # 至少需要正负样本各1个
            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(y_bin, y_proba[:, c])
                aucs.append(auc)
                weights.append(n_pos)

        if len(aucs) < 2:
            return None, None

        auc_macro = float(np.mean(aucs))
        w = np.array(weights, dtype=float) / np.sum(weights)
        auc_weighted = float(np.average(aucs, weights=w))
        return auc_macro, auc_weighted
    except Exception as e:
        print(f'  [WARN] AUC计算失败: {e}')
        return None, None

# ========== Keras 模型评估 ==========
def eval_keras(path):
    m = tf.keras.models.load_model(path, compile=False)
    y_proba = m.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    return m, y_proba, y_pred

# ========== TFLite 模型评估 ==========
def eval_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    out_dim = interp.get_output_details()[0]['shape'][-1]
    print(f'  TFLite输出维度: {interp.get_output_details()[0]["shape"]}')

    y_preds = []
    y_probas = np.zeros((len(X_test), num_classes), dtype=np.float32)

    for i, x in enumerate(X_test):
        interp.set_tensor(inp_idx, x[np.newaxis, ...].astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(out_idx)[0].astype(np.float32)

        # softmax 归一化
        exp_out = np.exp(out - np.max(out))
        proba = exp_out / np.sum(exp_out)

        if out_dim >= num_classes:
            y_probas[i, :] = proba[:num_classes]
        else:
            y_probas[i, :out_dim] = proba

        y_preds.append(np.argmax(proba))

    return np.array(y_preds), y_probas

# ========== 主评估 ==========
all_results = {}

# --- Keras 模型 ---
keras_models = {
    '教师模型':     f'{MODEL_DIR}/final_model.h5',
    '学生模型优化': f'{MODEL_DIR}/student_model_optimized.h5',
    '学生模型基础': f'{MODEL_DIR}/student_model.h5',
    '剪枝模型':     f'{MODEL_DIR}/pruned_model.h5',
}

print('\n' + '=' * 60)
print('评估 Keras 模型...')
print('=' * 60)

for name, path in keras_models.items():
    if not os.path.exists(path):
        print(f'\n[跳过] {name}: 文件不存在')
        continue

    print(f'\n>>> {name}')

    try:
        m, y_proba, y_pred = eval_keras(path)
    except Exception as e:
        print(f'  [错误] 加载失败: {e}')
        continue

    # 基础指标
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f'  准确率:           {acc:.4f}')
    print(f'  宏平均 Precision:  {macro_prec:.4f}')
    print(f'  宏平均 Recall:     {macro_recall:.4f}')
    print(f'  宏平均 F1:        {macro_f1:.4f}')
    print(f'  加权平均 F1:      {weighted_f1:.4f}')

    # 稀有类别
    report_dict = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4, output_dict=True, zero_division=0
    )
    print('  稀有类别 F1:')
    for i in rare_classes:
        c = class_names[i]
        p = report_dict[c]['precision']
        r = report_dict[c]['recall']
        f1 = report_dict[c]['f1-score']
        print(f'    {c:25s} P={p:.4f}  R={r:.4f}  F1={f1:.4f}')

    # AUC
    auc_m, auc_w = compute_auc(y_true, y_proba)
    if auc_m is not None:
        print(f'  Macro AUC:        {auc_m:.4f}')
        print(f'  Weighted AUC:     {auc_w:.4f}')

    # 模型信息
    params = m.count_params()
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f'  参数量:  {params:,}')
    print(f'  模型大小: {size_mb:.2f} MB')

    all_results[name] = {
        'accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_prec),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'macro_auc': auc_m,
        'weighted_auc': auc_w,
        'params': int(params),
        'model_size_mb': float(size_mb),
        'per_class': {
            c: {
                'precision': report_dict[c]['precision'],
                'recall': report_dict[c]['recall'],
                'f1': report_dict[c]['f1-score'],
                'support': int(report_dict[c]['support'])
            }
            for c in class_names
        }
    }

# --- TFLite 模型 ---
tflite_models = {
    '动态量化(快速)': f'{MODEL_DIR}/quantized_model_dynamic.tflite',
}

print('\n' + '=' * 60)
print('评估 TFLite 模型...')
print('=' * 60)

for name, path in tflite_models.items():
    if not os.path.exists(path):
        print(f'\n[跳过] {name}: 文件不存在')
        continue

    print(f'\n>>> {name}')

    try:
        y_pred, y_proba = eval_tflite(path)
    except Exception as e:
        print(f'  [错误] {e}')
        continue

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f'  准确率:       {acc:.4f}')
    print(f'  宏平均 F1:    {macro_f1:.4f}')
    print(f'  宏平均 Precision: {macro_prec:.4f}')
    print(f'  宏平均 Recall:    {macro_recall:.4f}')

    auc_m, auc_w = compute_auc(y_true, y_proba)
    if auc_m is not None:
        print(f'  Macro AUC:   {auc_m:.4f}')

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f'  模型大小: {size_mb:.2f} MB')
    if acc < 0.95:
        print(f'  ⚠️  量化后性能显著下降 ({acc:.1%})，论文中需说明此限制')

    all_results[name] = {
        'accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_auc': auc_m,
        'weighted_auc': auc_w,
        'model_size_mb': float(size_mb),
    }

# ========== 保存结果 ==========
print('\n' + '=' * 60)
print('保存结果...')

with open(f'{RESULT_DIR}/完整评估指标.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f'  ✅ {RESULT_DIR}/完整评估指标.json')

with open(f'{RESULT_DIR}/完整评估指标报告.txt', 'w', encoding='utf-8') as f:
    f.write('=' * 70 + '\n')
    f.write('毕设实验 - 完整评估指标报告\n')
    f.write('=' * 70 + '\n\n')

    for name, res in all_results.items():
        f.write('-' * 70 + '\n')
        f.write(f'模型: {name}\n')
        f.write('-' * 70 + '\n')
        metrics_to_write = [
            ('accuracy',        '准确率'),
            ('macro_precision', '宏平均 Precision'),
            ('macro_recall',    '宏平均 Recall'),
            ('macro_f1',       '宏平均 F1'),
            ('weighted_precision', '加权平均 Precision'),
            ('weighted_recall', '加权平均 Recall'),
            ('weighted_f1',    '加权平均 F1'),
            ('macro_auc',      'Macro AUC'),
            ('weighted_auc',   'Weighted AUC'),
        ]
        for key, label in metrics_to_write:
            if res.get(key) is not None:
                f.write(f'  {label}: {res[key]:.4f}\n')
        f.write(f'  模型大小: {res.get("model_size_mb", 0):.2f} MB\n')
        if res.get('params'):
            f.write(f'  参数量: {res["params"]:,}\n')
        f.write('\n')

    # 稀有类别汇总表
    f.write('\n' + '=' * 70 + '\n')
    f.write('稀有攻击类别 - 各模型 F1-Score 对比\n')
    f.write('=' * 70 + '\n')
    header = '类别'.ljust(28)
    for n in all_results:
        header += n[:14].ljust(14)
    f.write(header + '\n')
    f.write('-' * 70 + '\n')
    for i in rare_classes:
        c = class_names[i]
        row = c.ljust(28)
        for n, res in all_results.items():
            if 'per_class' in res and c in res['per_class']:
                row += f'{res["per_class"][c]["f1"]:.4f}'.ljust(14)
            else:
                row += 'N/A'.ljust(14)
        f.write(row + '\n')

print(f'  ✅ {RESULT_DIR}/完整评估指标报告.txt')

# ========== 控制台汇总 ==========
print('\n' + '=' * 75)
print('汇总结果')
print('=' * 75)
header = '模型'.ljust(22) + '准确率'.rjust(8) + '宏F1'.rjust(8) + 'AUC'.rjust(8) + '大小MB'.rjust(9) + '参数量'.rjust(12)
print(header)
print('-' * 75)
for name, res in all_results.items():
    auc_str = f'{res["macro_auc"]:.4f}' if res.get('macro_auc') else 'N/A'
    param_str = f'{res["params"]:,}' if res.get('params') else '-'
    row = (name[:18].ljust(22)
           + f'{res["accuracy"]:.4f}'.rjust(8)
           + f'{res["macro_f1"]:.4f}'.rjust(8)
           + auc_str.rjust(8)
           + f'{res["model_size_mb"]:.2f}'.rjust(9)
           + param_str.rjust(12))
    print(row)

print('\n✅ 所有评估完成！')
