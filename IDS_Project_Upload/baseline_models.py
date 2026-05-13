#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线模型对比实验
训练 SVM、XGBoost、CNN-LSTM 并与教师模型对比
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import joblib
import json
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = 'models'
DATA_DIR = 'data/processed'
RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 加载数据 ==========
print("=" * 60)
print("加载数据...")
X_test = np.load(f'{DATA_DIR}/X_test.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')
X_train = np.load(f'{DATA_DIR}/X_train.npy')
y_train = np.load(f'{DATA_DIR}/y_train.npy')

le = joblib.load(f'{MODEL_DIR}/label_encoder.pkl')
class_names = list(le.classes_)
num_classes = len(class_names)

# 为传统ML模型准备1D特征（把32x32x1展平)
X_train_1d = X_train.reshape(X_train.shape[0], -1)
X_test_1d  = X_test.reshape(X_test.shape[0], -1)

# 标准化（SVM/Gradient Boosting 需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_1d.astype(np.float64))
X_test_scaled  = scaler.transform(X_test_1d.astype(np.float64))

# 标签 one-hot（XGBoost 需要）
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)
print(f'数据加载完成。训练集: {X_train.shape}, 测试集: {X_test.shape}')
print(f'1D特征维度: {X_train_1d.shape[1]}')
print(f'类别数: {num_classes}')

results = {}

# ========== 1. 随机森林（已有，补跑）==========
print("\n" + "=" * 60)
print("[1/4] 训练随机森林 (Random Forest)...")
print("=" * 60)
from sklearn.ensemble import RandomForestClassifier

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rf.fit(X_train_scaled, y_train)
t1 = time.time()

rf_pred  = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average='macro', zero_division=0)
rf_prec = precision_score(y_test, rf_pred, average='macro', zero_division=0)
rf_rec  = recall_score(y_test, rf_pred, average='macro', zero_division=0)

print(f'  准确率:   {rf_acc:.4f}')
print(f'  宏F1:     {rf_f1:.4f}')
print(f'  训练时间: {t1-t0:.1f}s')

results['随机森林'] = {
    'accuracy': float(rf_acc),
    'macro_f1':  float(rf_f1),
    'macro_precision': float(rf_prec),
    'macro_recall':  float(rf_rec),
    'train_time': t1 - t0,
}

# ========== 2. SVM ==========
print("\n" + "=" * 60)
print("[2/4] 训练 SVM...")
print("=" * 60)
print("  (SVM 在大样本上很慢，使用子集加速...)")

# SVM 在大样本上很慢，采样加速
n_svm = min(50000, len(X_train_scaled))
idx = np.random.choice(len(X_train_scaled), n_svm, replace=False)
X_svm = X_train_scaled[idx]
y_svm = y_train[idx]

t0 = time.time()
svm = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    decision_function_shape='ovo',  # 多分类
    probability=True,
    random_state=42,
    verbose=False
)
svm.fit(X_svm, y_svm)
t1 = time.time()

svm_pred  = svm.predict(X_test_scaled)
svm_acc  = accuracy_score(y_test, svm_pred)
svm_f1   = f1_score(y_test, svm_pred, average='macro', zero_division=0)

print(f'  准确率:   {svm_acc:.4f}')
print(f'  宏F1:     {svm_f1:.4f}')
print(f'  训练时间: {t1-t0:.1f}s (使用 {n_svm} 样本训练)')
print(f'  ⚠️  SVM 使用采样加速，完整训练会非常慢')

results['SVM'] = {
    'accuracy': float(svm_acc),
    'macro_f1':  float(svm_f1),
    'train_time': t1 - t0,
    'note': f'使用{n_svm}样本训练'
}

# 保存 SVM 模型
joblib.dump(svm, f'{MODEL_DIR}/baseline_svm.pkl')
joblib.dump(scaler, f'{MODEL_DIR}/baseline_scaler.pkl')
print('  ✅ SVM 模型已保存')

# ========== 3. XGBoost ==========
print("\n" + "=" * 60)
print("[3/4] 训练 XGBoost...")
print("=" * 60)

try:
    import xgboost as xgb

    # XGBoost 需要整数标签，以及 DMatrix
    t0 = time.time()
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest  = xgb.DMatrix(X_test_scaled,  label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'nthread': -1,
    }

    bst = xgb.train(params, dtrain, num_boost_round=100)
    t1 = time.time()

    xgb_proba = bst.predict(dtest)
    xgb_pred  = np.argmax(xgb_proba, axis=1)
    xgb_acc   = accuracy_score(y_test, xgb_pred)
    xgb_f1    = f1_score(y_test, xgb_pred, average='macro', zero_division=0)

    print(f'  准确率:   {xgb_acc:.4f}')
    print(f'  宏F1:     {xgb_f1:.4f}')
    print(f'  训练时间: {t1-t0:.1f}s')

    results['XGBoost'] = {
        'accuracy': float(xgb_acc),
        'macro_f1':  float(xgb_f1),
        'train_time': t1 - t0,
    }

    # 保存
    bst.save_model(f'{MODEL_DIR}/baseline_xgboost.json')
    print('  ✅ XGBoost 模型已保存')

except ImportError:
    print('  ⚠️  xgboost 未安装，跳过 (pip install xgboost)')
    print('  （论文中可用随机森林结果代替 XGBoost）')

# ========== 4. CNN-LSTM ==========
print("\n" + "=" * 60)
print("[4/4] 训练 CNN-LSTM...")
print("=" * 60)

# CNN-LSTM 模型构建
def build_cnn_lstm(input_shape=(32, 32, 1), num_classes=15):
    """CNN + LSTM 混合模型"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, BatchNormalization, Activation,
        MaxPooling2D, Reshape, LSTM, Dense, Dropout, GlobalAveragePooling2D
    )

    inputs = Input(shape=input_shape)

    # CNN 部分
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    # 将 CNN 输出 reshape 为序列供 LSTM 处理
    # (None, 8, 8, 64) -> (None, 8, 8*64) -> (None, 8, 512)
    x = Reshape((8, 8 * 64))(x)

    # LSTM 部分
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

print('  构建 CNN-LSTM 模型...')
cnn_lstm = build_cnn_lstm(num_classes=num_classes)
cnn_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
cnn_lstm.summary(print_fn=lambda x: print('  ' + x))

# 加载验证集
X_val = np.load(f'{DATA_DIR}/X_val.npy')
y_val = np.load(f'{DATA_DIR}/y_val.npy')
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

t0 = time.time()
history = cnn_lstm.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    batch_size=256,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ],
    verbose=1
)
t1 = time.time()

# 评估
y_pred_proba = cnn_lstm.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
lstm_acc = accuracy_score(y_test, y_pred)
lstm_f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f'\n  CNN-LSTM 准确率: {lstm_acc:.4f}')
print(f'  CNN-LSTM 宏F1:    {lstm_f1:.4f}')
print(f'  训练时间: {t1-t0:.1f}s')

results['CNN-LSTM'] = {
    'accuracy': float(lstm_acc),
    'macro_f1':  float(lstm_f1),
    'train_time': t1 - t0,
}

cnn_lstm.save(f'{MODEL_DIR}/baseline_cnn_lstm.h5')
print('  ✅ CNN-LSTM 模型已保存')

# 保存 CNN-LSTM 训练曲线
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = list(range(1, len(acc) + 1))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='训练准确率', linewidth=2)
plt.plot(epochs_range, val_acc, label='验证准确率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN-LSTM Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='训练损失',   linewidth=2)
plt.plot(epochs_range, val_loss, label='验证损失',   linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN-LSTM Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/训练曲线_CNN-LSTM.png', dpi=300)
plt.close()
print('  ✅ CNN-LSTM 训练曲线已保存')

# ========== 汇总对比表 ==========
print("\n" + "=" * 70)
print("基线模型 vs 本文模型 - 对比汇总")
print("=" * 70)

# 动态读取完整评估报告中的本文模型结果（避免硬编码）
eval_json_path = f'{RESULT_DIR}/完整评估指标.json'
all_models = {}

if os.path.exists(eval_json_path):
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    for name, res in eval_results.items():
        all_models[name] = res
    print(f'  ✅ 已从 {eval_json_path} 加载本文模型结果')
else:
    print(f'  ⚠️  未找到 {eval_json_path}，跳过本文模型对比')
    print(f'  （请先运行 eval_full.py 生成评估报告）')

# 合并基线模型结果
all_models.update(results)
header = '{:<28}{:>10}{:>10}{:>15}'.format('模型', '准确率', '宏F1', '训练时间(s)')
print(header)
print("-" * 75)
for name, res in all_models.items():
    acc  = res.get('accuracy', 0)
    f1   = res.get('macro_f1', 0)
    time = res.get('train_time', None)
    time_str = f"{time:.1f}" if time else "N/A"
    print("{:<28}{:>10.4f}{:>10.4f}{:>15}".format(name, acc, f1, time_str))

# 保存结果
with open(f'{RESULT_DIR}/基线模型对比结果.json', 'w', encoding='utf-8') as f:
    json.dump(all_models, f, ensure_ascii=False, indent=2)

with open(f'{RESULT_DIR}/基线模型对比报告.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("基线模型对比实验报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(header + "\n")
    f.write("-" * 75 + "\n")
    for name, res in all_models.items():
        acc  = res.get('accuracy', 0)
        f1   = res.get('macro_f1', 0)
        time = res.get('train_time', None)
        time_str = f"{time:.1f}" if time else "N/A"
        f.write("{:<28}{:>10.4f}{:>10.4f}{:>15}\n".format(name, acc, f1, time_str))
    f.write("\n")
    f.write("注：SVM 使用 50000 样本训练（完整训练过慢）\n")
    f.write("    XGBoost 若未安装则跳过\n")

print(f"\n✅ 结果已保存: {RESULT_DIR}/基线模型对比报告.txt")
print("\n所有基线模型实验完成！")
