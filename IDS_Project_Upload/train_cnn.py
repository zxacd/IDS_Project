import tensorflow as tf
import logging
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense,
    Add, Multiply, Reshape, GlobalAveragePooling2D as GAP
)
from tensorflow.keras.layers import GlobalAveragePooling2D as GAP2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)

# 加载数据
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
X_val   = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy'))
X_test  = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
y_test  = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

num_classes = int(np.max(y_train)) + 1
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test, num_classes)

INPUT_SHAPE = (32, 32, 1)
print(f'数据加载完成: X_train={X_train.shape}, num_classes={num_classes}')

# ========== 残差块 + 通道注意力 (SE Block) ==========
def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation 通道注意力"""
    filters = input_tensor.shape[-1]
    se = GAP2D()(input_tensor)
    se = Reshape((filters,))(se)
    se = Dense(max(filters // ratio, 1), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([input_tensor, se])

def residual_block(x, filters, kernel_size=3, downsample=False):
    """带通道注意力的残差块"""
    identity = x
    stride = 2 if downsample else 1

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # 通道注意力
    x = se_block(x)

    # 捷径连接 (shortcut)
    if downsample or identity.shape[-1] != filters:
        identity = Conv2D(filters, 1, strides=stride, padding='same')(identity)
        identity = BatchNormalization()(identity)

    x = Add()([x, identity])
    x = Activation('relu')(x)
    return x

# ========== 构建模型 ==========
inputs = Input(shape=INPUT_SHAPE)

# 初始卷积
x = Conv2D(32, 3, strides=1, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# 残差阶段1: 32x32, 32通道
x = residual_block(x, 32)
x = residual_block(x, 32)

# 残差阶段2: 16x16, 64通道 (下采样)
x = residual_block(x, 64, downsample=True)
x = residual_block(x, 64)

# 残差阶段3: 8x8, 128通道 (下采样)
x = residual_block(x, 128, downsample=True)
x = residual_block(x, 128)

# 残差阶段4: 4x4, 256通道 (下采样)
x = residual_block(x, 256, downsample=True)
x = residual_block(x, 256)

# 全局池化 + 全连接
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    batch_size=512,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
logger.info('✅ 教师模型训练完成！')

# ========== 保存训练曲线 ==========
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = list(range(1, len(acc) + 1))

# 准确率曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,      label='训练准确率', linewidth=2)
plt.plot(epochs_range, val_acc,  label='验证准确率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,      label='训练损失',   linewidth=2)
plt.plot(epochs_range, val_loss,  label='验证损失',   linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/训练曲线_教师模型.png', dpi=300)
plt.close()
logger.info('✅ 训练曲线已保存: results/训练曲线_教师模型.png')

# 保存history为json（方便后续复用）
with open('results/训练历史_教师模型.json', 'w') as f:
    import json as json_lib
    data = {
        'accuracy':      [float(v) for v in acc],
        'val_accuracy':  [float(v) for v in val_acc],
        'loss':          [float(v) for v in loss],
        'val_loss':      [float(v) for v in val_loss],
        'epochs': epochs_range
    }
    json_lib.dump(data, f, indent=2)
logger.info('✅ 训练历史已保存: results/训练历史_教师模型.json')

# 最终评估
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
logger.info(f'测试集准确率: {test_acc:.4f}')
