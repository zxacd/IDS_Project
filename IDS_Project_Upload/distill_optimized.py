import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json as json_lib

MODEL_DIR = 'models'
PROCESSED_DIR = 'data/processed'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)

# ========== 加载数据 ==========
teacher = tf.keras.models.load_model(f'{MODEL_DIR}/final_model.h5')

X_train = np.load(f'{PROCESSED_DIR}/X_train.npy')
y_train = np.load(f'{PROCESSED_DIR}/y_train.npy')
X_val   = np.load(f'{PROCESSED_DIR}/X_val.npy')
y_val   = np.load(f'{PROCESSED_DIR}/y_val.npy')
X_test  = np.load(f'{PROCESSED_DIR}/X_test.npy')
y_test  = np.load(f'{PROCESSED_DIR}/y_test.npy')

num_classes = teacher.output_shape[-1]
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

print(f'训练集: {X_train.shape}, 验证集: {X_val.shape}, 类别数: {num_classes}')

# ========== SE 通道注意力 + 残差块 ==========
def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((filters,))(se)
    se = tf.keras.layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

def residual_block(x, filters, kernel_size=3, downsample=False):
    identity = x
    stride = 2 if downsample else 1

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = se_block(x)

    if downsample or identity.shape[-1] != filters:
        identity = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(identity)
        identity = tf.keras.layers.BatchNormalization()(identity)

    x = tf.keras.layers.Add()([x, identity])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# ========== 构建学生模型（轻量，带残差+注意力）==========
inputs = tf.keras.layers.Input(shape=(32, 32, 1))

x = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = residual_block(x, 32)
x = residual_block(x, 32)

x = residual_block(x, 64, downsample=True)
x = residual_block(x, 64)

x = residual_block(x, 128, downsample=True)
x = residual_block(x, 128)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

student = tf.keras.Model(inputs=inputs, outputs=outputs)
student.summary()

# ========== 蒸馏器（修复版）==========
# 修复1：test_step 返回的 key 去掉 val_ 前缀，Keras 会自动加 val_
# 修复2：用 float32 计算损失，避免 mixed_float16 下 nan
# 修复3：KL散度改用数值稳定的手写实现

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.7):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def train_step(self, data):
        x, y = data
        # 教师预测（float32，不参与梯度）
        teacher_pred = tf.cast(self.teacher(x, training=False), tf.float32)
        with tf.GradientTape() as tape:
            student_pred_raw = self.student(x, training=True)
            student_pred = tf.cast(student_pred_raw, tf.float32)

            # 硬标签损失
            loss_hard = tf.cast(self.loss_fn(y, student_pred), tf.float32)

            # 软标签：数值稳定的 KL 散度，全部在 float32 计算
            T = self.temperature
            teacher_soft = tf.nn.log_softmax(teacher_pred / T)
            student_soft = tf.nn.log_softmax(student_pred / T)
            p_teacher = tf.nn.softmax(teacher_pred / T)
            loss_soft = tf.reduce_mean(
                tf.reduce_sum(p_teacher * (teacher_soft - student_soft), axis=-1)
            ) * (T ** 2)

            loss = tf.cast(self.alpha * loss_hard + (1 - self.alpha) * loss_soft, tf.float32)
            # 混合精度：缩放损失防止 float16 下溢
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, self.student.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        # 梯度裁剪防止爆炸
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # ★ 修复：计算训练准确率，用于绘图和监控
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(student_pred, 1), tf.argmax(tf.cast(y, tf.float32), 1)),
                tf.float32
            )
        )
        return {
            'loss':      loss,
            'loss_hard': loss_hard,
            'loss_soft': loss_soft,
            'accuracy':  acc,   # ← 新增，否则 history 里没有训练准确率
        }

    def test_step(self, data):
        x, y = data
        student_pred = tf.cast(self.student(x, training=False), tf.float32)
        loss = tf.cast(self.loss_fn(y, student_pred), tf.float32)
        acc  = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(student_pred, 1), tf.argmax(tf.cast(y, tf.float32), 1)),
                tf.float32
            )
        )
        # ★ 关键修复：key 不带 val_ 前缀，Keras 会自动加
        return {
            'loss':     loss,
            'accuracy': acc,
        }

# ★ 自定义 checkpoint 回调：只保存内部 student（Functional 模型，支持 h5）
class StudentCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, student_model, filepath, monitor='val_loss', verbose=1):
        super().__init__()
        self.student_model = student_model
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            if self.verbose:
                print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving student model.')
            self.best = current
            self.student_model.save(self.filepath)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, verbose=1
    ),
    StudentCheckpoint(
        student_model=student,
        filepath=f'{MODEL_DIR}/student_model_optimized.h5',
        monitor='val_loss',
        verbose=1,
    ),
]

distiller = Distiller(student=student, teacher=teacher, temperature=3.0, alpha=0.7)
distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

history = distiller.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=80,
    batch_size=256,
    verbose=1,
    callbacks=callbacks
)

# ========== 评估 ==========
test_loss, test_acc = teacher.evaluate(X_test, y_test_cat, verbose=0)
student_pred = student.predict(X_test, verbose=0)
student_acc = float(np.mean(
    np.argmax(student_pred, axis=1) == np.argmax(y_test_cat, axis=1)
))

print(f'\n教师模型测试精度: {test_acc:.4f}')
print(f'学生模型测试精度: {student_acc:.4f}')
print(f'精度差距: {test_acc - student_acc:.4f}')

student.save(f'{MODEL_DIR}/student_model_optimized.h5')
print('✅ 优化学生模型已保存')

# ========== 保存蒸馏训练曲线 ==========
# 修复：用实际存在的 key 取历史记录
hist = history.history
print('history keys:', list(hist.keys()))   # 打印一下方便调试

loss     = hist.get('loss',         [])
val_loss = hist.get('val_loss',     [])
acc_h    = hist.get('accuracy',     hist.get('val_accuracy', []))
val_acc  = hist.get('val_accuracy', [])
epochs_range = list(range(1, len(loss) + 1))

plt.figure(figsize=(12, 4))

# --- 准确率 ---
plt.subplot(1, 2, 1)
if acc_h:
    plt.plot(epochs_range[:len(acc_h)],  acc_h,  label='训练准确率', linewidth=2)
if val_acc:
    plt.plot(epochs_range[:len(val_acc)], val_acc, label='验证准确率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Distillation - Accuracy')
if acc_h or val_acc:
    plt.legend()
plt.grid(True, alpha=0.3)

# --- 损失 ---
plt.subplot(1, 2, 2)
if loss:
    plt.plot(epochs_range[:len(loss)],     loss,     label='训练损失',   linewidth=2)
if val_loss:
    plt.plot(epochs_range[:len(val_loss)], val_loss, label='验证损失',   linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Distillation - Loss')
if loss or val_loss:
    plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/训练曲线_蒸馏模型.png', dpi=300)
plt.close()
print('✅ 蒸馏训练曲线已保存: results/训练曲线_蒸馏模型.png')

# 保存历史
with open('results/训练历史_蒸馏模型.json', 'w') as f:
    json_lib.dump({
        'loss':         [float(v) for v in loss],
        'val_loss':     [float(v) for v in val_loss],
        'accuracy':     [float(v) for v in acc_h],
        'val_accuracy': [float(v) for v in val_acc],
        'epochs':       epochs_range
    }, f, indent=2)
print('✅ 蒸馏训练历史已保存')
