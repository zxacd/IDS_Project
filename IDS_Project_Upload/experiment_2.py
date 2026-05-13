"""
实验 2：知识蒸馏效果分析
测试不同温度参数和 alpha 值对蒸馏效果的影响
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("实验 2：知识蒸馏效果分析")
print("=" * 60)

# 加载数据
print("\n[1] 加载数据...")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

num_classes = int(np.max(y_train)) + 1
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")
print(f"类别数: {num_classes}")

# 加载教师模型
print("\n[2] 加载教师模型...")
teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
teacher_acc = teacher.evaluate(X_test, y_test_cat, verbose=0)[1]
print(f"教师模型准确率: {teacher_acc:.4f}")

# 定义学生模型构建函数
def build_student():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

# 自定义蒸馏模型
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.7):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            loss_hard = self.loss_fn(y, student_pred)
            loss_soft = tf.keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_pred / self.temperature),
                tf.nn.softmax(student_pred / self.temperature)
            ) * (self.temperature ** 2)
            loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {"loss": loss, "loss_hard": loss_hard, "loss_soft": loss_soft}
    
    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        loss_hard = self.loss_fn(y, student_pred)
        return {"val_loss": loss_hard, "val_accuracy": tf.keras.metrics.categorical_accuracy(y, student_pred)}
    
    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

# 测试不同参数组合
print("\n[3] 测试不同蒸馏参数...")
results = []

temperatures = [1.0, 3.0, 5.0, 7.0]
alphas = [0.3, 0.5, 0.7, 0.9]

for T in temperatures:
    for alpha in alphas:
        print(f"\n  训练: T={T}, alpha={alpha}")
        
        student = build_student()
        distiller = Distiller(student=student, teacher=teacher, temperature=T, alpha=alpha)
        distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        start_time = time.time()
        distiller.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=15,  # 快速测试
            batch_size=256,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # 评估
        student_acc = student.evaluate(X_test, y_test_cat, verbose=0)[1]
        acc_diff = teacher_acc - student_acc
        
        results.append({
            'Temperature': T,
            'Alpha': alpha,
            'Student_Accuracy': student_acc,
            'Accuracy_Diff': acc_diff,
            'Train_Time': train_time
        })
        
        print(f"    学生准确率: {student_acc:.4f}, 差距: {acc_diff:.4f}")

# 保存结果
df_results = pd.DataFrame(results)
df_results.to_csv("results/distillation_params_results.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 参数测试结果已保存: results/distillation_params_results.csv")

# 可视化
print("\n[4] 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, alpha in enumerate([0.3, 0.5, 0.7, 0.9]):
    ax = axes[i//2, i%2]
    subset = df_results[df_results['Alpha'] == alpha]
    ax.plot(subset['Temperature'], subset['Student_Accuracy'], marker='o', label=f'alpha={alpha}')
    ax.axhline(y=teacher_acc, color='r', linestyle='--', label='Teacher')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Alpha = {alpha}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/distillation_temperature_effect.png", dpi=300)
plt.close()
print("✅ 温度参数影响图已保存: results/distillation_temperature_effect.png")

fig, ax = plt.subplots(figsize=(10, 6))
for T in temperatures:
    subset = df_results[df_results['Temperature'] == T]
    ax.plot(subset['Alpha'], subset['Student_Accuracy'], marker='o', label=f'T={T}')

ax.axhline(y=teacher_acc, color='r', linestyle='--', label='Teacher')
ax.set_xlabel('Alpha (Hard Label Weight)')
ax.set_ylabel('Accuracy')
ax.set_title('Effect of Alpha on Distillation Performance')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/distillation_alpha_effect.png", dpi=300)
plt.close()
print("✅ Alpha 影响图已保存: results/distillation_alpha_effect.png")

# 最佳参数
best_idx = df_results['Student_Accuracy'].idxmax()
best_params = df_results.iloc[best_idx]
print("\n" + "=" * 60)
print("最佳蒸馏参数")
print("=" * 60)
print(f"温度 T: {best_params['Temperature']}")
print(f"Alpha: {best_params['Alpha']}")
print(f"学生准确率: {best_params['Student_Accuracy']:.4f}")
print(f"与教师差距: {best_params['Accuracy_Diff']:.4f}")

print("\n" + "=" * 60)
print("实验 2 完成！")
print("=" * 60)
