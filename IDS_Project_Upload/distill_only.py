import os

import tensorflow as tf
import numpy as np

MODEL_DIR = "models"
PROCESSED_DIR = "data/processed"

teacher = tf.keras.models.load_model(f"{MODEL_DIR}/final_model.h5")

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_val = np.load(f"{PROCESSED_DIR}/X_val.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_val = np.load(f"{PROCESSED_DIR}/y_val.npy")
num_classes = teacher.output_shape[-1]
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

student = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
])

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = 3.0
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
            loss = loss_hard + 0.3 * loss_soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        loss_hard = self.loss_fn(y, student_pred)
        return {
            "val_loss": loss_hard,
            "val_accuracy": tf.keras.metrics.categorical_accuracy(y, student_pred)
        }

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
distiller.fit(X_train, y_train_cat,
              validation_data=(X_val, y_val_cat),
              epochs=30, batch_size=256, verbose=1,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
              ])
student.save(f"{MODEL_DIR}/student_model.h5")
print("✅ 知识蒸馏完成，学生模型已保存")
