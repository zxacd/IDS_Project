import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 强制 CPU，避免 CUDA 问题

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import joblib

MODEL_DIR = "models"
PROCESSED_DIR = "data/processed"

# 加载数据（使用少量样本加速，可选）
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")[:5000]   # 仅用5000样本
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")[:5000]

original_model = tf.keras.models.load_model(f"{MODEL_DIR}/final_model.h5")
num_classes = original_model.output_shape[-1]
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)

# === 剪枝 ===
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.3,
        final_sparsity=0.6,
        begin_step=0,
        end_step=2000
    )
}
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(original_model, **pruning_params)
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pruned_model.fit(X_train, y_train_cat, batch_size=32, epochs=4,
                 callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
pruned_model.save(f"{MODEL_DIR}/pruned_model.h5")
print("✅ 剪枝完成，模型已保存")
