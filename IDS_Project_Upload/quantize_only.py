import tensorflow as tf
import numpy as np
import os

MODEL_DIR = 'models'
PROCESSED_DIR = 'data/processed'
model = tf.keras.models.load_model(f"{MODEL_DIR}/pruned_model.h5")
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy").astype(np.float32)[:1000]

def representative_data_gen():
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
with open(f"{MODEL_DIR}/quant_model.tflite", 'wb') as f:
    f.write(tflite_model)

print("✅ 全整数量化完成！")