import tensorflow as tf
import os

model = tf.keras.models.load_model("models/pruned_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("models/dynamic_quant.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ 动态量化完成！")