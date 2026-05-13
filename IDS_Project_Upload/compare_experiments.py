import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

MODEL_DIR = 'models'

# 使用完整测试集进行准确评估
X_test = np.load('data/processed/X_test.npy')
y_true = np.load('data/processed/y_test.npy')
print(f"测试集大小: {len(y_true)}")

models = {
    "教师模型": f"{MODEL_DIR}/final_model.h5",
    "剪枝模型": f"{MODEL_DIR}/pruned_model.h5",
    "学生模型(基础)": f"{MODEL_DIR}/student_model.h5",
    "学生模型(优化)": f"{MODEL_DIR}/student_model_optimized.h5",
    "动态量化(快速)": f"{MODEL_DIR}/quantized_model_dynamic.tflite"
}

for name, path in models.items():
    if not os.path.exists(path):
        print(f"{name}: 文件 {path} 不存在，跳过")
        continue
    print(f"正在测试 {name}...")
    if path.endswith('.h5'):
        m = tf.keras.models.load_model(path)
        pred = np.argmax(m.predict(X_test, verbose=0), axis=1)
    else:
        try:
            interp = tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]['index']
            out = interp.get_output_details()[0]['index']
            preds = []
            for x in X_test:
                input_data = x[np.newaxis, ...].astype(np.float32)
                interp.set_tensor(inp, input_data)
                interp.invoke()
                preds.append(np.argmax(interp.get_tensor(out)))
            pred = np.array(preds)
        except Exception as e:
            print(f"  TFLite 推理失败: {e}")
            continue
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average='macro')
    print(f"{name}: 准确率={acc:.4f} F1={f1:.4f}")