import tensorflow as tf
import numpy as np
import time
import os

MODEL_DIR = 'models'
X_test = np.load('data/processed/X_test.npy')[:500].astype(np.float32)
print(f"Loaded {len(X_test)} test samples, shape: {X_test.shape}")

def test_model(path):
    if not os.path.exists(path):
        print(f"File {path} not found, skipping.")
        return
    print(f"\nTesting: {path}")
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]['index']
        out = interp.get_output_details()[0]['index']
        # 预热
        for _ in range(10):
            interp.set_tensor(inp, X_test[0:1])
            interp.invoke()
        times = []
        for x in X_test:
            s = time.time()
            interp.set_tensor(inp, x[None])
            interp.invoke()
            times.append(time.time() - s)
        avg = np.mean(times) * 1000
        print(f"Average latency: {avg:.2f} ms")
        print(f"Throughput: {int(1000 / avg)} QPS")
    except Exception as e:
        print(f"Error: {e}")

# 只测试快速动态量化模型，慢速模型已注释
# test_model(f"{MODEL_DIR}/dynamic_quant.tflite")
test_model(f"{MODEL_DIR}/quantized_model_dynamic.tflite")