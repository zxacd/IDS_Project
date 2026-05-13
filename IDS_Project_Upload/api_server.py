from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter, get_remote_address
import numpy as np
import tensorflow as tf
import joblib
from skimage.transform import resize
import logging
import os
import signal
import sys

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per minute"])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = None
scaler = None
label_encoder = None
feature_names = None
MODEL_DIR = 'models'
FEATURE_MAP_SIZE = (32, 32)

def handle_shutdown(signum, frame):
    logger.info("🛑 服务关闭")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)

def load_models():
    global model, scaler, label_encoder, feature_names
    try:
        model = tf.keras.models.load_model(f"{MODEL_DIR}/final_model.h5")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
        logger.info("✅ 模型加载成功")
        return True
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return False

def validate(data):
    if not isinstance(data, dict):
        return False, "必须JSON"
    try:
        ordered = [data[col] for col in feature_names]
        return True, ordered
    except Exception as e:
        return False, f"特征错误: {e}"

def preprocess(features):
    x = scaler.transform([features])
    side = int(np.ceil(np.sqrt(x.shape[1])))
    padded = np.pad(x[0], (0, side**2 - x.shape[1]))
    img = padded.reshape(side, side)
    img = resize(img, FEATURE_MAP_SIZE, mode='reflect').astype(np.float32)
    return img[None, ..., None]

@app.route('/predict', methods=['POST'])
@limiter.limit("120 per minute")
def predict():
    try:
        data = request.get_json()
        ok, val = validate(data)
        if not ok:
            return jsonify({"success": False, "error": val}), 400
        x = preprocess(val)
        pred = model.predict(x, verbose=0)
        cls = np.argmax(pred[0])
        conf = float(pred[0][cls])
        attack = label_encoder.inverse_transform([cls])[0]
        return jsonify({"success": True, "attack_type": attack, "confidence": conf})
    except Exception as e:
        logger.error(f"错误: {e}")
        return jsonify({"success": False, "error": "服务异常"}), 500

if __name__ == '__main__':
    if not load_models():
        sys.exit(1)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)