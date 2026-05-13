"""
模型加载修复工具 - 兼容 mixed_float16 旧模型
保存为 fix_model_loading.py，放在 IDS_Project 目录下
"""

import tensorflow as tf
import numpy as np
import json
import h5py
import types

print("正在修复 Keras 混合精度模型加载问题...")

# ========== Patch 1: Operation.from_config ==========
try:
    from tensorflow.keras.src.ops.operation import Operation

    _orig_op_from_config = Operation.from_config

    @classmethod
    def _new_op_from_config(cls, config):
        # 修复 dtype 字段
        if 'dtype' in config:
            d = config['dtype']
            if isinstance(d, str):
                # 将 mixed_float16 / mixed_bfloat16 字符串改为 float32
                config['dtype'] = 'float32'
            elif isinstance(d, dict):
                # 如果是字典格式，也改为 float32
                config['dtype'] = 'float32'
        # 调用原始方法
        return _orig_op_from_config.__func__(cls, config)

    Operation.from_config = _new_op_from_config
    print("  ✅ Patch Operation.from_config 成功")
except Exception as e:
    print(f"  ⚠️  Patch Operation.from_config 失败: {e}")

# ========== Patch 2: Layer.from_config ==========
try:
    from tensorflow.keras.layers import Layer

    _orig_layer_from_config = Layer.from_config

    @classmethod
    def _new_layer_from_config(cls, config):
        if 'dtype' in config:
            d = config['dtype']
            if isinstance(d, str) and ('mixed' in d or 'float16' in d):
                config['dtype'] = 'float32'
            elif isinstance(d, dict):
                config['dtype'] = 'float32'
        return _orig_layer_from_config.__func__(cls, config)

    Layer.from_config = _new_layer_from_config
    print("  ✅ Patch Layer.from_config 成功")
except Exception as e:
    print(f"  ⚠️  Patch Layer.from_config 失败: {e}")

# ========== Patch 3: 修复 model_from_config ==========
try:
    import tensorflow.keras.models as _km

    _orig_mfc = getattr(_km, 'model_from_config', None)
    if _orig_mfc:
        def _new_model_from_config(config, custom_objects=None):
            # 递归修复 config 中的 dtype
            def _fix(d):
                if isinstance(d, dict):
                    if 'dtype' in d:
                        v = d['dtype']
                        if isinstance(v, str) and ('mixed' in v or 'float16' in v):
                            d['dtype'] = 'float32'
                        elif isinstance(v, dict):
                            d['dtype'] = 'float32'
                    for k in d:
                        d[k] = _fix(d[k])
                elif isinstance(d, list):
                    return [_fix(i) for i in d]
                return d
            config = _fix(config)
            return _orig_mfc(config, custom_objects)
        setattr(_km, 'model_from_config', _new_model_from_config)
        print("  ✅ Patch model_from_config 成功")
except Exception as e:
    print(f"  ⚠️  Patch model_from_config 失败: {e}")

print("Patch 完成，现在可以尝试加载模型了。")
