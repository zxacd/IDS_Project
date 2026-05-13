"""
IDS_Project 运行测试脚本
用于在 VS Code 中验证项目是否能正常运行
"""
import sys
import traceback

def test_imports():
    """测试所有必要的库是否正确安装"""
    print("=" * 60)
    print("1. 测试库导入")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")

        import numpy as np
        print(f"✅ NumPy: {np.__version__}")

        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")

        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")

        import joblib
        print(f"✅ Joblib: 已安装")

        # 检查 GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU 可用: {len(gpus)} 个")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("⚠️  未检测到 GPU，将使用 CPU")

        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("2. 测试数据加载")
    print("=" * 60)
    try:
        import numpy as np
        import os

        data_dir = "data/processed"
        files = ["X_train.npy", "X_val.npy", "X_test.npy",
                 "y_train.npy", "y_val.npy", "y_test.npy"]

        for f in files:
            path = os.path.join(data_dir, f)
            if os.path.exists(path):
                data = np.load(path)
                print(f"✅ {f}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"❌ {f}: 文件不存在")
                return False

        print("✅ 所有数据文件加载成功")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("3. 测试模型加载")
    print("=" * 60)
    try:
        import tensorflow as tf
        import numpy as np
        import os

        models_dir = "models"

        # 测试教师模型
        teacher_path = os.path.join(models_dir, "final_model.h5")
        if os.path.exists(teacher_path):
            teacher = tf.keras.models.load_model(teacher_path)
            print(f"✅ 教师模型加载成功")
            print(f"   - 输入形状: {teacher.input_shape}")
            print(f"   - 输出形状: {teacher.output_shape}")
            print(f"   - 参数量: {teacher.count_params():,}")
        else:
            print(f"❌ 教师模型不存在: {teacher_path}")
            return False

        # 测试学生模型
        student_path = os.path.join(models_dir, "student_model_optimized.h5")
        if os.path.exists(student_path):
            student = tf.keras.models.load_model(student_path)
            print(f"✅ 学生模型加载成功")
            print(f"   - 输入形状: {student.input_shape}")
            print(f"   - 输出形状: {student.output_shape}")
            print(f"   - 参数量: {student.count_params():,}")
        else:
            print(f"⚠️  学生模型不存在: {student_path}")

        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False

def test_prediction():
    """测试模型预测功能"""
    print("\n" + "=" * 60)
    print("4. 测试模型预测")
    print("=" * 60)
    try:
        import tensorflow as tf
        import numpy as np
        import os

        # 加载数据
        X_test = np.load("data/processed/X_test.npy")
        y_test = np.load("data/processed/y_test.npy")
        num_classes = tf.keras.utils.to_categorical(y_test).shape[1]

        # 加载模型
        teacher = tf.keras.models.load_model("models/final_model.h5")

        # 测试预测
        print(f"测试数据形状: {X_test.shape}")
        print(f"类别数量: {num_classes}")

        # 用前5个样本测试
        test_samples = X_test[:5]
        predictions = teacher.predict(test_samples, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)

        print(f"✅ 预测成功")
        print(f"   - 测试样本数: 5")
        print(f"   - 预测形状: {predictions.shape}")
        print(f"   - 预测类别: {pred_classes}")

        # 计算准确率（完整测试集）
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
        _, accuracy = teacher.evaluate(X_test, y_test_cat, verbose=0)
        print(f"   - 测试集准确率: {accuracy:.4f}")

        return True
    except Exception as e:
        print(f"❌ 预测测试失败: {e}")
        traceback.print_exc()
        return False

def test_preprocessing():
    """测试预处理模块"""
    print("\n" + "=" * 60)
    print("5. 测试预处理模块")
    print("=" * 60)
    try:
        import pandas as pd
        import os

        # 检查原始数据
        raw_dir = "data/raw"
        if os.path.exists(raw_dir):
            csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
            print(f"✅ 找到 {len(csv_files)} 个原始 CSV 文件")
            for f in csv_files[:3]:  # 只显示前3个
                path = os.path.join(raw_dir, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   - {f}: {size_mb:.1f} MB")
        else:
            print(f"⚠️  原始数据目录不存在: {raw_dir}")

        # 测试预处理脚本能否导入
        print("\n测试 preprocess.py 模块导入...")
        import preprocess
        print("✅ preprocess.py 模块导入成功")

        return True
    except Exception as e:
        print(f"❌ 预处理测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("IDS_Project 运行测试")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("库导入", test_imports()))
    results.append(("数据加载", test_data_loading()))
    results.append(("模型加载", test_model_loading()))
    results.append(("模型预测", test_prediction()))
    results.append(("预处理模块", test_preprocessing()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！项目可以在 VS Code 中正常运行")
    else:
        print("⚠️  部分测试失败，请检查上面的错误信息")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
