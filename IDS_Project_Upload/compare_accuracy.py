"""
比较教师模型和学生模型的测试准确率
输出到文件方便查看
"""
import tensorflow as tf
import numpy as np
import os
import sys

# 将输出同时写入文件和终端
output_file = "test_result.txt"
f = open(output_file, "w", encoding="utf-8")
sys.stdout = f
sys.stderr = f

try:
    # 加载测试数据
    print("正在加载测试数据...")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    # 正确计算类别数：标签从 0 开始，所以类别数 = 最大标签值 + 1
    num_classes = int(np.max(y_test)) + 1
    print(f"标签范围: 0 ~ {int(np.max(y_test))}, 类别数: {num_classes}")
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"测试集大小: {X_test.shape[0]}")
    print(f"类别数: {num_classes}")
    print()

    # 加载模型
    print("正在加载模型...")
    teacher = tf.keras.models.load_model("models/final_model.h5", compile=False)

    student_path = "models/student_model_optimized.h5"
    if os.path.exists(student_path):
        student = tf.keras.models.load_model(student_path, compile=False)
        student_loaded = True
    else:
        print(f"⚠️  学生模型不存在: {student_path}")
        print("请先运行: python distill_optimized.py")
        student_loaded = False

    print()

    # 评估教师模型
    print("=" * 60)
    print("评估教师模型...")
    print("=" * 60)
    teacher.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    teacher_loss, teacher_acc = teacher.evaluate(X_test, y_test_cat, verbose=0)
    print(f"教师模型 - 测试准确率: {teacher_acc:.4f} ({teacher_acc*100:.2f}%)")
    print(f"教师模型 - 测试损失: {teacher_loss:.4f}")
    print()

    # 评估学生模型
    if student_loaded:
        print("=" * 60)
        print("评估学生模型...")
        print("=" * 60)
        student.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        student_loss, student_acc = student.evaluate(X_test, y_test_cat, verbose=0)
        print(f"学生模型 - 测试准确率: {student_acc:.4f} ({student_acc*100:.2f}%)")
        print(f"学生模型 - 测试损失: {student_loss:.4f}")
        print()

        # 比较
        print("=" * 60)
        print("模型对比")
        print("=" * 60)
        diff = teacher_acc - student_acc
        print(f"教师模型准确率: {teacher_acc:.4f}")
        print(f"学生模型准确率: {student_acc:.4f}")
        print(f"准确率差距: {diff:.4f} ({diff*100:.2f}%)")

        if diff < 0.02:
            print("✅ 学生模型表现优秀！差距小于 2%")
        elif diff < 0.05:
            print("⚠️  学生模型表现一般，差距在 2%-5% 之间")
        else:
            print("❌ 学生模型表现较差，差距超过 5%，建议重新蒸馏")

        # 模型大小对比
        teacher_size = os.path.getsize("models/final_model.h5") / (1024 * 1024)
        student_size = os.path.getsize(student_path) / (1024 * 1024)
        print()
        print("模型大小对比:")
        print(f"  教师模型: {teacher_size:.2f} MB")
        print(f"  学生模型: {student_size:.2f} MB")
        print(f"  压缩比: {student_size/teacher_size*100:.1f}%")

    print()
    print("=" * 60)
    print("评估完成")
    print("=" * 60)

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    f.close()
    print("结果已保存到 test_result.txt")
