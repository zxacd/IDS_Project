import sys
import traceback

filename = 'baseline_models.py'
with open(filename, 'r', encoding='utf-8') as f:
    code = f.read()

try:
    compile(code, filename, 'exec')
    print("✅ 语法检查通过")
except SyntaxError as e:
    print(f"❌ 语法错误!")
    print(f"  文件: {e.filename}")
    print(f"  行号: {e.lineno}")
    print(f"  列号: {e.offset}")
    print(f"  错误文本: {e.text}")
    print(f"  消息: {e.msg}")
    # 打印错误行及前后各2行
    lines = code.split('\n')
    start = max(0, e.lineno - 3)
    end = min(len(lines), e.lineno + 2)
    print(f"\n  上下文:")
    for i in range(start, end):
        marker = ">>> " if i == e.lineno - 1 else "    "
        print(f"  {marker}{i+1}: {lines[i]}")
