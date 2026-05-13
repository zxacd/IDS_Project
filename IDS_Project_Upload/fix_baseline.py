import re

with open('baseline_models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: header 行 f-string 语法错误
old = "header = f\"{'模型':<28} {'准确率':>10} {'宏F1':>10} {'训练时间(s)':>15}\""
new = "header = '{:<28}{:>10}{:>10}{:>15}'.format('模型', '准确率', '宏F1', '训练时间(s)')"
content = content.replace(old, new)

# 修复2: print 行里的 f-string 格式符写法
old2 = 'print(f"{name:<28} {acc:>10.4f} {f1:>10.4f} {time_str:>15}")'
new2 = 'print("{:<28}{:>10.4f}{:>10.4f}{:>15}".format(name, acc, f1, time_str))'
content = content.replace(old2, new2)

# 修复3: f.write 行里的 f-string（在 with open as f 块内）
old3 = '        f.write(f"{name:<28} {acc:>10.4f} {f1:>10.4f} {time_str:>15}\\n")'
new3 = '        f.write("{:<28}{:>10.4f}{:>10.4f}{:>15}\\n".format(name, acc, f1, time_str))'
content = content.replace(old3, new3)

with open('baseline_models.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 修复完成，验证语法...")
try:
    compile(content, 'baseline_models.py', 'exec')
    print("✅ 语法检查通过")
except SyntaxError as e:
    print(f"❌ 仍有语法错误: 第{e.lineno}行: {e.msg}")
    lines = content.split('\n')
    for i in range(max(0,e.lineno-2), min(len(lines),e.lineno+1)):
        mark = '>>> ' if i == e.lineno-1 else '    '
        print(f'{mark}{i+1}: {lines[i]}')
