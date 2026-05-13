import re

with open('baseline_models.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # 修复 header 行
    if "header = f\"{'模型':<28}" in line or "header = f\"{'模型':<28}" in line:
        line = "header = '{:<28}{:>10}{:>10}{:>15}'.format('模型', '准确率', '宏F1', '训练时间(s)')\n"
    # 修复 print(f"...{time_str}...") 行
    elif 'print(f"{name:<28} {acc:>10.4f} {f1:>10.4f} {time_str:>15}")' in line:
        line = "    print('{:<28}{:>10.4f}{:>10.4f}{:>15}'.format(name, acc, f1, time_str))\n"
    # 修复 f.write(f"...{time_str}...") 行（在 with open as f 块内）
    elif "f.write(f\"{name:<28} {acc:>10.4f} {f1:>10.4f} {time_str:>15}\\n\")" in line:
        line = "        f.write('{:<28}{:>10.4f}{:>10.4f}{:>15}\\n'.format(name, acc, f1, time_str))\n"
    new_lines.append(line)

with open('baseline_models.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ 修复完成，开始语法检查...")
with open('baseline_models.py', 'r', encoding='utf-8') as f:
    code = f.read()
try:
    compile(code, 'baseline_models.py', 'exec')
    print("✅ 语法检查通过！")
except SyntaxError as e:
    print(f"❌ 第{e.lineno}行仍有错误: {e.msg}")
    linelist = code.split('\n')
    for j in range(max(0,e.lineno-3), min(len(linelist),e.lineno+2)):
        mark = '>>> ' if j == e.lineno-1 else '    '
        print(f'{mark}{j+1}: {linelist[j]}')
