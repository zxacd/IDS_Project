#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 .docx 提取文本内容（使用zipfile，无需python-docx）
"""
import zipfile
import re
import os

docx_path = 'D:/Users/zxacd/source/毕设/毕业论文初稿.docx'
output_txt = 'D:/Users/zxacd/IDS_Project/论文初稿_提取内容.txt'

with zipfile.ZipFile(docx_path, 'r') as z:
    xml_bytes = z.read('word/document.xml')
    xml_content = xml_bytes.decode('utf-8')

# 提取所有 <w:t> 标签中的文本
text_parts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml_content)
full_text = ''.join(text_parts)

# 按段落分割（根据 </w:p> 标签）
paragraphs = re.split(r'</w:p>', xml_content)
extracted = []
for p in paragraphs:
    texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', p)
    if texts:
        extracted.append(''.join(texts))

with open(output_txt, 'w', encoding='utf-8') as f:
    for i, para in enumerate(extracted):
        f.write(f'[{i}] {para}\n')

print(f'提取完成，共 {len(extracted)} 个段落')
print(f'保存到: {output_txt}')

# 打印前100段预览
print('\n===== 前100段预览 =====')
for i, para in enumerate(extracted[:100]):
    print(f'[{i:03d}] {para}')
