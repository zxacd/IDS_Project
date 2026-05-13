# 基于深度学习的网络入侵检测系统（IDS）

> 结合 CNN、知识蒸馏（Knowledge Distillation）与模型压缩技术的轻量化网络入侵检测系统

---

## 项目简介

本项目基于 CIC-IDS2017 数据集，采用卷积神经网络（CNN）作为教师模型，通过知识蒸馏训练轻量化学生模型，并结合量化与剪枝技术，在保持检测精度的同时显著降低模型大小和推理延迟，最终部署为 Flask REST API 服务。

---

## 项目结构

```
IDS_Project/
├── data/                     # 数据目录（数据集需自行下载，见下方说明）
│   ├── raw/                  # 原始 CSV 数据集
│   └── processed/            # 预处理后的 .npy 特征文件
├── models/                   # 训练后的模型文件（运行训练脚本后生成）
├── results/                  # 实验结果图表与报告（训练后生成）
├── logs/                     # TensorBoard 日志
│
├── preprocess.py             # 数据预处理（特征工程、归一化、标签编码）
├── train_cnn.py              # 教师模型（CNN）训练
├── distill_only.py           # 知识蒸馏训练（学生模型）
├── distill_optimized.py      # 优化版知识蒸馏（含超参搜索）
├── lightweight.py            # 轻量化模型定义
├── quantize_only.py          # 模型量化（TFLite）
├── quantize_dynamic.py       # 动态量化
│
├── baseline_models.py        # 基线对比模型（SVM, RF, CNN-LSTM）
├── fix_baseline.py           # 基线模型修复脚本
├── fix_baseline2.py          # 基线模型修复脚本 v2
├── fix_model_loading.py      # 模型加载修复
│
├── experiment_1.py           # 实验一：CNN 入侵检测性能
├── experiment_2.py           # 实验二：知识蒸馏效果对比
├── experiment_3.py           # 实验三：模型压缩效果
├── experiment_4.py           # 实验四：与其他方法对比
├── run_all_experiments.py    # 一键运行所有实验
├── run_experiments.py        # 实验批量运行脚本
│
├── eval_full.py              # 完整模型评估
├── benchmark.py              # 推理基准测试
├── test_inference_latency.py # 推理延迟测试
├── compare_accuracy.py       # 精度对比
├── compare_experiments.py    # 实验结果对比
├── analyze_rare.py           # 稀有类别分析
│
├── api_server.py             # Flask REST API 部署服务
├── quick_test.py             # 快速功能测试
├── test_run.py               # 运行测试
├── check_classes.py          # 类别检查工具
├── check_syntax.py           # 代码语法检查
├── read_docx.py              # 文档读取工具
├── read_docx2.py             # 文档读取工具 v2
│
├── requirements.txt          # Python 依赖库
├── run_test.bat              # Windows 一键测试脚本
├── 测试环境.bat              # 环境测试脚本
└── 实验方案.md               # 实验设计方案
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

从 [CIC-IDS2017 官网](https://www.unb.ca/cic/datasets/ids-2017.html) 下载数据集，将 CSV 文件放入 `data/raw/` 目录。

### 3. 数据预处理

```bash
python preprocess.py
```

### 4. 训练模型

```bash
# 训练教师模型（CNN）
python train_cnn.py

# 训练学生模型（知识蒸馏）
python distill_only.py
```

### 5. 运行所有实验

```bash
python run_all_experiments.py
```

### 6. 启动 API 服务

```bash
python api_server.py
```

---

## 依赖环境

| 包名 | 版本 |
|------|------|
| tensorflow | 2.15.0 |
| scikit-learn | 1.3.2 |
| numpy | 1.26.4 |
| pandas | 2.1.4 |
| flask | 2.3.3 |
| flask-cors | 4.0.0 |

完整依赖见 `requirements.txt`。

---

## 主要技术

- **CNN 入侵检测**：将网络流量特征转换为 2D 特征图，使用卷积神经网络进行多分类检测（15 种攻击类型）
- **知识蒸馏**：教师-学生架构，通过软标签 + 硬标签联合训练压缩模型
- **模型量化**：TFLite 动态量化，减少模型体积与推理延迟
- **模型剪枝**：移除冗余权重，进一步轻量化
- **REST API**：Flask 部署，支持实时入侵检测推理

---

## 数据集说明

本项目使用以下数据集（**不包含在仓库中**，需自行下载）：

- **CIC-IDS2017**：[https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **UNSW-NB15**（可选对比）：[https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## License

MIT License
