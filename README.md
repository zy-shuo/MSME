# 多阶段多专家立场检测框架 (Multi-stage Multi-expert Stance Detection Framework)

本项目实现了一个多阶段多专家的立场检测框架，用于分析文本中对特定目标的立场。

## 框架结构

该框架分为三个主要阶段：

1. **知识准备阶段 (Knowledge Preparation)**
   - 基于目标关键词进行网络检索
   - 提取并清洗相关背景知识
   - 文本分块与去重
   - 筛选最相关的知识块

2. **专家推理阶段 (Expert Reasoning)**
   - 多专家协同分析
   - 知识专家（Knowledge Expert）提炼知识并进行推理
   - 标签专家（Label Expert）构建细粒度立场标签体系
   - 语用专家（Pragmatic Expert）分析修辞手法和逻辑关系

3. **结果整合阶段 (Decision Aggregation)**
   - 决策者（Meta-Judge）判断最终立场

## 项目结构

```
├── data/                      # 数据集目录
│   ├── SEM16.xlsx             # SemEval 2016 数据集
│   ├── P-Stance.xlsx          # P-Stance 数据集
│   └── Weibo-SD.xlsx          # 微博立场检测数据集
├── knowledge_preparation/     # 知识准备阶段代码
│   ├── retriever.py           # 网络检索模块
│   ├── text_processor.py      # 文本处理模块
│   ├── knowledge_selector.py  # 知识选择模块
│   └── stance_label_generator.py  # 立场标签生成模块
├── expert_reasoning/          # 专家推理阶段代码
│   ├── knowledge_expert.py    # 知识专家模块
│   ├── label_expert.py        # 标签专家模块
│   ├── pragmatic_expert.py    # 语用专家模块
│   └── expert_coordinator.py  # 专家协调器模块
├── embedding_models/          # 嵌入模型目录
│   ├── bge-base-zh-v1.5/      # 中文嵌入模型
│   └── bge-base-en-v1.5/      # 英文嵌入模型
├── expert_reasoning/          # 专家推理阶段代码
│   ├── knowledge_expert.py    # 知识专家模块
│   ├── label_expert.py        # 标签专家模块
│   ├── pragmatic_expert.py    # 语用专家模块
│   └── expert_coordinator.py  # 专家协调器模块
├── embedding_models/          # 嵌入模型目录
│   ├── bge-base-zh-v1.5/      # 中文嵌入模型
│   └── bge-base-en-v1.5/      # 英文嵌入模型
├── examples/                  # 示例代码
│   ├── knowledge_preparation_example.py  # 知识准备阶段使用示例
│   ├── stance_label_generation_example.py  # 立场标签生成示例
│   └── full_expert_reasoning_example.py  # 专家推理阶段使用示例
├── run_knowledge_preparation.py  # 知识准备阶段主脚本
├── run_stance_label_generation.py  # 立场标签生成主脚本
├── run_expert_reasoning.py    # 专家推理阶段主脚本
├── process_single_target.py   # 处理单个目标的完整脚本
├── quick_process.py           # 快速处理单个目标的简化脚本
├── update_dataset_with_knowledge.py  # 更新数据集的脚本
└── requirements.txt           # 项目依赖
```

## 使用方法

### 设置API密钥

首先需要设置API密钥，推荐使用.env文件方式：

1. 复制`env.example`为`.env`：
```bash
cp env.example .env
```

2. 编辑`.env`文件，填入您的API密钥：
```
# OpenAI API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# SerpAPI密钥（用于网络检索）
SERPAPI_API_KEY=your_serpapi_key_here
```

3. 程序会自动从.env文件中读取这些配置

也可以使用环境变量方式：

```bash
# SerpAPI密钥（用于网络检索）
# Linux/Mac
export SERPAPI_API_KEY="your_serpapi_key_here"
# Windows
set SERPAPI_API_KEY=your_serpapi_key_here

# OpenAI API密钥（用于立场标签生成）
# Linux/Mac
export OPENAI_API_KEY="your_openai_key_here"
# Windows
set OPENAI_API_KEY=your_openai_key_here
```

详细配置说明请参考 [环境变量配置指南](docs/env_setup.md)。

### 运行示例
运行该文件以获取一个完整的示例。
```bash

# 多专家立场决策示例
python examples/full_expert_reasoning_example.py
```

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd <repository_directory>
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载嵌入模型（可选，如果没有预下载模型，程序会自动从Hugging Face下载）：
```bash
# 下载中文模型
mkdir -p embedding_models
cd embedding_models
git lfs install
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5 bge-base-zh-v1.5

# 下载英文模型
git clone https://huggingface.co/BAAI/bge-base-en-v1.5 bge-base-en-v1.5
```

## 环境要求

- Python 3.8+
- pandas
- openpyxl
- sentence-transformers
- requests
- beautifulsoup4
- google-search-results (SerpAPI)
- openai
- numpy
- tqdm

## 许可证

MIT