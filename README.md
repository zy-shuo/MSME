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

## 配置参数

### 知识准备阶段配置

知识准备阶段包含以下关键配置参数：

#### 搜索API配置
- **搜索引擎**: Google (通过SerpAPI)
- **搜索结果数量**: 3个网页/目标
- **时间窗口**: 无限制（获取最相关结果）
- **请求超时**: 10秒

#### 文本处理配置
- **分块大小**: 200 tokens (约1000字符)
- **块间重叠**: 20 tokens (约100字符)
- **断点策略**: 优先在句末标点处断开 (`[.!?]\s+`)
- **字符编码**: UTF-8

#### 嵌入模型配置
- **中文模型**: BAAI/bge-base-zh-v1.5
- **英文模型**: BAAI/bge-base-en-v1.5
- **向量维度**: 768维
- **语言检测**: 基于Unicode字符范围 (`\u4e00-\u9fff` 为中文)

#### 去重与选择配置
- **相似度阈值**: 0.85 (超过此值视为重复)
- **相似度计算**: 余弦相似度
- **去重策略**: 贪心算法
- **知识选择数量**: Top-3 最相关片段

#### 立场标签生成配置
- **语言模型**: Your LLM
- **生成温度**: 0.7
- **最大输出**: 300 tokens
- **重试次数**: 3次
- **API调用间隔**: 1秒

### 完整配置字典

```python
knowledge_preparation_config = {
    # 搜索配置
    "search_api": "SerpAPI",
    "search_engine": "google", 
    "num_results": 3,
    "request_timeout": 10,
    
    # 文本处理配置
    "chunk_size": 200,
    "overlap": 20,
    "char_per_token": 5,
    "break_patterns": r'[.!?]\s+',
    
    # 嵌入模型配置
    "embedding_models": {
        "zh": "BAAI/bge-base-zh-v1.5",
        "en": "BAAI/bge-base-en-v1.5"
    },
    "embedding_dimension": 768,
    
    # 去重与选择配置
    "similarity_threshold": 0.85,
    "top_k": 3,
    "similarity_metric": "cosine",
    
    # LLM配置
    "llm_model": "Your LLM",
    "temperature": 0.7,
    "max_tokens": 300,
    "max_retries": 3,
    "api_call_delay": 1
}
```

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
