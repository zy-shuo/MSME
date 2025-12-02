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
│   ├── __init__.py
│   ├── retriever.py           # 网络检索模块
│   ├── text_processor.py      # 文本处理模块
│   ├── knowledge_selector.py  # 知识选择模块
│   └── stance_label_generator.py  # 立场标签生成模块
├── expert_reasoning/          # 专家推理阶段代码
│   ├── __init__.py
│   ├── knowledge_expert.py    # 知识专家模块
│   ├── label_expert.py        # 标签专家模块
│   ├── pragmatic_expert.py    # 语用专家模块
│   ├── expert_coordinator.py  # 专家协调器模块
│   └── meta_judge.py          # 元判断器模块
├── embedding_models/          # 嵌入模型目录
│   ├── bge-base-zh-v1.5/      # 中文嵌入模型
│   └── bge-base-en-v1.5/      # 英文嵌入模型
├── examples/                  # 示例代码
│   └── example.py             # 完整的使用示例
├── outputs/                   # 输出目录（存放结果文件）
├── run_full_pipeline.py       # 完整流程主脚本（推荐使用）
├── run_batch.sh               # 批处理脚本
├── requirements.txt           # 项目依赖
├── README_ZH.md               # 中文说明文档
└── README_EN.md               # 英文说明文档
```

## 快速开始

### 安装

1. 克隆仓库：

```bash
git clone <repository_url>
cd MSME_StanceDetection
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置 API 密钥：

在项目根目录创建 `.env` 文件：

```bash
# .env
SERPAPI_API_KEY=your_serpapi_key_here
OPENAI_API_KEY=your_openai_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

4. 下载嵌入模型（可选，如果没有预下载模型，程序会自动从Hugging Face下载）：

```bash
# 程序会自动下载模型到 embedding_models/ 目录
# 如需手动下载，可访问：
# 中文模型: https://huggingface.co/BAAI/bge-base-zh-v1.5
# 英文模型: https://huggingface.co/BAAI/bge-base-en-v1.5
```

### 数据集格式

输入数据集应为 Excel 格式 (.xlsx)，包含以下必需列：
- `target`: 立场目标
- `text`: 待分析的文本
- `label`: 真实标签（可选，用于评估）

可选列（如果已有知识，可跳过知识准备阶段）：
- `raw_knowledge`: 原始背景知识
- `ESL`: 显式立场标签

## 使用方法

### 1. 设置API密钥

在项目根目录创建 `.env` 文件，配置 API 密钥：

```bash
# .env 文件内容
# SerpAPI密钥（用于网络检索）
SERPAPI_API_KEY=your_serpapi_key_here

# OpenAI API密钥（用于LLM调用）
OPENAI_API_KEY=your_openai_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

**注意**：
- 请将 `your_serpapi_key_here` 和 `your_openai_key_here` 替换为您的实际 API 密钥
- `.env` 文件不应提交到版本控制系统（已在 `.gitignore` 中配置）
- 程序会自动从 `.env` 文件中读取配置

### 2. 运行完整流程

使用 `run_full_pipeline.py` 运行完整的立场检测流程（包括知识准备和专家推理两个阶段）：

```bash
# 处理完整数据集
python run_full_pipeline.py --dataset data/SEM16.xlsx --output_dir outputs

# 处理指定范围的数据（例如：处理前100条）
python run_full_pipeline.py --dataset data/SEM16.xlsx --output_dir outputs --start 0 --end 100


# 自定义知识检索参数
python run_full_pipeline.py \
  --dataset data/P-Stance.xlsx \
  --output_dir outputs \
  --num_search_results 5 \
  --top_k_chunks 5
```

**参数说明：**
- `--dataset`: 数据集文件路径（必需）
- `--output_dir`: 输出目录，默认为 `outputs`
- `--start`: 起始索引（包含），默认为 0
- `--end`: 结束索引（不包含），默认为 None（处理到末尾）
- `--num_search_results`: 每个目标检索的网页数量，默认为 3
- `--top_k_chunks`: 选择的知识块数量，默认为 3
- `--skip_knowledge_preparation`: 跳过知识准备阶段（如果数据集已包含知识）

**输出文件：**
- `{dataset_name}_with_knowledge.xlsx`: 包含知识的数据集
- `{dataset_name}_results.xlsx`: 立场检测结果（Excel格式）
- `{dataset_name}_results.json`: 立场检测结果（JSON格式，包含详细信息）

## 工作流程

`run_full_pipeline.py` 实现了完整的两阶段立场检测流程：

### 阶段 0: 知识准备 (Knowledge Preparation)

对数据集中的每个唯一目标执行以下步骤：

1. **网络检索**: 使用 SerpAPI 检索与目标相关的网页
2. **文本分块**: 将检索到的文本分割成固定大小的块
3. **嵌入计算**: 使用 BGE 模型计算文本块的向量表示
4. **知识选择**: 选择与目标最相关的 Top-K 知识块
5. **标签生成**: 使用 LLM 生成显式立场标签 (ESL)

### 阶段 1: 专家推理 (Expert Reasoning)

对数据集中的每条文本执行以下步骤：

1. **知识专家**: 提炼背景知识并进行初步立场推理
2. **标签专家**: 基于显式立场标签进行细粒度分析
3. **语用专家**: 分析修辞手法和逻辑关系
4. **元判断器**: 综合三位专家的意见，做出最终立场判断

## 配置参数

### 知识准备阶段配置

#### 搜索API配置
- **搜索引擎**: Google (通过SerpAPI)
- **搜索结果数量**: 3个网页/目标（可通过 `--num_search_results` 调整）
- **时间窗口**: 无限制（获取最相关结果）
- **请求超时**: 10秒

#### 文本处理配置
- **分块大小**: 300 tokens (约300个中文字符，1200个英文字符)
- **块间重叠**: 30 tokens (约30个中文字符，120个英文字符)
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
- **知识选择数量**: Top-3 最相关片段（可通过 `--top_k_chunks` 调整）

#### 立场标签生成配置
- **语言模型**: 通过 OpenAI API 调用
- **生成温度**: 0.0
- **最大输出**: 2048 tokens
- **重试次数**: 3次


### 3. 运行示例

查看完整的使用示例：

```bash
# 运行示例代码（演示多专家立场检测流程）
python examples/example.py
```

### 4. 批量处理

使用批处理脚本处理多个数据集：

```bash
# Linux/Mac
bash run_batch.sh

# Windows
# 请手动运行 run_full_pipeline.py 多次，或创建对应的 .bat 脚本
```

## 完整示例

以下是使用 `run_full_pipeline.py` 处理 SemEval 2016 数据集的完整示例：

```bash
# 步骤 1: 创建 .env 文件并配置 API 密钥
# 在项目根目录创建 .env 文件，内容如下：
# SERPAPI_API_KEY=your_serpapi_key_here
# OPENAI_API_KEY=your_openai_key_here
# OPENAI_API_BASE=https://api.openai.com/v1

# 步骤 2: 运行完整流程（处理前50条数据）
python run_full_pipeline.py \
  --dataset data/SEM16.xlsx \
  --output_dir outputs \
  --start 0 \
  --end 50 \
  --num_search_results 3 \
  --top_k_chunks 3

# 步骤 3: 查看结果
# 结果将保存在 outputs/ 目录下：
# - SEM16_with_knowledge.xlsx: 包含知识的数据集
# - SEM16_results.xlsx: 立场检测结果（Excel格式）
# - SEM16_results.json: 立场检测结果（JSON格式，包含详细信息）
```

**程序执行流程：**

1. **阶段 0 - 知识准备**：
   - 自动识别数据集中的唯一目标（如 "Hillary Clinton", "Donald Trump" 等）
   - 对每个目标进行网络检索，获取相关背景知识
   - 生成显式立场标签 (ESL)
   - 将知识保存到数据集中

2. **阶段 1 - 专家推理**：
   - 对每条文本调用三位专家进行分析
   - 知识专家提炼背景知识并推理
   - 标签专家基于 ESL 进行细粒度分析
   - 语用专家分析修辞手法
   - 元判断器综合三位专家意见，给出最终立场

3. **结果保存**：
   - 每处理 10 条数据自动保存一次
   - 支持断点续传（如果程序中断，可以从上次停止的地方继续）

## 输出结果说明

### Excel 结果文件 (`{dataset_name}_results.xlsx`)

包含以下列：
- `index`: 数据索引
- `target`: 立场目标
- `text`: 原始文本
- `label`: 真实标签（如果数据集中有）
- `raw_knowledge`: 原始背景知识
- `ESL`: 显式立场标签
- `refined_knowledge`: 知识专家提炼的知识
- `knowledge_expert_response`: 知识专家的完整响应
- `knowledge_expert_stance`: 知识专家的立场判断
- `label_expert_response`: 标签专家的完整响应
- `label_expert_stance`: 标签专家的立场判断
- `pragmatic_expert_response`: 语用专家的完整响应
- `pragmatic_expert_stance`: 语用专家的立场判断
- `meta_judge_response`: 元判断器的完整响应
- `final_stance`: 最终立场判断（FAVOR/AGAINST/NONE）

### JSON 结果文件 (`{dataset_name}_results.json`)

包含与 Excel 文件相同的信息，但格式更适合程序化处理。每条记录是一个 JSON 对象，包含所有专家的详细分析过程。

## 注意事项

1. **API 调用成本**：该框架会调用多次 LLM API（每条文本约 4-5 次），请注意 API 使用成本
2. **处理时间**：完整流程包括网络检索和多次 LLM 调用，处理速度取决于网络状况和 API 响应速度
3. **断点续传**：程序支持断点续传，如果中途中断，重新运行相同命令会跳过已处理的数据


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