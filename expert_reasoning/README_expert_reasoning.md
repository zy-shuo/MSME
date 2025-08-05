# Expert Reasoning 阶段使用指南

本文档介绍如何使用多专家立场检测框架中的Expert Reasoning阶段。

## 专家角色介绍

Expert Reasoning阶段包含四个专家角色：

1. **Knowledge Expert (知识专家)**
   - 负责从原始知识中提炼出相关知识
   - 基于提炼的知识进行推理，判断立场
   - 生成refined_knowledge供其他专家使用

2. **Label Expert (标签专家)**
   - 构建细粒度立场标签体系
   - 从标签的角度推理立场
   - 使用知识专家提供的refined_knowledge

3. **Pragmatic Expert (语用专家)**
   - 分析评论中的修辞手法
   - 从语用角度分析评论
   - 使用知识专家提供的refined_knowledge

4. **Meta Judge (元判断专家)**
   - 综合三个专家的分析结果
   - 提取有用信息并进行自己的分析
   - 做出最终的立场判断

## 使用方法

### 1. 处理单个数据集

使用`process_dataset_with_experts.py`脚本处理整个数据集：

```bash
python process_dataset_with_experts.py --dataset data/your_dataset.xlsx --output results/output_with_experts.xlsx
```

参数说明：
- `--dataset`: 输入数据集路径（必须包含target, text, raw_knowledge, ESL字段）
- `--output`: 输出结果文件路径
- `--model`: OpenAI模型名称（默认：gpt-3.5-turbo）
- `--api_key`: OpenAI API密钥（可选，也可通过.env文件或环境变量设置）
- `--base_url`: OpenAI API基础URL（可选，也可通过.env文件或环境变量设置）
- `--batch_size`: 中间结果保存批次大小（默认：10）
- `--no_intermediate`: 禁用中间结果保存

### 2. 运行示例

运行示例脚本，了解专家推理过程：

```bash
python examples/full_expert_reasoning_example.py
```

此示例会：
1. 创建一个小型示例数据集
2. 对每条数据运行四个专家的分析（知识专家、标签专家、语用专家和元判断专家）
3. 提取并显示分析结果
4. 保存结果到Excel文件

### 3. 直接使用专家协调器

在自定义代码中使用专家协调器：

```python
from expert_reasoning.expert_coordinator import coordinate_experts

# 示例数据
target = "Atheism"
raw_knowledge = "Atheism, in the broadest sense, is an absence of belief in the existence of deities. Less broadly, atheism is a rejection of the belief that any deities exist. In an even narrower sense, atheism is specifically the position that there are no deities. Atheism is contrasted with theism, which is the belief that at least one deity exists."
esl = "A. Favor: Support atheism as a valid worldview.\nB. Against: Oppose atheism and advocate for theism.\nC. Neutral/None"
text = "He who exalts himself shall be humbled; and he who humbles himself shall be exalted.Matt 23:12."

# 运行专家协调器
results = coordinate_experts(
    target=target,
    raw_knowledge=raw_knowledge,
    esl=esl,
    text=text
)

# 获取结果
refined_knowledge = results["refined_knowledge"]
knowledge_expert_response = results["knowledge_expert"]["response"]
label_expert_response = results["label_expert"]["response"]
pragmatic_expert_response = results["pragmatic_expert"]["response"]
meta_judge_response = results["meta_judge"]["response"]

# 注意：所有专家的温度参数均设置为0，确保输出稳定
```

### 4. 单独使用各专家

也可以单独使用各个专家：

```python
from expert_reasoning.knowledge_expert import analyze_knowledge
from expert_reasoning.label_expert import analyze_labels
from expert_reasoning.pragmatic_expert import analyze_pragmatics

# 知识专家
refined_knowledge, knowledge_response = analyze_knowledge(
    target=target,
    raw_knowledge=raw_knowledge,
    esl=esl,
    text=text
)

# 标签专家
label_response = analyze_labels(
    target=target,
    refined_knowledge=refined_knowledge,
    esl=esl,
    text=text
)

# 语用专家
pragmatic_response = analyze_pragmatics(
    target=target,
    refined_knowledge=refined_knowledge,
    esl=esl,
    text=text
)
```

## 输出结果说明

处理后的数据集将包含以下新字段：

1. **refined_knowledge**: 知识专家提炼的知识
2. **knowledge_expert_response**: 知识专家的完整响应
3. **label_expert_response**: 标签专家的完整响应
4. **pragmatic_expert_response**: 语用专家的完整响应
5. **meta_judge_response**: 元判断专家的完整响应（最终立场判断）

## 注意事项

1. 确保输入数据集包含必要的字段：target, text, raw_knowledge, ESL
2. 设置OpenAI API密钥（推荐通过.env文件设置，也可通过环境变量或直接传入）
3. 对于大型数据集，建议设置适当的batch_size以定期保存中间结果
4. 处理过程可能需要较长时间，取决于数据集大小和API响应速度
5. 所有专家模型的温度参数均设置为0，确保输出结果的稳定性和一致性