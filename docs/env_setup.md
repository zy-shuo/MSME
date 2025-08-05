# 环境变量配置指南

本项目使用`.env`文件来管理API密钥和其他敏感配置信息。这样可以避免将敏感信息直接硬编码到代码中或暴露在命令行参数中。

## 设置步骤

1. 在项目根目录下创建一个名为`.env`的文件（注意文件名前面有一个点）。

2. 复制`env.example`文件的内容到`.env`文件中：
   ```bash
   cp env.example .env
   ```

3. 编辑`.env`文件，填入您的实际API密钥和其他配置信息：
   ```
   # OpenAI API配置
   OPENAI_API_KEY=your_actual_api_key_here
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

4. 如果您使用的是Azure OpenAI服务，请相应地修改配置：
   ```
   OPENAI_API_KEY=your_azure_api_key
   OPENAI_API_BASE=https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name
   OPENAI_API_VERSION=2023-05-15
   ```

## 注意事项

- `.env`文件包含敏感信息，已被添加到`.gitignore`中，不会被提交到版本控制系统。
- 确保不要意外地将您的API密钥提交到公共仓库中。
- 如果您在团队中工作，每个团队成员都需要创建自己的`.env`文件。

## 使用方法

一旦设置好`.env`文件，程序会自动从中读取配置信息，您无需在命令行中指定API密钥和基础URL：

```bash
# 运行专家推理过程
python process_dataset_with_experts.py --dataset your_dataset.xlsx --output results.xlsx
```

如果您仍然想在命令行中覆盖这些设置，可以使用相应的参数：

```bash
python process_dataset_with_experts.py --dataset your_dataset.xlsx --output results.xlsx --api_key "your_api_key" --base_url "your_base_url"
```

## 依赖

本项目使用`python-dotenv`库来加载`.env`文件中的环境变量。确保已安装此依赖：

```bash
pip install python-dotenv
```

或者通过项目的`requirements.txt`安装所有依赖：

```bash
pip install -r requirements.txt
```