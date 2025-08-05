# Multi-stage Multi-expert Stance Detection Framework

This project implements a multi-stage multi-expert stance detection framework for analyzing stance towards specific targets in text.

## Framework Structure

The framework consists of three main stages:

1. **Knowledge Preparation**
   - Web retrieval based on target keywords
   - Extraction and cleaning of relevant background knowledge
   - Text chunking and deduplication
   - Selection of the most relevant knowledge blocks

2. **Expert Reasoning**
   - Multi-expert collaborative analysis
   - Knowledge Expert refines knowledge and performs reasoning
   - Label Expert constructs fine-grained stance label systems
   - Pragmatic Expert analyzes rhetorical devices and logical relationships

3. **Decision Aggregation**
   - Meta-Judge determines the final stance

## Project Structure

```
├── data/                      # Dataset directory
│   ├── SEM16.xlsx             # SemEval 2016 dataset
│   ├── P-Stance.xlsx          # P-Stance dataset
│   └── Weibo-SD.xlsx          # Weibo stance detection dataset
├── knowledge_preparation/     # Knowledge preparation stage code
│   ├── retriever.py           # Web retrieval module
│   ├── text_processor.py      # Text processing module
│   ├── knowledge_selector.py  # Knowledge selection module
│   └── stance_label_generator.py  # Stance label generation module
├── expert_reasoning/          # Expert reasoning stage code
│   ├── knowledge_expert.py    # Knowledge expert module
│   ├── label_expert.py        # Label expert module
│   ├── pragmatic_expert.py    # Pragmatic expert module
│   └── expert_coordinator.py  # Expert coordinator module
├── embedding_models/          # Embedding models directory
│   ├── bge-base-zh-v1.5/      # Chinese embedding model
│   └── bge-base-en-v1.5/      # English embedding model
├── expert_reasoning/          # Expert reasoning stage code
│   ├── knowledge_expert.py    # Knowledge expert module
│   ├── label_expert.py        # Label expert module
│   ├── pragmatic_expert.py    # Pragmatic expert module
│   └── expert_coordinator.py  # Expert coordinator module
├── embedding_models/          # Embedding models directory
│   ├── bge-base-zh-v1.5/      # Chinese embedding model
│   └── bge-base-en-v1.5/      # English embedding model
├── examples/                  # Example code
│   ├── knowledge_preparation_example.py  # Knowledge preparation stage usage example
│   ├── stance_label_generation_example.py  # Stance label generation example
│   └── full_expert_reasoning_example.py  # Expert reasoning stage usage example
├── run_knowledge_preparation.py  # Knowledge preparation stage main script
├── run_stance_label_generation.py  # Stance label generation main script
├── run_expert_reasoning.py    # Expert reasoning stage main script
├── process_single_target.py   # Complete script for processing a single target
├── quick_process.py           # Simplified script for quick processing of a single target
├── update_dataset_with_knowledge.py  # Script for updating datasets
└── requirements.txt           # Project dependencies
```

## Usage

### Setting up API Keys

First, you need to set up API keys. The recommended method is using a .env file:

1. Copy `env.example` to `.env`:
```bash
cp env.example .env
```

2. Edit the `.env` file and fill in your API keys:
```
# OpenAI API configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# SerpAPI key (for web retrieval)
SERPAPI_API_KEY=your_serpapi_key_here
```

3. The program will automatically read these configurations from the .env file

Alternatively, you can use environment variables:

```bash
# SerpAPI key (for web retrieval)
# Linux/Mac
export SERPAPI_API_KEY="your_serpapi_key_here"
# Windows
set SERPAPI_API_KEY=your_serpapi_key_here

# OpenAI API key (for stance label generation)
# Linux/Mac
export OPENAI_API_KEY="your_openai_key_here"
# Windows
set OPENAI_API_KEY=your_openai_key_here
```

For detailed configuration instructions, please refer to the [Environment Variable Configuration Guide](docs/env_setup.md).

### Running Examples
Run this file to get a complete example.
```bash

# Multi-expert stance decision example
python examples/full_expert_reasoning_example.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download embedding models (optional, if models are not pre-downloaded, the program will automatically download them from Hugging Face):
```bash
# Download Chinese model
mkdir -p embedding_models
cd embedding_models
git lfs install
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5 bge-base-zh-v1.5

# Download English model
git clone https://huggingface.co/BAAI/bge-base-en-v1.5 bge-base-en-v1.5
```

## Requirements

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

## License

MIT