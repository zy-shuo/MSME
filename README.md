# Multi-stage Multi-expert Stance Detection Framework

This project implements a multi-stage multi-expert stance detection framework for analyzing the stance of text towards specific targets.

## Framework Structure

The framework is divided into three main stages:

1. **Knowledge Preparation**
   - Web search based on target keywords
   - Extract and clean relevant background knowledge
   - Text chunking and deduplication
   - Select the most relevant knowledge chunks

2. **Expert Reasoning**
   - Multi-expert collaborative analysis
   - Knowledge Expert refines knowledge and performs reasoning
   - Label Expert constructs fine-grained stance label system
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
│   ├── __init__.py
│   ├── retriever.py           # Web retrieval module
│   ├── text_processor.py      # Text processing module
│   ├── knowledge_selector.py  # Knowledge selection module
│   └── stance_label_generator.py  # Stance label generation module
├── expert_reasoning/          # Expert reasoning stage code
│   ├── __init__.py
│   ├── knowledge_expert.py    # Knowledge expert module
│   ├── label_expert.py        # Label expert module
│   ├── pragmatic_expert.py    # Pragmatic expert module
│   ├── expert_coordinator.py  # Expert coordinator module
│   └── meta_judge.py          # Meta-judge module
├── embedding_models/          # Embedding model directory
│   ├── bge-base-zh-v1.5/      # Chinese embedding model
│   └── bge-base-en-v1.5/      # English embedding model
├── examples/                  # Example code
│   └── example.py             # Complete usage example
├── outputs/                   # Output directory (stores result files)
├── run_full_pipeline.py       # Complete pipeline main script (recommended)
├── run_batch.sh               # Batch processing script
├── requirements.txt           # Project dependencies
├── README_ZH.md               # Chinese documentation
└── README_EN.md               # English documentation
```

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd MSME_StanceDetection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API keys:

Create a `.env` file in the project root directory:

```bash
# .env
SERPAPI_API_KEY=your_serpapi_key_here
OPENAI_API_KEY=your_openai_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

4. Download embedding models (optional, if models are not pre-downloaded, the program will automatically download from Hugging Face):

```bash
# The program will automatically download models to the embedding_models/ directory
# For manual download, visit:
# Chinese model: https://huggingface.co/BAAI/bge-base-zh-v1.5
# English model: https://huggingface.co/BAAI/bge-base-en-v1.5
```

### Dataset Format

Input datasets should be in Excel format (.xlsx) and contain the following required columns:
- `target`: Stance target
- `text`: Text to be analyzed
- `label`: Ground truth label (optional, for evaluation)

Optional columns (if knowledge is already available, knowledge preparation stage can be skipped):
- `raw_knowledge`: Raw background knowledge
- `ESL`: Explicit Stance Labels

## Usage

### 1. Set API Keys

Create a `.env` file in the project root directory and configure API keys:

```bash
# .env file content
# SerpAPI key (for web search)
SERPAPI_API_KEY=your_serpapi_key_here

# OpenAI API key (for LLM calls)
OPENAI_API_KEY=your_openai_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

**Note**:
- Replace `your_serpapi_key_here` and `your_openai_key_here` with your actual API keys
- The `.env` file should not be committed to version control (already configured in `.gitignore`)
- The program will automatically read configuration from the `.env` file

### 2. Run Complete Pipeline

Use `run_full_pipeline.py` to run the complete stance detection pipeline (including both knowledge preparation and expert reasoning stages):

```bash
# Process complete dataset
python run_full_pipeline.py --dataset data/SEM16.xlsx --output_dir outputs

# Process specified range of data (e.g., process first 100 entries)
python run_full_pipeline.py --dataset data/SEM16.xlsx --output_dir outputs --start 0 --end 100


# Customize knowledge retrieval parameters
python run_full_pipeline.py \
  --dataset data/P-Stance.xlsx \
  --output_dir outputs \
  --num_search_results 5 \
  --top_k_chunks 5
```

**Parameter Description:**
- `--dataset`: Dataset file path (required)
- `--output_dir`: Output directory, default is `outputs`
- `--start`: Start index (inclusive), default is 0
- `--end`: End index (exclusive), default is None (process to the end)
- `--num_search_results`: Number of web pages retrieved per target, default is 3
- `--top_k_chunks`: Number of knowledge chunks to select, default is 3
- `--skip_knowledge_preparation`: Skip knowledge preparation stage (if dataset already contains knowledge)

**Output Files:**
- `{dataset_name}_with_knowledge.xlsx`: Dataset with knowledge
- `{dataset_name}_results.xlsx`: Stance detection results (Excel format)
- `{dataset_name}_results.json`: Stance detection results (JSON format, with detailed information)

## Workflow

`run_full_pipeline.py` implements the complete two-stage stance detection pipeline:

### Stage 0: Knowledge Preparation

For each unique target in the dataset, perform the following steps:

1. **Web Search**: Use SerpAPI to retrieve web pages related to the target
2. **Text Chunking**: Split retrieved text into fixed-size chunks
3. **Embedding Computation**: Compute vector representations of text chunks using BGE model
4. **Knowledge Selection**: Select Top-K knowledge chunks most relevant to the target
5. **Label Generation**: Use LLM to generate Explicit Stance Labels (ESL)

### Stage 1: Expert Reasoning

For each text in the dataset, perform the following steps:

1. **Knowledge Expert**: Refine background knowledge and perform preliminary stance reasoning
2. **Label Expert**: Conduct fine-grained analysis based on explicit stance labels
3. **Pragmatic Expert**: Analyze rhetorical devices and logical relationships
4. **Meta-Judge**: Synthesize opinions from three experts and make final stance judgment

## Configuration Parameters

### Knowledge Preparation Stage Configuration

#### Search API Configuration
- **Search Engine**: Google (via SerpAPI)
- **Number of Search Results**: 3 web pages/target (adjustable via `--num_search_results`)
- **Time Window**: Unlimited (get most relevant results)
- **Request Timeout**: 10 seconds

#### Text Processing Configuration
- **Chunk Size**: 300 tokens (approximately 300 Chinese characters, 1200 English characters)
- **Chunk Overlap**: 30 tokens (approximately 30 Chinese characters, 120 English characters)
- **Breakpoint Strategy**: Prioritize breaking at sentence-ending punctuation (`[.!?]\s+`)
- **Character Encoding**: UTF-8

#### Embedding Model Configuration
- **Chinese Model**: BAAI/bge-base-zh-v1.5
- **English Model**: BAAI/bge-base-en-v1.5
- **Vector Dimension**: 768 dimensions
- **Language Detection**: Based on Unicode character range (`\u4e00-\u9fff` for Chinese)

#### Deduplication and Selection Configuration
- **Similarity Threshold**: 0.85 (above this value is considered duplicate)
- **Similarity Calculation**: Cosine similarity
- **Deduplication Strategy**: Greedy algorithm
- **Knowledge Selection Count**: Top-3 most relevant fragments (adjustable via `--top_k_chunks`)

#### Stance Label Generation Configuration
- **Language Model**: Called via OpenAI API
- **Generation Temperature**: 0.0
- **Maximum Output**: 2048 tokens
- **Retry Count**: 3 times


### 3. Run Example

View complete usage example:

```bash
# Run example code (demonstrates multi-expert stance detection workflow)
python examples/example.py
```



## Complete Example

Here is a complete example of using `run_full_pipeline.py` to process the SemEval 2016 dataset:

```bash
# Step 1: Create .env file and configure API keys
# Create .env file in project root directory with the following content:
# SERPAPI_API_KEY=your_serpapi_key_here
# OPENAI_API_KEY=your_openai_key_here
# OPENAI_API_BASE=https://api.openai.com/v1

# Step 2: Run complete pipeline (process first 50 entries)
python run_full_pipeline.py \
  --dataset data/SEM16.xlsx \
  --output_dir outputs \
  --start 0 \
  --end 50 \
  --num_search_results 3 \
  --top_k_chunks 3

# Step 3: View results
# Results will be saved in the outputs/ directory:
# - SEM16_with_knowledge.xlsx: Dataset with knowledge
# - SEM16_results.xlsx: Stance detection results (Excel format)
# - SEM16_results.json: Stance detection results (JSON format, with detailed information)
```

## Output Results Description

### Excel Result File (`{dataset_name}_results.xlsx`)

Contains the following columns:
- `index`: Data index
- `target`: Stance target
- `text`: Original text
- `label`: Ground truth label (if available in dataset)
- `raw_knowledge`: Raw background knowledge
- `ESL`: Explicit Stance Labels
- `refined_knowledge`: Knowledge refined by Knowledge Expert
- `knowledge_expert_response`: Complete response from Knowledge Expert
- `knowledge_expert_stance`: Stance judgment from Knowledge Expert
- `label_expert_response`: Complete response from Label Expert
- `label_expert_stance`: Stance judgment from Label Expert
- `pragmatic_expert_response`: Complete response from Pragmatic Expert
- `pragmatic_expert_stance`: Stance judgment from Pragmatic Expert
- `meta_judge_response`: Complete response from Meta-Judge
- `final_stance`: Final stance judgment (FAVOR/AGAINST/NONE)

### JSON Result File (`{dataset_name}_results.json`)

Contains the same information as the Excel file, but in a format more suitable for programmatic processing. Each record is a JSON object containing the detailed analysis process from all experts.

## Notes

1. **API Call Cost**: This framework makes multiple LLM API calls (approximately 4-5 times per text), please be aware of API usage costs
2. **Processing Time**: The complete pipeline includes web search and multiple LLM calls, processing speed depends on network conditions and API response speed
3. **Checkpoint Resumption**: The program supports checkpoint resumption; if interrupted midway, rerunning the same command will skip already processed data


## Environment Requirements

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


