# Claude Code -> GPT OSS Fine-tuned Mentoring Model

A comprehensive GPU-based fine-tuning pipeline for creating Claude-style mentoring AI models using GPT-2 architecture with advanced logging and evaluation capabilities.
The goal of this repo is to tune the open source GPT-OSS (20 GB) model using data collected from my Claude Code sessions in order to have a local LLM to run on my 32 GB MacBook Pro 
with hopefully similar capabilities. This will be a work in progress and require frequent fine tuning.
 
## 📊 Training Results

- **Model**: GPT-2 → Claude Mentor Fine-tuned
- **Loss Reduction**: 4.08 → 3.03 (26% improvement)
- **Training Efficiency**: ~8 seconds on RTX 4090
- **Dataset**: 40 high-quality mentoring conversations
- **Model Quality**: 83% responses rated "Excellent" (≥0.8 quality score)

## 🏗️ Architecture

### Components

```
claude-mentoring-finetune/
├── scripts/                    # Fine-tuning scripts
│   ├── auto_setup_and_run.py   # Smart training with API handling
│   ├── enhanced_gpu_finetune.py # Advanced GPU training
│   ├── optimized_gpu_finetune.py # Optimized parameters
│   └── robust_finetune.py       # Conservative training approach
├── evaluation/                 # Model testing and evaluation
│   ├── test_final_model.py      # Comprehensive model evaluation
│   └── langsmith_evaluation.py # LangSmith integration
├── config/                     # Configuration and environment
│   ├── .env.template           # API key template
│   └── update_api_keys.py      # Configuration utility
├── data/                       # Training data
│   └── claude_mentoring_dataset.jsonl # Training conversations
└── models/                     # Trained model outputs
    └── claude_mentor_final/    # Production model
```

## 🔧 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/claude-mentoring-finetune.git
cd claude-mentoring-finetune

# Copy environment template
cp config/.env.template .env

# Edit .env with your API keys
nano .env
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ML libraries
pip install transformers datasets accelerate wandb langsmith

# Or use requirements.txt
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Interactive configuration
python config/update_api_keys.py --interactive

# Or edit manually
nano .env
```

### 4. Run Training

```bash
# Automatic setup with smart fallbacks
python scripts/auto_setup_and_run.py

# Or optimized GPU training
python scripts/optimized_gpu_finetune.py
```

### 5. Evaluate Model

```bash
# Comprehensive evaluation
python evaluation/test_final_model.py

# LangSmith evaluation (requires API key)
python evaluation/langsmith_evaluation.py
```
