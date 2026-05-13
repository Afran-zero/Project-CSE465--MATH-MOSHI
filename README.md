# Math Moshi: Teaching Moshi to Handle Complex Math Questions

## Project Overview

This project investigates whether **Moshi**, a full-duplex spoken language model, can be adapted to better handle complex mathematical question answering through parameter-efficient fine-tuning. The research explores the intersection of spoken dialogue systems and mathematical reasoning, proposing a novel pipeline that converts reasoning-augmented text data into stereo audio for LoRA-based adaptation of a pretrained speech model.

### Research Highlights

- **Full-duplex spoken model adaptation**: Extends Moshi's conversational capabilities to mathematical reasoning
- **Chain-of-thought spoken dialogue**: Converts text-based reasoning into synthetic spoken data for training
- **Parameter-efficient fine-tuning**: Uses LoRA (Low-Rank Adaptation) to minimize memory requirements
- **Proof-of-concept pipeline**: Demonstrates feasibility of teaching speech models mathematical language and structure
- **Resource-constrained optimization**: Addresses practical GPU memory limitations through architectural choices (rank-8 LoRA, 10-second audio, batch size 1)

## Team Members & Contributions

| Name | Student ID | Email | Contributions |
|------|-----------|-------|---------------|
| **Md Sabbir Hossain Tamim** | 2222037642 | sabbir.tamim3@northsouth.edu | Data preparation pipeline, stereo audio conversion, evaluation metrics |
| **Md Kaif Afran** | 2222134642 | Kaif.khan@northsouth.edu | Training infrastructure, LoRA configuration, checkpoint management, GitHub repository |
| **Sumaiya Binte Shamim** | 2212427042 | sumaiyabinte22@gmail.com | Literature review, report writing, qualitative analysis, visualization |

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended GPU: RTX 3090 or similar with ≥24GB VRAM)
- Git and pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Afran-zero/Project-CSE465--MATH-MOSHI
cd Project-CSE465--MATH-MOSHI

# Create virtual environment
python -m venv moshi_env
source moshi_env/bin/activate  # On Windows: moshi_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Data Preparation
Convert math reasoning dataset into stereo audio format:
```bash
python build_moshi_stereo.py \
  --input processed_dataset.jsonl \
  --output_dir ./data/training_audio/ \
  --sample_rate 16000
```

#### 2. Fine-Tuning
Train Moshi with LoRA adapters:
```bash
python moshi-finetune/train.py \
  --config training_config.yaml
```

#### 3. Evaluation
Evaluate the fine-tuned model:
```bash
python evaluate.py \
  --model_path ./checkpoints/final \
  --test_data ./data/test_samples.jsonl \
  --output results/comparison_report.csv
```

## Project Structure

```
Project-CSE465--MATH-MOSHI/
├── README.md                          # This file
├── report.tex                         # Full project report (LaTeX)
├── requirements.txt                   # Python dependencies
├── processed_dataset.jsonl            # Math reasoning dataset
├── processed_dataset.csv              # Dataset in CSV format
├── build_moshi_stereo.py             # Stereo audio conversion script
├── evaluate.py                        # Evaluation script
├── training_config.yaml               # LoRA fine-tuning configuration
├── moshi-finetune/
│   ├── train.py                       # Main training script
│   └── config.py                      # Training configuration parser
├── data/
│   ├── daily-talk-contiguous/        # Stereo audio training data
│   └── test_samples.jsonl            # Test evaluation samples
├── checkpoints/                       # Saved model checkpoints
│   ├── checkpoint-20/                # Intermediate checkpoint
│   ├── checkpoint-40/                # Intermediate checkpoint
│   └── checkpoint-100/               # Final checkpoint with LoRA adapters
└── results/
    ├── comparison_report.csv         # Evaluation results
    ├── loss.png                      # Training loss curve
    ├── throughput.png                # Throughput curve
    ├── learning_rate.png             # Learning rate schedule
    ├── probability.png               # Token probability curve
    └── gpu_usage.png                 # GPU memory usage curve
```

## Key Features

### 1. Reasoning-Aware Data Pipeline
- Structured math Q&A with chain-of-thought explanations
- Dual representation: question + reasoning-based answer
- Synthetic speech generation via TTS

### 2. Stereo Duplex Formatting
- **Left channel**: Assistant response (answer + reasoning)
- **Right channel**: User question
- Enables structured supervised learning for two-way interaction

### 3. LoRA-Based Adaptation
- **Rank**: 8 (reduced for memory efficiency)
- **Scaling**: 4.0
- **Frozen base model**: Preserves original speech capabilities
- **Memory efficient**: Gradient checkpointing enabled

### 4. Comprehensive Monitoring
- Real-time training loss, throughput, and memory tracking
- OneCycle learning rate schedule
- Token probability statistics
- GPU utilization monitoring

## Performance Summary

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Mean WER** | 0.95 - 1.12 | Surface-level word alignment with challenges |
| **Mean Semantic Similarity** | 0.47 - 0.68 | Moderate semantic alignment on test set |
| **Training Loss Trend** | Decreasing (with spikes) | Model learning despite limited data |
| **GPU Memory Usage** | ~20-24 GB | Optimized for constrained environments |
| **Training Time** | ~2 hours (RTX 3090) | Efficient fine-tuning loop |

### Results Analysis
- **Strengths**: Improved mathematical terminology recognition, better number handling, traces of reasoning patterns
- **Limitations**: Sequence truncation above ~30 seconds, weak end-task accuracy on complex reasoning, limited training data (100 examples)
- **Conclusion**: Proof-of-concept successful; strong performance requires larger datasets and more GPU memory

## Training Configuration

```yaml
# Data
data:
  train_data: '/content/data/daily-talk-contiguous/dailytalk.jsonl'
  shuffle: true

# Model
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

# LoRA Configuration
full_finetuning: false
lora:
  enable: true
  rank: 8
  scaling: 4.0
  ft_embed: false

# Training Hyperparameters
duration_sec: 10
batch_size: 1
max_steps: 100
gradient_checkpointing: true

# Optimization
optim:
  lr: 2.e-6
  weight_decay: 0.1
  pct_start: 0.05

# Checkpointing
ckpt_freq: 20
save_adapters: true
run_dir: "/content/test"
```

## Drive Link (Checkpoints, Codes & Environments)

All trained checkpoints, preprocessed datasets, TTS-generated audio, environment snapshots, and full training logs are available at:

**[Google Drive Link - To be provided]**

The drive contains:
- ✅ **Checkpoint-20, Checkpoint-40, Checkpoint-100** (with LoRA adapters)
- ✅ **Preprocessed stereo audio dataset** (daily-talk-contiguous/)
- ✅ **Training logs and metrics** (loss, throughput, GPU stats)
- ✅ **Evaluation results** (comparison reports and WER scores)
- ✅ **Environment snapshot** (Python dependencies, package versions)
- ✅ **Inference code** (for running fine-tuned model)

> Note: Due to file size limitations, checkpoints and audio data are hosted on Google Drive. Clone this repository and use the drive link to download the large artifacts.

## Reproduction Instructions

For detailed step-by-step reproduction instructions with automatic execution via autonomous agents, see **Section D: Agent Reproduction Prompt** in the full [report.tex](report.tex).

Quick summary:
1. Clone repository and install dependencies
2. Download pretrained Moshi checkpoint
3. Run stereo audio conversion on dataset
4. Execute fine-tuning with provided YAML config
5. Evaluate model on test samples
6. Compare results against expected metrics

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 20 GB | 24+ GB (RTX 3090, A100) |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32+ GB |
| **Disk** | 50 GB | 100+ GB |
| **Training Time** | 2-4 hours | 1-2 hours (A100) |

## References

- **Moshi**: Kyutai Labs full-duplex spoken language model
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Chain-of-Thought**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- **TTS**: Coqui TTS and related speech synthesis frameworks

## Citation

If you use this project in your research, please cite:

```bibtex
@article{tamim2024mathMoshi,
  title={Math Moshi: Teaching Moshi to Handle Complex Math Questions},
  author={Tamim, Md Sabbir Hossain and Afran, Md Kaif and Shamim, Sumaiya Binte},
  journal={CSE465 Pattern Recognition and Neural Networks Project},
  school={North South University},
  year={2024}
}
```

## License

This project is part of the CSE465 course at North South University. For academic use and collaboration, please contact the team members.

## Contact & Support

- **Project Repository**: https://github.com/Afran-zero/Project-CSE465--MATH-MOSHI
- **Report**: See [report.tex](report.tex) for detailed methodology, results, and analysis
- **Issues**: Please file issues on GitHub for bugs or questions

---

**Last Updated**: May 2024  
**Institution**: North South University, Dhaka, Bangladesh  
**Course**: CSE465 - Pattern Recognition and Neural Networks
