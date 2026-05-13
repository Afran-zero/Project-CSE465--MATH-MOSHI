# Math Moshi: Teaching Moshi to Handle Complex Math Questions

## Project Overview

This project investigates whether **Moshi**, a full-duplex spoken language model, can be adapted to better handle complex mathematical question answering through parameter-efficient fine-tuning. The research explores the intersection of spoken dialogue systems and mathematical reasoning, proposing a novel pipeline that converts reasoning-augmented text data into stereo audio for LoRA-based adaptation of a pretrained speech model.
## Drive Link (Checkpoints, Codes & Environments)

All trained checkpoints, preprocessed datasets, TTS-generated audio, environment snapshots, and full training logs are available at:

** https://drive.google.com/file/d/1L2OBo9zhhE1guS33q1_MIXnBH__fPC6X/view?usp=sharing **

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


> Note: Due to file size limitations, checkpoints and audio data are hosted on Google Drive. Clone this repository and use the drive link to download the large artifacts.

## Reproduction Instructions

For detailed step-by-step reproduction instructions with automatic execution via autonomous agents, see **Section D: Agent Reproduction Prompt** in the full [report.tex](report.tex).

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
