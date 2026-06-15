# Extractive and Abstractive Summarization for Indonesian News Using QLoRA Fine-Tuning on Komodo 7B

This repository implements **extractive and abstractive text summarization** for Indonesian news articles using **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning on the [Komodo 7B](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base) large language model. The project leverages [Unsloth](https://github.com/unslothai/unsloth) for accelerated training and inference.

## Overview

Text summarization is a natural language processing task that aims to condense long documents into shorter, informative summaries. This project explores two approaches:

- **Extractive Summarization** — selects the most important sentences directly from the source text.
- **Abstractive Summarization** — generates new, paraphrased sentences that capture the core meaning.

Both tasks are fine-tuned using QLoRA on the Komodo 7B base model, an Indonesian language LLM developed by [Yellow AI NLP](https://huggingface.co/Yellow-AI-NLP).

## Key Features

- **QLoRA Fine-Tuning** — 4-bit quantized training with LoRA adapters (`r=32`, `alpha=32`, `dropout=0.05`), targeting `q_proj` and `v_proj` modules, training only ~0.25% of total parameters.
- **Unsloth Acceleration** — 2x faster free fine-tuning with Unsloth's optimized patches for xformers and gradient checkpointing.
- **Dual Summarization** — supports both extractive and abstractive summarization pipelines.
- **Comprehensive Evaluation** — uses ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) and BERTScore for automatic quality assessment.

## Repository Structure

```
├── README.md
├── training_32-32-0.05.ipynb     # Training notebook (QLoRA fine-tuning)
└── testing_32-32-0.05.ipynb      # Testing notebook (inference + evaluation)
└── testing_abstractive_with_generated_summary_and_scores-32-32-0.05.csv     # Abstractive testing result and score
└── testing_extractive_with_generated_summary_and_scores-32-32-0.05.csv      # Extractive testing result and score
└── hasil-training-32-32-0.05.csv    # Training and validation loss during training
```

## Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `Yellow-AI-NLP/komodo-7b-base` |
| Quantization | 4-bit (NF4) |
| LoRA Rank (`r`) | 32 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `v_proj` |
| Max Sequence Length | 2048 |
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW (default) |
| Precision | FP16 |
| Evaluation Strategy | Every 200 steps |
| Training Samples | 9,000 |
| Validation Samples | 500 |

## Tech Stack

- **Python 3.12**
- **PyTorch** — deep learning framework
- **Transformers** (Hugging Face) — model loading and tokenization
- **Unsloth** — fast LLM fine-tuning
- **PEFT** — parameter-efficient fine-tuning (LoRA)
- **TRL** — supervised fine-tuning trainer (`SFTTrainer`)
- **Datasets** (Hugging Face) — data handling
- **ROUGE Score** — reference-based evaluation
- **BERTScore** — semantic similarity evaluation
- **BitsAndBytes** — 4-bit quantization

## Usage

### 1. Training

Open `training_32-32-0.05.ipynb` in Jupyter Notebook or Kaggle. The notebook follows these steps:

1. **Install requirements** — installs all dependencies via `requirements.txt`.
2. **Import libraries** — loads Unsloth, Transformers, TRL, etc.
3. **Load data & model** — loads CSV datasets and the Komodo 7B base model with 4-bit quantization.
4. **Fine-tune with QLoRA** — applies LoRA adapters and trains using `SFTTrainer` for 3 epochs.
5. **Save model** — saves the best LoRA adapter weights and training logs.

### 2. Testing / Evaluation

Open `testing_32-32-0.05.ipynb`. The notebook:

1. **Install requirements** — installs dependencies via `requirements2.txt` (includes `rouge_score` and `bert_score`).
2. **Import libraries** — loads Unsloth, PEFT, ROUGE, BERTScore, etc.
3. **Load model + merge LoRA** — loads the base model and attaches the fine-tuned LoRA adapter.
4. **Run inference** — generates summaries for both extractive and abstractive test sets.
5. **Evaluate** — computes ROUGE and BERTScore metrics against reference summaries.

## Dataset

The model is trained on Indonesian news articles with paired summaries. The dataset is split into:

- **Training set** (`train_sample_prompt.csv`) — 9,000 samples
- **Validation set** (`valid_sample_prompt.csv`) — 500 samples
- **Test sets** — `test_ext_sample_prompt.csv` (extractive) and `test_abs_sample_prompt.csv` (abstractive)

Each sample contains a `text` field with the formatted input prompt for the summarization task.

## Results

Training logs (r=32, alpha=32, dropout=0.05):

| Step | Training Loss | Validation Loss |
|---:|---:|---:|
| 200 | 1.0068 | 1.0427 |
| 1000 | 0.9540 | 1.0222 |
| 2000 | 0.9526 | 1.0130 |
| 3000 | 0.9440 | 1.0049 |
| 4000 | 0.9240 | 1.0013 |
| 5000 | 0.8947 | 1.0012 |
| 6000 | 0.9246 | 1.0008 |
| 6750 | — | **0.9991** |

The model converges steadily, reaching a final validation loss of ~0.999 after 6,750 total training steps (~5.5 hours on a single Tesla V100 16GB GPU).

## Requirements

### Training (`requirements.txt`)
- `accelerate`
- `pandas`
- `torch`
- `datasets`
- `transformers`
- `unsloth`
- `peft`
- `trl`
- `tensorboardX`

### Testing (`requirements2.txt`)
- `unsloth`
- `rouge_score`
- `bert_score`

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) — for making LLM fine-tuning faster and more accessible.
- [Yellow AI NLP](https://huggingface.co/Yellow-AI-NLP) — for the Komodo 7B base model.
- [Hugging Face](https://huggingface.co/) — for the Transformers ecosystem.
