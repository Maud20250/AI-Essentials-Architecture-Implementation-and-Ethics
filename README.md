# Generative AI Essentials — Text Generation with a Word-Level LSTM

**Course:** Diploma in Data Analysis and Artificial Intelligence — Willis College
**Assignment:** Assignment 13 — Generative AI Essentials
**Author:** Jean Billa

## Project Overview

This project implements and explores a basic text-generation model as an introduction to
Generative AI and the ideas behind GPT-style language models. It covers:

- The Transformer/attention architecture and how GPTs generate text (tokenization,
  probability distributions, sequence generation) — see the project report.
- A from-scratch **Embedding → LSTM → Dense(softmax)** generative model trained on the
  **Tiny Shakespeare** dataset (public-domain text, commonly sourced from Project Gutenberg).
- A custom text-generation function implementing the four standard GPT decoding controls:
  **greedy decoding**, **temperature scaling**, **top-k sampling**, and **top-p (nucleus) sampling**.
- A simple **content-creation demo** showing how prompt engineering steers generated text.
- A discussion of the ethical considerations raised by generative AI systems.

## Repository Contents

| File | Description |
|---|---|
| `Assignment_13_Generative_AI_Essentials.ipynb` | Main Google Colab notebook: data prep, model, training, and all 5 exercises |
| `Generative_AI_Essentials_Report.pdf` | 3-page report covering architecture, methodology, applications, and ethics |
| `training_curves.png` | Training loss/accuracy plot generated during training |
| `README.md` | This file |

## Dataset

**Tiny Shakespeare** — a public-domain collection of Shakespeare's plays and poems, widely used
as a lightweight benchmark for text-generation demos. Downloaded from:
```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## How to Run

1. Open `Assignment_13_Generative_AI_Essentials.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells in order (Runtime → Run all). The notebook will:
   - Download the dataset
   - Build the word-level tokenizer and training sequences
   - Train the Embedding/LSTM model for 25 epochs (~2 minutes on Colab CPU)
   - Run all five exercises (basic generation, length control, temperature, top-k/top-p, prompt engineering)

### Dependencies
```
tensorflow
numpy
pandas
matplotlib
```
Install with:
```bash
pip install tensorflow numpy pandas matplotlib
```

## Model Summary

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding (vocab=3,461, dim=64) | (None, 6, 64) | 221,504 |
| LSTM (128 units) | (None, 128) | 98,816 |
| Dense (softmax, vocab=3,461) | (None, 3,461) | 446,469 |
| **Total** | | **766,789** |

Trained for 25 epochs (batch size 256, Adam optimizer, categorical cross-entropy loss).
Training accuracy rose from 3.6% to 9.6%; loss fell from 7.14 to 5.11 (see `training_curves.png`).

## Key Findings

- **Greedy decoding** quickly loops into repetitive phrases — a well-known limitation of
  deterministic decoding.
- **Temperature** controls the fluency/diversity trade-off: low temperature → safe, repetitive
  text; high temperature → more varied but less grammatical text.
- **Top-k / top-p sampling** restrict the candidate pool at each generation step, giving more
  control over output diversity than temperature alone.
- **Prompt engineering** (choice of seed phrase) visibly steers the vocabulary and tone of
  generated text, even in this small model.

## Ethical Considerations

See Section 5 of `Generative_AI_Essentials_Report.pdf` for a full discussion of misinformation,
bias, copyright/data provenance, misuse, and environmental cost, along with mitigation strategies.

## Notes

- This project trains a small model **from scratch** rather than loading a large pre-trained
  checkpoint (e.g., GPT-2), keeping the notebook fully reproducible offline within the assignment's
  time and compute constraints, while still demonstrating every core GPT concept (tokenization,
  next-token probability, and the standard decoding strategies).
- The model is intentionally lightweight for course purposes; a larger model, more training data,
  or a pre-trained Transformer would produce more fluent text.
