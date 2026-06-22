# Contexta 🧠

**A character-aware Generative Pre-trained Transformer built from scratch in PyTorch — trained locally, controlled entirely by you.**

---

What if you could build something close to how GPT actually works — not wrap an API, but write the attention mechanism yourself, train weights from a raw text file, and watch a model generate coherent sentences character by character? That's exactly what this project is.

NanoForge GPT is a minimal yet complete implementation of the Transformer decoder architecture, packaged with an interactive Streamlit interface so you can experiment with temperature, sampling strategies, and repetition control in real time. It runs entirely on your local machine — no internet required after setup, no API keys, no external model weights.

---

## 🔬 Research Background

This project grew out of reading foundational research in deep learning and natural language processing. The core architecture is directly inspired by the landmark 2017 paper:

> **Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)**  
> *The paper that introduced the Transformer — replacing recurrent networks with pure self-attention mechanisms, fundamentally reshaping the landscape of NLP and generative AI.*

Additional concepts implemented here draw from:

- **Radford et al., "Language Models are Unsupervised Multitask Learners" (OpenAI, 2019)** — the GPT-2 paper, which demonstrated how a decoder-only Transformer trained on raw text can generalize across diverse tasks.
- **Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)** — scaling and emergent capabilities from large-scale language modeling.
- **Karpathy, "nanoGPT" (2022)** — a minimal, pedagogically clear implementation that bridged the gap between the original paper and practical PyTorch code.

The goal wasn't to replicate a production-grade LLM. It was to deeply understand *how* these systems work at the layer level — token embeddings, positional encodings, masked self-attention, feedforward projections, and stochastic sampling — by building each piece from scratch.

---

## 🗂️ Project Structure

```
Contexta/
│
├── app.py              # Streamlit UI — the interactive front end
├── gpt.py              # The Transformer model (embeddings, attention heads, blocks)
├── data_loader.py      # Custom regex-based tokenizer and vocabulary builder
├── train.py            # Training loop from scratch
├── train_resume.py     # Resume training from a saved checkpoint
├── input.txt           # Your training corpus (plain text)
└── model.pth           # Saved model weights (generated after training)
```

---

## 🧠 How It Works

### 1. Tokenization (`data_loader.py`)
Rather than using off-the-shelf tokenizers like BPE (Byte-Pair Encoding) or SentencePiece, this project implements a custom regex-based tokenizer from scratch. It parses the corpus into discrete tokens — words, numbers, punctuation, and whitespace — builds a vocabulary dictionary (`stoi` for string-to-index, `itos` for index-to-string), and encodes the entire corpus into integer sequences for training.

This is intentionally kept simple to keep the data pipeline transparent and auditable.

### 2. The Transformer Model (`gpt.py`)
The model follows the decoder-only Transformer architecture from "Attention Is All You Need":

- **Token + Positional Embeddings** — Each token is mapped to a dense vector, combined with a learned positional embedding so the model understands sequence order.
- **Multi-Head Self-Attention** — The `Head` class computes Query, Key, and Value projections, then calculates scaled dot-product attention scores between all tokens in the context window. A lower-triangular mask ensures the model never "sees" future tokens during training (causal masking).
- **Feedforward Layers** — Each attention block is followed by a position-wise feedforward network with a nonlinear activation.
- **Regularization** — Dropout is applied throughout to reduce overfitting on small corpora.
- **Optimizer** — AdamW is used for training, with weight decay applied to non-bias parameters.

### 3. Stochastic Text Generation
At inference time, the model doesn't just pick the most probable next token every time (that would produce repetitive, boring output). Instead:

| Parameter | What It Does |
|---|---|
| **Temperature** | Scales the raw logits before softmax. Lower → more conservative, higher → more creative/chaotic. |
| **Top-K Sampling** | Limits sampling to only the K most probable tokens, filtering out unlikely words entirely. |
| **Repetition Penalty** | Dynamically reduces the probability of tokens already seen in the current sequence, breaking loops. |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- A GPU is helpful but not required — the model is intentionally small enough to train on CPU too (just slower)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sayankumarnandi2003/NanoForge-GPT.git
cd NanoForge-GPT
```

### Step 2: Install Dependencies

```bash
pip install torch streamlit
```

> If you have an NVIDIA GPU and want to use CUDA acceleration, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🚀 Running the Project

### Step 1: Add Your Training Data

Open `input.txt` and replace its contents with any plain text you want the model to learn from — a book, historical records, song lyrics, code snippets, whatever you like. The larger and more consistent the text, the better the model will generalize.

> The current corpus contains historical and geographical text (used for development and testing).

### Step 2: Train the Model

```bash
python train.py
```

This will:
- Tokenize `input.txt` and build the vocabulary
- Initialize the Transformer weights
- Run the training loop and periodically print the loss
- Save the trained weights to `model.pth` when done

Training time depends on your corpus size, context window, and hardware. On a mid-range GPU, expect a few minutes to a few hours depending on depth settings.

### Step 3: Resume Training (Optional)

If training was interrupted or you want to continue from a checkpoint:

```bash
python train_resume.py
```

### Step 4: Launch the Interactive UI

```bash
python -m streamlit run app.py
```

Then open your browser at `http://localhost:8501`. You'll see sliders and input fields for:
- Prompt (seed text to start generation)
- Max tokens to generate
- Temperature
- Top-K value
- Repetition penalty

Adjust and generate — no retraining needed between runs.

---

## 🎛️ Tuning Tips

| Goal | Suggested Settings |
|---|---|
| Coherent, conservative output | Temperature: 0.6–0.8, Top-K: 10–20 |
| Creative, diverse output | Temperature: 1.0–1.3, Top-K: 40–100 |
| Fix repetitive output | Increase Repetition Penalty (try 1.2–1.5) |
| Short focused responses | Lower max tokens, lower temperature |

---

## 📌 Current Limitations & Honest Notes

- This is a **research and learning project**, not a production chatbot. It doesn't have billions of parameters, RLHF fine-tuning, or instruction-following capabilities.
- Quality of output is directly tied to the size and quality of `input.txt`. Small or noisy corpora will produce limited results.
- No streaming output is implemented yet — generation completes before displaying.

These are known limitations and potential directions for future work.

---

## 🔭 What's Next (Potential Extensions)

- [ ] Byte-Pair Encoding (BPE) tokenization for better subword handling
- [ ] Configurable depth (number of layers and heads) via CLI or config file
- [ ] Streaming token output in the Streamlit UI
- [ ] Fine-tuning on domain-specific corpora
- [ ] Export to ONNX for inference optimization

---

## 📄 License

MIT License. Free to use, modify, and build on.

---

## 🙋 About

Built by **Sayan Kumar Nandi** as part of an independent deep learning study focused on understanding the internals of large language models — from tokenization to attention to autoregressive generation.

If you have questions, suggestions, or want to collaborate, feel free to open an issue or reach out.

---

*"The goal of this project was never to compete with GPT-4. It was to understand how the first GPT came to be."*