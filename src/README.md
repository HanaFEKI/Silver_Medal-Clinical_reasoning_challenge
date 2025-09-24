# BART vs BERT vs GPT

## 1. What is BART?

**BART (Bidirectional and Auto-Regressive Transformers)** is a Transformer model that combines:
- **BERT-like encoder** (bidirectional: sees both left and right context).
- **GPT-like decoder** (autoregressive: generates one token at a time).

It is pretrained as a **denoising autoencoder**: the input text is **corrupted** (masking, deletion, sentence shuffling…), and BART learns to **reconstruct** the original text.  
👉 This makes it very effective for **summarization, translation, text infilling, and paraphrasing**.

---

## 2. BART Training Process — Step by Step

1. **Corrupt input text**  
   Example:  
   Original:  
   > `The clinic reported a sudden increase in malaria cases in the northern region.`  
   Corrupted (masking + deletion):  
   > `The clinic reported a <mask> in malaria cases in the northern <mask>.`

2. **Encoder (BERT-like)**  
   Reads the corrupted input bidirectionally and produces contextual embeddings.

3. **Decoder (GPT-like)**  
   Autoregressively generates the original text, token by token.

4. **Loss**  
   Trains by maximizing the likelihood of reconstructing the original sentence.

---

## 3. Why BART is Useful
- Combines **deep understanding** (BERT encoder) + **fluent generation** (GPT decoder).  
- Pretraining with multiple noise types makes it robust for **seq2seq tasks**.  
- Strong out-of-the-box model for summarization (`facebook/bart-large-cnn`).

---

## 4. Comparing Models

| Model | Architecture | Pretraining Objective | Best For |
|-------|--------------|-----------------------|----------|
| **BERT** | Encoder-only | Masked Language Modeling | Classification, embeddings, search |
| **GPT** | Decoder-only | Next-token prediction | Free text generation, dialogue |
| **BART** | Encoder-Decoder | Denoising Autoencoder | Summarization, translation, infilling |

---

## 5. Example Phrase

Input:  
> `"The clinic reported a sudden increase in malaria cases in the northern region; patients commonly presented with fever, headaches, and vomiting."`

- **BART (summarization)** →  
  `"Malaria cases surge in the northern region; patients show fever, headaches and vomiting."`

- **GPT (continuation)** →  
  `"Local authorities are investigating possible mosquito breeding sites near rivers..."`  
  (fluent but not concise or guaranteed factual)

- **BERT (embeddings)** →  
  `[0.12, -0.07, 0.44, ...]` (vector used for classification/search, not text generation)

---

## 6. Code Examples

Install dependencies:
```bash
pip install transformers torch sentencepiece
