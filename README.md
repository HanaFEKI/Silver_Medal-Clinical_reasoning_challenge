# 🥈 Zindi Challenge: Kenya Clinical Reasoning Challenge – Silver Medal

This repository contains my solution to the **Kenya Clinical Reasoning Challenge** hosted on the Zindi platform. I ranked **62nd out of 440 participants**, earning a **Silver Medal** 🥈 by building an NLP-based model that attempts to emulate the clinical reasoning of frontline Kenyan healthcare workers.

---

## 🩺 Context

In resource-limited settings like rural Kenya, nurses must make high-stakes medical decisions with limited tools and specialist backup. This challenge provided **400 authentic clinical vignettes**—each simulating a real-world scenario involving patient presentation, facility type, and nurse background. The objective was to **predict the most appropriate clinician response** to each case.

Each response was previously evaluated by human experts and compared with top-tier AI models including **GPT-4**, **Gemini**, and **LLaMA**.

---

## 📦 About the Challenge

The dataset comprises **authentic clinical vignettes** sourced from Kenyan frontline healthcare environments. Each vignette describes a case scenario with relevant context such as:

- Patient symptoms and background
- Nurse experience level
- Facility type (e.g., dispensary, hospital)

Our task was to **predict the clinician’s written response** to each scenario — responses may include medical abbreviations, structured reasoning (e.g., "Summary:", "Diagnosis:", "Plan:"), or free-form clinical notes.

> 🔍 These vignettes simulate daily decisions made by nurses under resource constraints, often without access to advanced diagnostics or specialist support.

---

## 📊 Dataset Summary

- `train.csv`: 400 prompts and clinician responses  
- `test.csv`: 100 prompts (unlabeled)  
- Data is small but **high quality**, curated by domain experts  
- Tasks span **multiple medical specialties**, **geographies**, and **clinical contexts**  
- Responses are free text, requiring **robust natural language understanding**

---

## 🧠 My Approach

### 🔹 1. Text Preprocessing
- Cleaned and normalized text data with care to retain medical context
- Tokenized using **transformer-compatible tokenizers**

### 🔹 2. Feature Sensitivity
- Preserved structural cues common in clinical responses (e.g., `"Plan:"`, `"Vitals:"`, `"Dx:"`, `"Assessment:"`)
- Appended metadata (e.g., **nurse experience**, **facility level**) directly into the input sequence to inform the model contextually

### 🔹 3. Modeling
- Fine-tuned a **`facebook/bart-large`** model for a **text-to-text generation** task
- Employed **5-fold cross-validation** to maximize generalization and reliability on the limited training data
- Used **ROUGE score** for both validation and final leaderboard evaluation

> ⚠️ However, I believe **ROUGE is too restrictive** for evaluating free-text clinical outputs, as it focuses heavily on n-gram overlap and penalizes semantically correct but lexically different responses.  
> In the context of clinical reasoning—where clarity, structure, and medical accuracy matter more than phrasing—this leads to underperformance despite meaningful outputs.  

### 🔹 4. Evaluation Strategy
- Official evaluation used ROUGE only

---

### 🛠️ Resource Restrictions

| Constraint                        | Limit                                              |
|----------------------------------|----------------------------------------------------|
| **Model size**                   | ≤ 1 billion parameters                             |
| **Inference time**               | < 100ms per vignette                               |
| **Inference RAM usage**          | < 2 GB                                             |
| **Quantization required**        | Must be quantized for deployment                   |
| **Training time constraint**     | ≤ 24h on NVIDIA T4 or equivalent GPU               |
| **Inference device**             | Must run on Jetson Nano or equivalent (low-power)  |

These constraints made it especially difficult to train and deploy large models like `bart-large` efficiently, forcing careful trade-offs in model size, training time, and deployment feasibility.

---

## 📈 Results

| Metric              | Value             |
|---------------------|-------------------|
| Final Rank          | 62 / 440          |
| Medal               | 🥈 Silver         |
| Evaluation Metric   | ROUGE Score       |
| Private Leaderboard | 0.410231945       |

---

## 📁 Project Structure
```
├── notebooks/ # EDA and experiment tracking
├── src/
│ ├── preprocess.py # Tokenization & formatting
│ ├── model.py # Fine-tuning transformer models
│ ├── inference.py # Prediction script
│ └── utils.py # Text comparison and scoring
├── outputs/ # Submission files
├── requirements.txt # Dependencies
└── README.md # You're here!
```

---

## 💡 Key Learnings

- Medical NLP must prioritize interpretability and clinical fidelity  
- Evaluation metrics like ROUGE can fail to capture real clinical reasoning  
- A small dataset still offers deep insights when annotated by true domain experts  
- Balancing performance with deployment constraints is vital in real-world healthcare AI

---

## 🚀 Setup

```bash
# Clone repo
git clone https://github.com/HanaFEKI/Clinical_reasoning_challenge
cd zindi-kenya-clinical-nlp

# Install dependencies
pip install -r requirements.txt

# Run training
python src/model.py
```

---
# Author
Hana Feki 
Applied Math engineering student @ ENSTA Paris
Email: hana.feki@ensta.fr
Linkedin : https://www.linkedin.com/in/hana-feki/
