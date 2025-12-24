[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-Emotion%20Analysis-purple.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CONTRIBUTING.md)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

# Multi-Label Emotion Classification System

An **advanced emotion classification system** built with **Python, PyTorch, and Hugging Face Transformers**, powered by **RoBERTa fine-tuned on the GoEmotions dataset**.

The system detects **28 human emotions simultaneously (multi-label)** from English text and runs fully **offline** using locally stored model files, making it suitable for **secure, low-connectivity, and production environments**.

---

## **Features**

* **Multi-Label Emotion Detection:** Detect multiple emotions from a single text.
* **28 Emotion Categories:** Fine-grained emotional understanding.
* **Offline Inference:** No internet required during prediction.
* **Batch & Single Text Support:** Analyze one or multiple texts at once.
* **Transformer-Based Model:** High accuracy using RoBERTa architecture.
* **Lightweight Inference Mode:** Uses `model.eval()` and `torch.no_grad()`.
* **Easy Integration:** Can be embedded into APIs, web apps, or NLP pipelines.

---

## **Task Details**

| Property              | Description                                  |
| --------------------- | -------------------------------------------- |
| **Task**              | Multi-Label Emotion Classification           |
| **Number of Classes** | 28                                           |
| **Output Type**       | Multi-Label (Sigmoid Activation)             |
| **Framework**         | PyTorch                                     |
| **Model Type**        | RoBERTa (Hugging Face Transformers)          |
| **Inference Mode**    | Offline / Local                              |

---

## **Emotion Labels**

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

---

## **Technology Stack**

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP Library:** Hugging Face Transformers
* **Tokenizer:** AutoTokenizer
* **Model Loader:** AutoModelForSequenceClassification

---

## **Project Structure**

```

Emotion-Classification-System/
│
├── emotion_model/                # Local model & tokenizer files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── Emotion_Analysis.py           # Emotion prediction script
├── Download.py                   # Model download script
└── README.md                     # Project documentation

````

---

## **Setup Instructions**

### 1. Install Dependencies

```bash
pip install torch transformers
````

> ⚠️ Internet is **not required** during inference if the model is already stored locally.

---

### 2. Download the Model

Run once to download and save the model locally:

```bash
python Download.py
```

This stores the model at:

```
./Emotion_Labels/emotion_model
```

---

### 3. Run Emotion Prediction

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./Emotion_Labels/emotion_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

emotion_map = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval",
    11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy",
    18: "love", 19: "nervousness", 20: "optimism", 21: "pride",
    22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

def predict_emotion(texts, threshold=0.5):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)

    results = []
    for prob_vector in probs:
        emotions = [
            emotion_map[i]
            for i, score in enumerate(prob_vector)
            if score >= threshold
        ]
        results.append(emotions if emotions else ["neutral"])

    return results
```

---

## **Usage Example**

```python
print(predict_emotion("I am extremely happy today")[0])
```

**Output**

```
['joy', 'excitement', 'optimism']
```

```python
texts = ["I feel scared and nervous", "Thank you so much for your help"]
print(predict_emotion(texts))
```

**Output**

```
[['fear', 'nervousness'], ['gratitude', 'joy']]
```

---

## **Tips for Better Results**

* Adjust emotion **threshold values** for precision/recall balance.
* Keep text length under **512 tokens**.
* Batch processing improves inference speed.
* Works best on conversational or social text.

---

## **Applications**

* Emotion-aware chatbots
* Social media monitoring
* Customer feedback analysis
* Psychological text analysis
* Threat & risk analysis systems
* Offline NLP pipelines

---

## **License**

This project is **open-source** and released under the **MIT License**.

---

## **Author**

**Zihadul Islam**
GitHub: [https://github.com/zihadulislam99](https://github.com/zihadulislam99)
