# NLP-Project
# Hate Speech Detection using DistilBERT with Attentive Pooling

This project aims to automatically classify user-generated social media content into **Hate Speech**, **Offensive Language**, or **Neutral Content**. We fine-tune a DistilBERT-based transformer model enhanced with attentive pooling and Focal Loss to address challenges like semantic overlap and class imbalance.

## 🔍 Project Overview

Social media platforms like Twitter often contain harmful content. The goal of this project is to develop a multi-class classifier that distinguishes:

- **Hate Speech**
- **Offensive Language** (non-hate)
- **Neutral Content**

We utilize DistilBERT with attentive pooling and Focal Loss to improve detection accuracy and manage class imbalance.

## 📊 Dataset

- **Source:** Manually labeled dataset of 5,000 tweets  
- **Classes:**
  - Hate Speech (10%)
  - Offensive Language (30%)
  - Neither (60%)
- **Split:** 70% Train, 15% Validation, 15% Test (Stratified)

## 🛠️ Methodology

### 🔧 Preprocessing
- Lowercasing, URL & HTML tag removal  
- Replacing mentions with `[USER]` and numbers with `[NUM]`  
- Emoji conversion using `emoji.demojize()`  
- Repeated character normalization  
- Hashtag retention for semantic context

### 🤖 Model Architecture
- **Backbone:** DistilBERT (fine-tuned)  
- **Attention:** Custom attentive pooling mechanism  
- **Classifier Head:** Dense → LayerNorm → Dropout  
- **Loss Function:** Focal Loss with class weighting  
- **Alternative Approach:** SMOTE on embeddings for classical ML classifiers

### ⚙️ Training Details
- `max_len=128`, `batch_size=32`  
- Layer-wise learning rates (2e-5 for classifier, 2e-6 for BERT)  
- Early stopping and ReduceLROnPlateau scheduler  
- Optimizer: Adam  
- Evaluation metrics: Precision, Recall, F1-score, Macro F1

## 🧪 Results

| Class              | Precision | Recall | F1-score |
|--------------------|-----------|--------|----------|
| Hate Speech        | 0.82      | 0.75   | 0.78     |
| Offensive Language | 0.85      | 0.88   | 0.86     |
| Neutral Content    | 0.91      | 0.94   | 0.93     |

- **Macro F1-score:** 0.86  
- **Best Classical Model (SMOTE + Logistic Regression):** Accuracy 93%

## 🧠 Key Features

- Custom `EnhancedHateSpeechDataset` and tokenizer handling  
- `AttentivePooling` layer to better capture contextual clues  
- Improved classification performance using Focal Loss  
- Embedding-level SMOTE to balance minority class representation  
- In-depth error analysis and visualizations
