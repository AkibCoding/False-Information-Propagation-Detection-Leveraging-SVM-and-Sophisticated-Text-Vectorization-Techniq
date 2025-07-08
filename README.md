# 📰 Strengthening False Information Propagation Detection  
## Leveraging SVM and Sophisticated Text Vectorization Techniques in Comparison to BERT  

![IEEE QPAIN 2025](https://img.shields.io/badge/IEEE-QPAIN--2025-blue)  
**Authors:** Ahmed Akib Jawad Karim (BRAC University), Kazi Hafiz Md Asad (North South University), Aznur Azam (BAUST)  

---

## 🧠 Abstract

The rapid spread of fake news through online platforms has become a serious societal issue. This project presents a comparative study of traditional machine learning (SVM) and transformer-based models (BERT) for detecting fake news. We explore three text vectorization techniques—**Bag of Words (BoW)**, **TF-IDF**, and **Word2Vec**—for training SVM classifiers, and benchmark their performance against a **BERT-base** model. Despite BERT's superior accuracy, our findings reveal that **SVM with BoW** offers highly competitive results with minimal computational resources.

---

## 📦 Dataset

We used the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle, originally compiled by the ISOT Research Group at the University of Victoria.  

- **Real News:** 21,477 articles  
- **Fake News:** 23,421 articles  
- **Split:**  
  - Train: 17,121 real / 18,796 fake  
  - Test: 4,356 real / 4,624 fake  

---

## 🔄 Preprocessing

- Stopword removal (using NLTK)
- Tokenization (Gensim + BERT Tokenizer)
- Words shorter than 3 characters removed
- Vectorization Techniques:
  - **Bag of Words (BoW)**
  - **TF-IDF**
  - **Word2Vec (CBOW architecture)**

---

## 🧪 Models

### 🧩 Support Vector Machine (SVM)
- Kernels: **Linear** and **Radial Basis Function (RBF)**

### 🧠 BERT-base
- Pretrained `bert-base-uncased` fine-tuned for classification
- Trained for **3 epochs** on GPU (NVIDIA Tesla T4)

---

## 📊 Results

### 📈 SVM Performance Summary

| Vectorization | Kernel | Accuracy | F1-Score |
|---------------|--------|----------|----------|
| **BoW**       | Linear | **99.81%** | **0.9980** |
| TF-IDF        | Linear | 99.52%   | 0.9949   |
| Word2Vec      | Linear | 96.54%   | 0.9644   |
| BoW           | RBF    | 99.62%   | 0.9961   |
| TF-IDF        | RBF    | 99.31%   | 0.9928   |
| Word2Vec      | RBF    | 97.75%   | 0.9767   |

---

### 🤖 BERT-base Model Performance

| Epoch | Accuracy | F1-Score |
|-------|----------|----------|
| 3     | **99.98%** | **0.9998** |

- **Confusion Matrix:**  
  - TP (real): 4689  
  - TN (fake): 4265  
  - FP: 7  
  - FN: 19  

---

## ⚖️ Trade-off: Accuracy vs Resources

| Model | Device | Training Time | Accuracy |
|-------|--------|----------------|----------|
| SVM (BoW) | CPU (i7, 16GB RAM) | < 3 min | 99.81% |
| BERT-base | GPU (Tesla T4, 16GB VRAM) | ~1h 45m | 99.98% |

---

## 🎯 Key Takeaways

- **SVM with BoW** is nearly as accurate as **BERT** with drastically lower computational cost
- **Word2Vec** underperformed BoW and TF-IDF in this classification context
- **RBF kernel** offered marginal performance boosts over the linear kernel
- **BERT** is ideal for environments with high compute resources; **SVM** is better for real-time, lightweight applications

---

## 🔮 Future Work

- Hybrid architectures (e.g., BERT embeddings + SVM)
- Inclusion of metadata (e.g., user behavior, source credibility)
- Multimodal models (text + images)
- Real-time misinformation detection pipelines

---

## 📂 File Structure

