# 📰 Fake News Detection using AI

## 📌 Project Overview
Fake news spreads rapidly across digital platforms and social media, making it difficult to distinguish between real and misleading information.  
This project presents an AI-based Fake News Detection System using the DistilBERT transformer model to automatically classify news articles as **REAL** or **FAKE**.

The system is trained using deep learning and Natural Language Processing (NLP) techniques to detect misinformation effectively.

---

## 🎯 Problem Statement
- Rapid spread of fake news on online platforms  
- Manual verification is time-consuming and inefficient  
- Need for an automated AI-based fake news detection system  

---

## 💡 Proposed Solution
We developed a deep learning model using **DistilBERT**, a lightweight transformer model, to classify news articles.

The system:
- Takes news text as input  
- Performs preprocessing and tokenization  
- Uses a fine-tuned DistilBERT model  
- Outputs prediction: REAL or FAKE  

---

## 🛠️ Technologies Used
- Python  
- HuggingFace Transformers  
- DistilBERT (Pretrained Model)  
- PyTorch  
- Pandas  
- Scikit-learn  
- Streamlit (for deployment)  

---

## 📂 Dataset Used
- WELFake Dataset (Fake & Real News)
- Real Science News Articles (Reuters, BBC)

Dataset includes labeled news articles for training and testing the model.

---

## ⚙️ System Development Approach

### 1️⃣ Data Collection
- Collected fake and real news datasets

### 2️⃣ Data Preprocessing
- Removed missing values  
- Cleaned text  
- Tokenized using DistilBERT tokenizer  

### 3️⃣ Model Training
- Fine-tuned DistilBERT for sequence classification  
- Used training & validation split  
- Optimized using AdamW optimizer  

### 4️⃣ Deployment
- Built a simple Streamlit web application  
- Users can input news text and get prediction  

---

## 🔁 Workflow

User Input  
↓  
Text Preprocessing  
↓  
Tokenization (DistilBERT)  
↓  
Model Prediction  
↓  
REAL / FAKE Output  

---

## 📊 Results
- Achieved approximately **85–90% accuracy**
- Successfully classified fake and real news articles



