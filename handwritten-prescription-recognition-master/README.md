# 📝 Handwritten Prescription Recognition using Deep Learning

This project is an end-to-end Optical Character Recognition (OCR) system for recognizing handwritten medical prescriptions. Built using TensorFlow and Python, the system utilizes a hybrid deep learning architecture designed for sequence modeling and transcription of cursive handwriting.

---

## 📌 Overview

The OCR system combines convolutional feature extraction, temporal sequence modeling, and Connectionist Temporal Classification (CTC) decoding to recognize handwritten words from scanned images.

- Trained on over **115,000 word-images**
- Achieved **95.7% accuracy** on IAM Test Dataset
- Achieved **89.4% accuracy** on real-world doctor's prescriptions

---

## 🧠 Model Architecture

| Component | Description |
|----------|-------------|
| **CNN Layers (5)** | Extract visual features from word images using ReLU activations |
| **RNN Layers (2)** | Bi-directional LSTM layers for temporal sequence modeling |
| **CTC Layer** | Decodes sequences without needing character alignment |
| **Optimizer** | RMSProp minimizing batch-wise CTC loss |

---

## 📂 Dataset

- **Training:** [IAM On-Line Handwriting Database (IAM-OnDB)](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database)
- **Testing (real-world):** Scanned handwritten doctor prescriptions (custom dataset)

---

## 📈 Results

| Dataset | Accuracy |
|---------|----------|
| IAM Test Set | 95.7% |
| Real-world prescriptions | 89.4% |

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies
```bash
pip install tensorflow numpy matplotlib
