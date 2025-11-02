# 🧠 Brain Tumor Detection using Transfer Learning (Deep Learning Approach)

<div align="center">

### Automated Brain Tumor Classification from MRI Images  
**Using VGG16, ResNet50, and MobileNetV2 Pre-Trained CNN Models**

</div>

---

## 🩺 Project Overview

This project focuses on **automatic detection and classification of brain tumors** from MRI images using **Transfer Learning**, a deep learning approach.  
The goal is to assist in the early and accurate diagnosis of tumors, reducing human error and improving decision-making in medical imaging.

Pre-trained CNN architectures like **VGG16**, **ResNet50**, and **MobileNetV2** were implemented and compared.  
After performance evaluation, **MobileNetV2 achieved the highest accuracy** and was finalized as the main model.

---

## 🎯 Objective

- To classify MRI brain images into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**  
- To utilize **transfer learning** for faster training and improved accuracy  
- To compare different CNN architectures and visualize model performance  
- To enable **single image tumor detection** for real-time testing  

---

## 🧩 Dataset Description

The dataset used in this project is a **Brain MRI Dataset**, which contains labeled MRI scans of the human brain.

**Folder Structure:**

Brain Tumor MRI/
│
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── notumor/
│
└── Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/


Each image is resized to **128×128 pixels** before training.

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras |
| **Data Handling** | NumPy, scikit-learn, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Models Used** | VGG16, ResNet50, MobileNetV2 |

---

## 🧠 Model Details

### 📘 Architecture

- **Base Model:** MobileNetV2 (Pre-trained on ImageNet)  
- **Layers Added:**
  - Flatten  
  - Dense (128 neurons, ReLU)  
  - Dropout (0.3 & 0.2 for regularization)  
  - Output Layer (Softmax – 4 classes)

### ⚙️ Hyperparameters

| Parameter | Value |
|------------|--------|
| Image Size | 128 × 128 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 20 |

---

## 📊 Model Comparison

| Model | Accuracy | Remarks |
|--------|-----------|----------|
| **VGG16** | Good | Slower, more parameters |
| **ResNet50** | not good | Deeper network, complex |
| **MobileNetV2** | **Best ✅** | Lightweight, fast, accurate |

---

## 📈 Performance Visualization

- Accuracy & Loss Curves for each model  
- Confusion Matrix Heatmaps to evaluate predictions  
- ROC Curves & AUC Scores for multi-class evaluation  
- Classification Reports showing precision, recall & F1-score  

---

### 📈 Results Summary  

| 🧠 Model | 🎯 Accuracy (%) | 💬 Observations |
|:---------:|:---------------:|:----------------|
| **VGG16** | 96.72 | Good performance, slower convergence |
| **ResNet50** | 75.29 | Stable and balanced results |
| **MobileNetV2** | 🏆 **96.87** | Best performing and fastest model |

🧾 Requirements:
tensorflow  
numpy  
matplotlib  
seaborn  
pillow  
scikit-learn  


🔮 Future Enhancements:

Build a Web Interface using Flask or Streamlit

Integrate Grad-CAM Visualization for explainable AI

Enable real-time tumor detection from webcam input

Extend dataset for more tumor subtypes

📚 References:

TensorFlow & Keras Documentation

ImageNet Pre-trained Models (VGG16, ResNet50, MobileNetV2)

Brain MRI Dataset (Kaggle / Public Repositories)

✨ This project demonstrates how transfer learning can be effectively used to classify medical images and assist in brain tumor diagnosis.
