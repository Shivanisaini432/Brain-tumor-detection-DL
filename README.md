🧠 Brain Tumor Detection using Transfer Learning (Deep Learning Approach)

🩺 Project Overview

This project focuses on automatic detection and classification of brain tumors from MRI images using Transfer Learning, a deep learning approach.
The goal is to assist in the early and accurate diagnosis of tumors, reducing human error and improving decision-making in medical imaging.

Pre-trained CNN architectures like VGG16, ResNet50, and MobileNetV2 were implemented and compared.
After performance evaluation, MobileNetV2 achieved the highest accuracy and was finalized as the main model.

🎯 Objective

To classify MRI brain images into four categories: glioma, meningioma, pituitary, and no tumor.

To utilize transfer learning for faster training and improved accuracy.

To compare different CNN architectures and visualize model performance.

To enable single image tumor detection for real-time testing.

🧩 Dataset Description

The dataset used in this project is a Brain MRI Dataset, which contains labeled MRI scans of the human brain.

Structure:

Brain Tumor MRI/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/


Each image is resized to 128×128 pixels before training.

⚙️ Technologies Used
Category	Tools / Libraries
Language	Python
Frameworks	TensorFlow, Keras
Data Handling	NumPy, scikit-learn, Pillow
Visualization	Matplotlib, Seaborn
Models Used	VGG16, ResNet50, MobileNetV2
🧠 Model Details
📘 Architecture:

Base Model: MobileNetV2 (Pre-trained on ImageNet)

Layers Added:

Flatten

Dense (128 neurons, ReLU)

Dropout (0.3 & 0.2 for regularization)

Output Layer (Softmax – 4 classes)

⚙️ Hyperparameters:
Parameter	Value
Image Size	128 × 128
Optimizer	Adam
Learning Rate	0.0001
Loss Function	Sparse Categorical Crossentropy
Epochs	10
Batch Size	20
📊 Model Comparison
Model	Accuracy	Remarks
VGG16	Good	Slower, more parameters
ResNet50	not Good	Deeper network, complex
MobileNetV2	Best	Lightweight, fast, accurate ✅
📈 Performance Visualization

Accuracy & Loss Curves for each model

Confusion Matrix Heatmaps to evaluate predictions

ROC Curves & AUC Scores for multi-class evaluation

Classification Reports showing precision, recall & F1-score

🖼️ Single Image Prediction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load model
model = load_model("MobileNetV2_model.keras")

# Load and preprocess image
img_path = "sample_image.jpg"
img = load_img(img_path, target_size=(128,128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
result = classes[np.argmax(predictions)]

print(f"Predicted Tumor Type: {result}")

🧮 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Shivanisaini432/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Add Dataset

Place the dataset folders inside your project directory.

4️⃣ Train Models
python brain_tumor_training.py

5️⃣ Evaluate & Compare Models
python evaluate_models.py

6️⃣ Predict a Single MRI Image
python predict_single_image.py

🧾 Requirements
tensorflow
numpy
matplotlib
seaborn
pillow
scikit-learn



🔮 Future Enhancements

Build a Web Interface using Flask or Streamlit

Integrate Grad-CAM Visualization for explainable AI

Enable real-time tumor detection from webcam input

Extend dataset for more tumor subtypes

📚 References

TensorFlow & Keras Documentation

ImageNet Pre-trained Models (VGG16, ResNet50, MobileNetV2)

Brain MRI Dataset (Kaggle / Public Repositories)

✨ This project demonstrates how transfer learning can be effectively used to classify medical images and assist in brain tumor diagnosis.
