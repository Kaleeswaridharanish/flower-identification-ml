# flower-identification-ml
A machine learning project for flower identification using logistic regression and feature extraction from CNN.

🌸 flower-identification-ml
▶A machine learning project for flower identification using logistic regression and feature extraction from CNN (Convolutional Neural Networks).

📌 Project Overview
▶This project aims to identify different types of flowers by combining the power of deep learning and classical machine learning. We use a pre-trained CNN model to extract deep features from flower images, which are then classified using a logistic regression model.

🧠 Technologies Used Python 🐍

▶TensorFlow / Keras

▶Scikit-learn

▶NumPy & Pandas

▶Matplotlib & Seaborn

▶OpenCV (optional for image preprocessing)

⚙️ Workflow 

▶Data Collection 

Flower image dataset with 5 classes.

▶Preprocessing

Resize and normalize images
Convert labels into categorical format

▶Feature Extraction

Use a pre-trained CNN (e.g., VGG16 or MobileNet)

Extract features from intermediate layers

▶Model Training

Train a logistic regression classifier using the extracted features

▶Evaluation

Accuracy, precision, recall, and confusion matrix

Visual inspection of correctly and incorrectly classified images

📁 Folder Structure

flower-identification-ml/
├── train_model.py        # CNN feature extraction + logistic regression training
├── app.py                # Interface to upload and identify flower images
├── requirements.txt      # Project dependencies
├── README.md             # Project description
└── dataset/              # Folder containing flower images

💡 Key Features

▶Lightweight classifier with high accuracy

▶Efficient use of CNN features

▶Scalable to more flower classes

▶Clean and user-friendly interface

👥 Team Members

A. Kaleeswari

K. Santhiya

K. Preethi

