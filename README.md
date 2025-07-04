# Brain Tumor Classification using CNNs

This repository contains the implementation of a Convolutional Neural Network (CNN) model aimed at classifying MRI images to detect brain tumors. The project focuses on preparing the data, developing and training the model, and evaluating its accuracy in distinguishing between tumor and non-tumor cases.

---

## Requirements

Ensure you have the following installed in your environment:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (optional, for experimentation)

Install dependencies using:

        !pip install tensorflow numpy matplotlib scikit-learn

---

# Project Overview

  ## Data Description
  - Data obtained from Kaggle.
  - Classes: Tumor and Non-Tumor.
  - Dataset Split:
      - Train:Validation:Test = 7:2:1
      - Tumor: 98 images (train: 68, val: 20, test: 10)
      - Non-Tumor: 154 images (train: 108, val: 30, test: 16)

  ## Model Architecture
  - A simple CNN model with three convolutional layers, dropout, and dense layers.
  - The model includes max-pooling for dimensionality reduction.
  - The final activation layer uses sigmoid for binary classification.

  ## Compilation and Callbacks
  - Optimizer: Adam (learning rate = 0.0001)
  - Loss: Binary Crossentropy
  - Metrics: Accuracy
  - Callbacks: Learning Rate Scheduler, Early Stopping

  ## Evaluation
  - Confusion Matrices and Accuracy for Train and Validation data.
  - Train and Validation Loss/Accuracy visualized with graphs.

 ##  Testing
  - Model tested on unseen data.
  - Threshold of 0.5 used for classification:
      - Prediction < 0.5: Healthy
      - Prediction ≥ 0.5: Tumor
  Correctly classified images included.

  ---

# Highlights

  ## Visual Analysis:
  - Train vs. Validation graphs of accuracy and loss.
  - Confusion Matrices for both train and validation sets.
  ## Testing Results:
  - Demonstration of model's performance on unseen data with accompanying visual examples.

---

# Acknowledgments

  This project was made possible with data contributions from Kaggle and open-source resources. The implementation is a simple yet effective demonstration of CNNs for medical imaging classification tasks.
  
  Feel free to fork, star ⭐, and contribute to this repository!
