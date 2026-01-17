# 🩺 Skin Cancer Prediction & Classification (Explainable AI)

A state-of-the-art medical imaging project focused on the detection and classification of skin cancer using multiple deep learning and machine learning architectures. This project standout by integrating **Explainable AI (XAI)** techniques to provide transparency in clinical decision-making.

## 🌟 Overview

Early detection of skin cancer is crucial for successful treatment. This project implements a robust pipeline to classify skin lesions from the **ISIC dataset** into various categories, utilizing diverse modeling strategies to ensure high accuracy and reliability.

### Key Features:
- **Multi-Model Approach**: Implements CNNs (DenseNet), Vision Transformers, XGBoost, and SVM for comprehensive performance comparison.
- **Explainable AI (Ex-AI)**: Uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps, showing which regions of a lesion the model is focusing on.
- **Advanced Training**: Employs K-Fold Cross Validation, stratified sampling, and specialized activation functions like **Swish**.
- **Automated Pipeline**: Includes modular utilities for data preprocessing, model training, and rigorous validation.

## 🛠️ Tech Stack

- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)
- **Transformers**: Vision Transformers (ViT) implementation
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/) (SVM, K-Fold)
- **Explainable AI**: Grad-CAM for saliency maps
- **Data Processing**: [OpenCV](https://opencv.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), Seaborn

## 📁 Project Structure

```text
Skin_Cancer_Prediction_and_Classification_Ex_AI/
├── ISIC Skin Cancer/       # Lesion images categorized by type
├── DataUtility.py          # Utilities for data loading and augmentation
├── GradCamUtility.py       # Implementation of XAI (Grad-CAM) visualizations
├── ModelTrainer.py         # Script for automated model training and CV
├── ModelValidator.py       # Tools for clinical validation and scoring
├── Prediction_DeepLearning_Dense.ipynb  # CNN/DenseNet architecture
├── Prediction_Transformers.ipynb       # Vision Transformer (ViT) implementation
├── Prediction_Xgboost.ipynb             # Feature-extracted XGBoost classification
├── requirements.txt        # Python library dependencies
└── README.md               # Extensive project documentation
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/namanviber/Skin_Cancer_Prediction_and_Classification_Ex_AI.git
   cd Skin_Cancer_Prediction_and_Classification_Ex_AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the Models**:
   Open any of the provided Jupyter Notebooks to see the training and explanation process in action.

## 📊 Methodology

1. **Data Preprocessing**: Images are resized, normalized, and augmented to ensure model generalization.
2. **Feature Extraction**: Deep features are extracted using pre-trained backbones for classic ML models (XGBoost/SVM).
3. **Training**: Models are trained using advanced techniques like learning rate reduction and early stopping.
4. **Explanation**: Grad-CAM heatmaps are generated for test cases to validate that the model is identifying clinically relevant features.
