# Stroke Detection System Using Deep Learning

A deep learning-based computer-aided diagnosis (CAD) system for detecting stroke in brain CT images using Convolutional Neural Networks (CNN) and Gradient-weighted Class Activation Mapping (Grad-CAM).

##  Overview

An automated system that detects stroke presence in brain CT scans using **deep learning** with **Grad-CAM explainability**. The system provides both high-accuracy classification and visual explanations of model decisions, making it suitable for clinical decision support.

**Key Features:**
- Binary classification: Normal vs Stroke brain CT images
- CNN architecture with 4 convolutional blocks
- Grad-CAM for model interpretability
- Comprehensive performance metrics and visualizations
- GPU-accelerated training with TensorFlow

---

##  Output Visualizations

### Training Progress
<img width="1634" height="458" alt="image" src="https://github.com/user-attachments/assets/bbe7ad84-8b07-4e7a-a6d3-ddfe807a1509" />
Training and validation accuracy/loss curves showing model convergence

### Classification Performance
<img width="775" height="597" alt="image" src="https://github.com/user-attachments/assets/2b05531e-3626-43a9-88f6-a86e7f107779" />
Confusion matrix heatmap showing classification breakdown

### ROC Analysis
<img width="850" height="598" alt="image" src="https://github.com/user-attachments/assets/86c5f02e-dfbc-4bf8-be66-54af82fb8638" />
ROC curve with AUC score demonstrating model discrimination

### Clinical Dashboard
<img width="764" height="501" alt="image" src="https://github.com/user-attachments/assets/e82c257c-5635-4814-a934-cb5804f50991" />

6 test cases with original images, Grad-CAM heatmaps, and predictions

##  Dataset

| Parameter | Value |
|-----------|-------|
| Source | Kaggle - Brain Stroke CT Dataset |
| Total Images | 2,501 |
| Normal (Class 0) | 1,551 images |
| Stroke (Class 1) | 950 images |
| Image Size | 128x128 pixels (grayscale) |
| Split | Train 70% / Validation 15% / Test 15% |

---

##  Model Architecture

| Component | Details |
|-----------|---------|
| **Input** | 128x128x1 grayscale CT images |
| **Conv Blocks** | 4 blocks (32, 64, 128, 256 filters) |
| **Each Block** | Conv2D + BatchNormalization + MaxPooling |
| **Classifier** | GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1) |
| **Activation** | ReLU (hidden) / Sigmoid (output) |
| **Optimizer** | Adam |
| **Loss** | Binary Crossentropy |
| **Metrics** | Accuracy, Precision, Recall, F1-Score |

---

##  Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.20.0+
- NVIDIA GPU (recommended)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/stroke-detection.git
cd stroke-detection

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook stroke_detection.ipynb
```

### Requirements.txt

```bash
tensorflow>=2.20.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
pandas>=2.0.0
kagglehub>=0.1.0
```

---

##  Performance Metrics

The system evaluates using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness |
| **Precision** | Positive prediction reliability |
| **Recall** | Ability to detect actual strokes |
| **F1-Score** | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Model discrimination ability |

*Specific values generated in execution summary after training*

---

##  Grad-CAM Explainability

Gradient-weighted Class Activation Mapping provides:

- **Activation heatmaps** showing regions the model focuses on
- **Overlay on original CT** for clinical interpretation
- **Visual validation** of model decision-making
- **Support for** radiologist trust and adoption


##  Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.20.0 | Deep learning framework |
| Keras | - | Neural network API |
| OpenCV | 4.8.0 | Image processing |
| NumPy | 1.24.0 | Numerical computations |
| Matplotlib | 3.7.0 | Visualization |
| Seaborn | 0.12.0 | Statistical plots |
| Scikit-learn | 1.3.0 | Metrics & validation |
| KaggleHub | 0.1.0 | Dataset management |
