# Facial Emotion Recognition

A deep learning project for **facial emotion recognition** using the **FER2013 dataset**.  
The system classifies facial expressions into emotion categories and includes a **real-time emotion detection dashboard built with Gradio**.

---

# Overview

This project trains deep learning models to recognize facial emotions and deploys the best model in a **real-time interactive dashboard**.

The pipeline includes:

- training multiple neural network architectures
- evaluating model performance using multiple metrics
- comparing architectures
- deploying a real-time inference interface

---

# Models and Evaluation

Three architectures were explored and compared:

- MobileNetV3 Small
- EfficientNet-B0
- ViT Small

The models were evaluated using:

- Accuracy
- Macro F1 Score (primary metric)
- Balanced Accuracy
- Confusion Matrix

  see [comparison_table.md](emotion-recognition/results/comparison_table.md) for above metric comparison
  
---

# Emotion Detection Dashboard

The project includes a **Gradio-based web dashboard** for real-time emotion recognition.

### Features

The dashboard allows users to:

- capture images **directly from the webcam**
- upload images from their device
- detect facial emotions in real time

For each prediction the system displays:

- predicted emotion
- **probability distribution for all emotions**
- **confidence score**
- **fairness check across classes**

The probability visualization helps interpret the model's predictions and ensures transparent decision-making.

---

# Running the Dashboard

Install dependencies:
pip install torch torchvision timm gradio opencv-python scikit-learn numpy pillow

Run the dashboard: [gradio_app.py](emotion-recognition/app/gradio_app.py)

after running open the given local host in browser.

# File Structure

```
emotion-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/                    # Original FER2013 download (gitignored)
│ │ ├── train/
│ │ └── test/
│ │
│ └── balanced/               # Augmented balanced dataset (gitignored)
│ └── train/
│
├── notebooks/
│ ├── 01_data_loading.ipynb   # Download & organize FER2013
│ ├── 02_eda.ipynb            # EDA: class distribution, sample visualization, duplicates, corrupts
│ ├── 03_augmentation.ipynb   # Balancing strategies: oversampling, undersampling, class weights
│ ├── 04_training.ipynb       # Main training with all architecture experiments
│ └── 05_evaluation.ipynb     # Confusion matrix, per-class metrics, fairness checks
│ 
├── src/
│ ├── init.py
│ ├── dataset.py              # Custom Dataset class, transforms, data loading utilities
│ ├── train.py                # Training loop, validation, checkpoint saving
│ ├── losses.py               # CrossEntropy, FocalLoss, label smoothing
| └── models/                 # model factory
|   ├── mobilenetV3.py
|   ├── efficientNet.py
|   └── ViT.py
│
├── checkpoints/              # Saved .pth files (gitignored)
│ ├── mobilenet_v3_small_best.pth
│ ├── efficientnet_b0_best.pth
│ └── vit_small_best.pth
│
├── results/
│ ├── training_logs/           # accuracy logs per experiment
│ ├── confusion_matrices/      # Saved confusion matrix plots
│ └── comparison_table.md      # Final architecture × loss × balancing comparison
│
└── app/
  └── gradio_app.py            # Gradio webcam demo with emotion probabilities
```
