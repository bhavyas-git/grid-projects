
# Facial Emotion Recognition

A deep learning project for **facial emotion recognition** using the **FER2013 dataset**.  
The system classifies facial expressions into emotion categories and includes a **real-time emotion detection dashboard built with Gradio**.

Live Dashboard URL: https://bhavyagrid-emotion-recognition.hf.space
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

- Balanced Accuracy
- Macro F1 Score (primary metric)
- Balanced Accuracy
- Confusion Matrix
- Per-Class Accuracy

  see [comparison_table.md](emotion-recognition/results/comparison_table.md) for above metric comparison

  run the evaluation notebook:
  ```code
  notebooks/05_evaluation.ipynb
  ```
  Results will be saved inside:
  ```code
  results/
  ```
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

### Deploy to Hugging Face Spaces

This repo is now configured for direct deployment to a Gradio Space.

Live Space URL: https://bhavyagrid-emotion-recognition.hf.space

Files added for deployment:
- `app.py` as the Space entrypoint
- `packages.txt` for OpenCV/video system libraries
- YAML metadata at the top of this README for Space configuration

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
# Setup & Running the Project

This section explains how to install dependencies, download the dataset, train the models, and run the emotion detection dashboard.

---

### Clone the Repository

First clone the repository and move into the project directory.

```bash
git clone https://github.com/<your-username>/emotion-recognition.git
cd emotion-recognition
```
### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```
### Install Dependencies

```bash
pip install -r requirements.txt
```
Main libraries used in this project:

- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- OpenCV
- Gradio
- Matplotlib
- Seaborn
- Kaggle API

### Running Experiments

All experiments are organized inside the notebooks folder.
Suggested order:

```code
notebooks/01_data_loading.ipynb
notebooks/02_eda.ipynb
notebooks/03_augmentation.ipynb
notebooks/04_training.ipynb
notebooks/05_evaluation.ipynb
```
### training a model

Models can be trained directly using the training script.
```bash
python src/train.py
```
Training outputs are automatically saved to:

```code
logs/
checkpoints/
```
These include:

- training logs
- best model checkpoints
- experiment metadata

note that ipynb files have records of most recently trained models but experimentation with different finetuning techniques and architectures can be seen within 

```code
results/training_logs
```
