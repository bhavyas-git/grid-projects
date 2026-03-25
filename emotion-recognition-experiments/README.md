

# Emotion Recognition from Faces

This project builds a facial emotion recognition pipeline on **FER2013** and extends it into five applied tasks:

1. model optimization and deployment export
2. attention mechanism analysis for ViT
3. class balancing experiments
4. adversarial robustness testing
5. real-time fairness-aware Gradio deployment

The repository mixes notebooks for experimentation and reporting with `src/` modules for reusable training, evaluation, and task-specific utilities.

## Project Summary

The core classification problem is 7-class facial emotion recognition:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Architectures explored in the project:

- `MobileNetV3 Small`
- `EfficientNet-B0`
- `ViT Small`

Primary outputs of the project:

- trained checkpoints
- per-experiment logs and summaries
- optimization artifacts such as ONNX and Core ML exports
- analysis notebooks for attention and adversarial robustness
- a Gradio dashboard for local and Hugging Face Spaces deployment

## The Five Tasks

### Task 1: Optimization and Deployment Readiness

Goal:
- reduce inference cost while preserving acceptable predictive behavior

What was done:
- exported trained models to `ONNX`
- converted/exported to `Core ML`
- created a quantized mobile-friendly checkpoint
- benchmarked latency and accuracy tradeoffs
- added a numerical consistency check across exported formats

Relevant files:
- [08_coreml_analysis.ipynb](notebooks/08_coreml_analysis.ipynb)
- [benchmark.py](src/task_1_optimization/benchmark.py)
- [convert_coreml.py](src/task_1_optimization/convert_coreml.py)
- [export_onnx.py](src/task_1_optimization/export_onnx.py)
- [results.py](src/task_1_optimization/results.py)

### Task 2: Attention Mechanism Analysis

Goal:
- inspect which spatial regions and attention heads matter for ViT-based emotion recognition

What was done:
- extracted last-block multi-head attention from the fine-tuned ViT
- computed per-emotion mean spatial attention maps
- estimated mutual information for attention heads
- identified high-MI heads and summarized their per-emotion behavior
- tested ablation by masking selected heads

Relevant files:
- [07_attention_analysis.ipynb](notebooks/07_attention_analysis.ipynb)
- [extract.py](src/task_2_attention/extract.py)
- [mutual_info.py](src/task_2_attention/mutual_info.py)
- [ablation.py](src/task_2_attention/ablation.py)

### Task 3: Class Balancing

Goal:
- improve minority-class behavior and reduce class imbalance effects

What was done:
- compared balancing strategies such as `ADASYN`, `Focal Loss`, and `MixUp`
- trained separate variants for each balancing strategy
- recorded per-class outcomes and training summaries
- compared balanced training runs against baseline behavior

Relevant files:
- [06_class_balancing.ipynb](notebooks/06_class_balancing.ipynb)
- [train_adasyn.py](src/task_3_balancing/train_adasyn.py)
- [train_focal.py](src/task_3_balancing/train_focal.py)
- [train_mixup.py](src/task_3_balancing/train_mixup.py)

### Task 4: Adversarial Robustness

Goal:
- test how brittle the emotion classifier is under small perturbations and whether adversarial training improves robustness

What was done:
- trained a standard ResNet-based classifier with cross-entropy loss
- implemented `FGSM` and `PGD`
- benchmarked clean, FGSM, and PGD accuracy before and after robust training
- trained a robust model with PGD-based adversarial training on a subset of batches
- visualized adversarial perturbations per emotion

Relevant files:
- [09_adversarial_analysis.ipynb](notebooks/09_adversarial_analysis.ipynb)
- [attacks.py](src/task_4_adversarial/attacks.py)
- [evaluate.py](src/task_4_adversarial/evaluate.py)
- [train_robust.py](src/task_4_adversarial/train_robust.py)

### Task 5: Real-Time Fairness Dashboard

Goal:
- serve the trained model in an interactive application with prediction inspection and fairness-oriented monitoring

What was done:
- built a Gradio dashboard for webcam and video analysis
- displayed emotion probability distributions and fairness heatmaps
- logged predictions with timestamps
- added sliding-window fairness summaries and alerting logic
- prepared a Hugging Face Spaces deployment path

Relevant files:
- [gradio_app.py](app/gradio_app.py)
- [app.py](app.py)
- [packages.txt](packages.txt)

Live Space:
- https://bhavyagrid-emotion-recognition.hf.space

## Experimentation and Analysis Flow

The project was developed as a staged workflow rather than one single training script.

### 1. Data preparation and baseline understanding

The early notebooks establish the dataset and the core problem:

- [01_data_loading.ipynb](notebooks/01_data_loading.ipynb) organizes FER2013 data
- [02_eda.ipynb](notebooks/02_eda.ipynb) inspects class distribution and sample quality
- [03_augmentation.ipynb](notebooks/03_augmentation.ipynb) explores augmentation and balancing ideas

### 2. Model training and model selection

[04_training.ipynb](notebooks/04_training.ipynb) is where the main architecture experiments were run.

Typical comparison dimensions:

- architecture choice
- loss function choice
- balancing strategy
- checkpoint quality at best epoch

Outputs were saved to:

- `results/checkpoints/`
- `results/training_logs/`
- `results/summaries/`

### 3. Evaluation and comparison

[05_evaluation.ipynb](notebooks/05_evaluation.ipynb) consolidates the trained models and compares them using:

- accuracy
- balanced accuracy
- macro F1
- per-class behavior
- confusion analysis

### 4. Task-specific follow-up analyses

After selecting promising models, the project branches into task-specific analysis:

- balancing experiments
- attention interpretability
- optimization/export analysis
- adversarial robustness
- fairness-aware deployment

This separation is deliberate: each task answers a different engineering question rather than trying to overload one notebook with every result.

## Current File Structure

```text
emotion-recognition/
├── README.md                             # project overview, task summary, setup, deployment notes
├── .gitignore                            # local artifacts and generated file exclusions
├── .gitattributes                        # git attributes used for the repo
├── requirements.txt                      # runtime dependencies currently used by the app/Space
├── packages.txt                          # system packages for Hugging Face Spaces
├── app.py                                # Hugging Face Spaces entrypoint for the Gradio app
├── report.pdf                            # final report for this project
│
├── app/
│   ├── __init__.py                       # package marker for app imports
│   ├── gradio_app.py                     # main fairness-aware webcam/video dashboard
│   └── fairness_log.csv                  # local prediction log generated by the dashboard (gitignored)
│
├── data/
│   ├── raw/                              # FER2013 train/test folders used by notebooks and loaders (gitignored)
│   └── balanced/                         # balanced dataset artifacts used in balancing experiments (gitignored)
│
├── notebooks/
│   ├── 01_data_loading.ipynb             # dataset loading and organization
│   ├── 02_eda.ipynb                      # class imbalance and data inspection
│   ├── 03_augmentation.ipynb             # augmentation and balancing exploration
│   ├── 04_training.ipynb                 # main model training experiments
│   ├── 05_evaluation.ipynb               # evaluation and architecture comparison
│   ├── 06_class_balancing.ipynb          # Task 3 balancing analysis
│   ├── 07_attention_analysis.ipynb       # Task 2 ViT attention attribution analysis
│   ├── 08_coreml_analysis.ipynb          # Task 1 optimization/export analysis
│   └── 09_adversarial_analysis.ipynb     # Task 4 adversarial robustness analysis
│
├── results/
│   ├── adasyn_per_class.txt              # per-class metrics for ADASYN experiment
│   ├── focal_per_class.txt               # per-class metrics for focal-loss experiment
│   ├── mixup_per_class.txt               # per-class metrics for MixUp experiment
│   ├── balancing_comparison.md           # written summary for Task 3
│   ├── attention_summary.md              # written summary for Task 2 
│   ├── adversarial_results.md            # written summary for Task 4 
│   ├── coreml_benchmark.md               # written summary for Task 1 
│   ├── model.mlmodel                     # Core ML export 
│   ├── model_quantized.mlmodel           # quantized Core ML export
│   ├── model_fp32.onnx                   # ONNX export 
│   ├── checkpoints/
│   │   ├── adasyn_best.pth               # best ADASYN-balanced checkpoint (gitignored)
│   │   ├── focal_best.pth                # best focal-loss checkpoint (gitignored)
│   │   ├── mixup_best.pth                # best MixUp checkpoint (gitignored)
│   │   ├── mobilenet_best.pth            # best MobileNet dashboard checkpoint (gitignored)
│   │   ├── mobilenet_cross_entropy_best.pth  # baseline MobileNet checkpoint (gitignored)
│   │   ├── mobilenet_quantized.pth       # quantized MobileNet checkpoint (gitignored)
│   │   └── vit_best.pth                  # best ViT checkpoint for attention analysis (gitignored)
│   ├── summaries/
│   │   ├── adasyn_best.md                # best-epoch summary for ADASYN run 
│   │   ├── focal_best.md                 # best-epoch summary for focal-loss run 
│   │   ├── mixup_best.md                 # best-epoch summary for MixUp run 
│   │   ├── mobilenet_best.md             # best-epoch summary for MobileNet run
│   │   ├── mobilenet_cross_entropy_best.md # best-epoch summary for baseline MobileNet run
│   │   └── vit_best.md                   # best-epoch summary for ViT run 
│   └── training_logs/
│       ├── adasyn_*.log                  # training log for ADASYN experiment 
│       ├── focal_*.log                   # training log for focal-loss experiment 
│       ├── mixup_*.log                   # training log for MixUp experiment 
│       ├── mobilenet_*.log               # training log for MobileNet experiment 
│       ├── mobilenet_cross_entropy_*.log # training log for baseline MobileNet experiment 
│       └── vit_*.log                     # training log for ViT experiment 
│
└── src/
    ├── __init__.py                       # source package marker
    ├── core/
    │   ├── __init__.py                   # package marker for shared training utilities
    │   ├── comparison.py                 # model comparison helpers
    │   ├── dataset.py                    # transforms and dataloader creation
    │   ├── losses.py                     # loss definitions used in experiments
    │   └── train.py                      # shared training loop utilities
    ├── models/
    │   ├── __init__.py                   # package marker for model definitions
    │   ├── efficientNet.py               # EfficientNet model setup
    │   ├── mobilenetV3.py                # MobileNetV3 model setup
    │   └── ViT.py                        # ViT model setup
    ├── task_1_optimization/
    │   ├── __init__.py                   # package marker
    │   ├── benchmark.py                  # latency/accuracy benchmarking for exports
    │   ├── compress_coreml.py            # Core ML compression utilities
    │   ├── convert_coreml.py             # Core ML export conversion
    │   ├── export_onnx.py                # ONNX export logic
    │   └── prune.py                      # pruning utilities
    ├── task_2_attention/
    │   ├── __init__.py                   # package marker
    │   ├── ablation.py                   # attention head ablation logic
    │   ├── extract.py                    # last-block attention extraction and plotting
    │   └── mutual_info.py                # MI scoring for attention heads
    ├── task_3_balancing/
    │   ├── __init__.py                   # package marker
    │   ├── train_adasyn.py               # ADASYN-based training pipeline
    │   ├── train_focal.py                # focal-loss training pipeline
    │   └── train_mixup.py                # MixUp training pipeline
    └── task_4_adversarial/
        ├── __init__.py                   # package marker
        ├── attacks.py                    # FGSM and PGD implementations
        ├── evaluate.py                   # clean/adversarial evaluation helpers
        └── train_robust.py               # adversarial training loop
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/emotion-recognition.git
cd emotion-recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the notebooks

Suggested order:

```text
01_data_loading.ipynb
02_eda.ipynb
03_augmentation.ipynb
04_training.ipynb
05_evaluation.ipynb
06_class_balancing.ipynb
07_attention_analysis.ipynb
08_coreml_analysis.ipynb
09_adversarial_analysis.ipynb
```

### Run the dashboard locally

```bash
./.venv/bin/python app/gradio_app.py
```

### Deploy to Hugging Face Spaces

This repo is configured for a Gradio Space.

Space repo notes:

- the Space repo should not store `mobilenet_best.pth` directly
- the app can download the checkpoint from a separate Hugging Face model repo
- set `HF_MODEL_REPO_ID` in the Space variables
- optionally set `HF_MODEL_FILE` if the filename differs
- set `HF_TOKEN` only if the model repo is private

Recommended model repo:

```text
Repo type: Model
Repo name: emotion-recognition-checkpoints
File: mobilenet_best.pth
```

## Notes

- `results.py` under Task 1 is a random-input export sanity check, not the benchmark script
- the Gradio app is the main deployment artifact for Task 5
- notebooks are used to document experiments, while `src/` holds reusable task logic
- in task 5 detect_demographics is random. group labels are synthetic and metrics are illustrative, not real fairness measurement.
