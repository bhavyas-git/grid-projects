# Adversarial Robustness Analysis

This report summarizes the evaluation of adversarial robustness for the FER2013 emotion classification model using FGSM and PGD attacks, along with adversarial training.

---

# Experimental Setup

- **Dataset**: FER2013
- **Model**: ResNet18 (fine-tuned)
- **Loss Function**: Cross-Entropy
- **Optimizer**: Adam (lr = 1e-3)

---

# Adversarial Attack Configuration

### FGSM
- ε = 8 / 255 ≈ 0.03137

### PGD
- ε = 8 / 255 ≈ 0.03137
- α = 1 / 255 ≈ 0.00392
- Steps = 10

---

# Baseline Performance (Before Adversarial Training)

| Setting | Accuracy |
|--------|---------|
| Clean | 0.5339 |
| FGSM (ε=0.03137) | 0.0132 |
| PGD (ε=0.03137, α=0.00392, steps=10) | 0.0000 |

### Observations

- Significant drop from **0.5339 → 0.0132 (FGSM)**
- Even larger drop to **0.0000 under PGD**
- PGD demonstrates stronger attack capability due to iterative optimization

---

# Adversarial Training

- PGD adversarial examples injected into **~10% of training batches**
- Same attack parameters used during training
- Trained for 3 settings (consistent across experiments)

---

# Performance After Adversarial Training

| Setting | Before | After | Δ (Improvement) |
|--------|--------|--------|----------------|
| Clean | 0.5339 | 0.5995 | 0.0656 |
| FGSM | 0.0132 | 0.0857 | 0.0724 |
| PGD | 0.0000 | 0.0061 | 0.0061 |

---

# Key Insights

### 1. Robustness Improvement
- FGSM accuracy improved by **~0.0724**
- PGD accuracy improved by **~0.0061**
- Model becomes more stable under perturbations

### 2. Clean Accuracy Behavior
- Clean accuracy change: **0.5339 → 0.5995**
- Indicates regularization effect from adversarial training

### 3. Attack Strength Comparison
- PGD remains significantly stronger than FGSM
- Even after training, PGD accuracy is still lowest

---

# Visualization Analysis

- Adversarial perturbations are **visually imperceptible**
- Perturbation maps (×10 scaling) reveal structured noise patterns
- Some samples show:
  - No prediction change → model robustness
  - Prediction flips → vulnerability to small input changes

---

# Overall Conclusion

- The baseline model is **highly vulnerable** to adversarial attacks
- Adversarial training provides **consistent robustness gains**
- PGD remains a strong benchmark for worst-case robustness
- The approach improves both:
  - **Reliability under noise**
  - **Generalization performance**
---
