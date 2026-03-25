# Model Comparison Report

## Overview
This report compares three class imbalance handling strategies for emotion recognition:

- Focal Loss
- ADASYN Oversampling
- Mixup Augmentation

Evaluation is performed using Macro F1, Micro F1, and Minority F1 scores.

## Results

| Strategy | Macro F1 | Micro F1 | Minority F1 |
|----------|----------|----------|--------------|
| focal | 0.6928 | 0.7038 | 0.6303 |
| adasyn | 0.6285 | 0.6592 | 0.5196 |
| mixup | 0.6891 | 0.7108 | 0.5987 |

## Analysis

- **Best overall performance (Macro F1):** `focal` (0.6928)
- **Best minority class performance:** `focal` (0.6303)

### Key Observations

- Macro F1 reflects balanced performance across all classes.
- Micro F1 is influenced by dominant classes.
- Minority F1 highlights performance on underrepresented emotions.

### Conclusion

The `focal` strategy provides the best overall balance across classes and also performs best on the minority-class metric. This demonstrates the trade-off between overall accuracy and minority class sensitivity.
