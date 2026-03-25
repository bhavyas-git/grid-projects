# Attention Analysis Summary

## Key Results

- Baseline Accuracy: **0.6452**
- Mask Low-MI Heads: **0.6452** (Δ 0.0000)
- Mask High-MI Heads: **0.6452** (Δ 0.0000)

## Important Heads
Top MI heads: **[3, 4, 2]**

## Emotion-wise Attention

- Angry: Head 4 → mouth/jaw (secondary: nose/cheeks), right face
- Disgust: Head 2 → mouth/jaw (secondary: nose/cheeks), center face
- Fear: Head 3 → mouth/jaw (secondary: nose/cheeks), right face
- Happy: Head 4 → mouth/jaw (secondary: nose/cheeks), center face
- Sad: Head 2 → mouth/jaw (secondary: nose/cheeks), center face
- Surprise: Head 1 → mouth/jaw (secondary: nose/cheeks), right face
- Neutral: Head 3 → mouth/jaw (secondary: nose/cheeks), right face

## Insights

- Attention is concentrated mainly on **mouth/jaw regions**
- A small subset of heads carries most useful information
- Ablation impact: Weak separation between heads