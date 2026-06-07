# Model Performance Comparison Summary

This document summarizes the performance evaluation of the two deep learning architectures tested on the holdout test split (15%) of the NEU Metal Surface Defect Database.

## Evaluation Metrics Table

| Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 92.59% | 0.9284 | 0.9259 | 0.9252 | ~3.08 minutes |
| **MobileNetV2** | **98.15%** | **0.9824** | **0.9815** | **0.9815** | ~5.75 minutes |

### Analysis
* **MobileNetV2** significantly outperformed the Custom CNN across all classification metrics. The pre-trained ImageNet weights provided highly robust feature extractors for recognizing industrial textures.
* **Custom CNN** trained much faster per epoch, resulting in a lower overall training duration, but struggled slightly with the more complex defect classes, resulting in a lower F1 score.

Both models generalize well, but MobileNetV2 is recommended for the final factory deployment due to its superior accuracy.
