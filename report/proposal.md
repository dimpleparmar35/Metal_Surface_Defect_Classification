# Project Proposal: Industrial Defect Detection - Metal Surface Defect Classification Model

## 1. Project Title
Industrial Defect Detection: Metal Surface Defect Classification Model

## 2. Abstract
The automation of quality control in industrial manufacturing is essential to improve production efficiency and ensure product reliability. This project proposes a deep learning-based image classification system to automatically detect and classify defects on metal surfaces. Utilizing the NEU Metal Surface Defect Dataset, we will design a Custom Convolutional Neural Network (CNN) and implement a Transfer Learning approach using MobileNetV2. The performance of both models will be evaluated and compared to determine the most effective architecture for real-time defect classification.

## 3. Background and Motivation
In the metallurgical industry, surface defects on rolled steel or other metal surfaces severely impact product quality. Traditional manual inspection is labor-intensive, time-consuming, and highly subjective, leading to inconsistent quality assurance. With the advent of deep learning and computer vision, automated surface defect inspection systems have become viable, offering high accuracy and real-time processing capabilities. This project aims to bridge the gap between academic research and industrial application by building a robust classification model.

## 4. Problem Statement
The primary challenge is the accurate identification and classification of various metal surface defects that vary in size, shape, and texture, often under challenging lighting conditions. The system must automatically map a given image of a metal surface to one of six predefined defect categories with high precision and recall.

## 5. Project Objectives
* To preprocess and augment a dataset of metal surface defects to improve model generalization.
* To design, train, and optimize a custom CNN architecture tailored for surface defect classification.
* To implement a Transfer Learning strategy using a pre-trained MobileNetV2 model.
* To evaluate and compare both models using comprehensive metrics (Accuracy, Precision, Recall, F1 Score).
* To provide visual explanations of the model's predictions using Grad-CAM.

## 6. Research Questions
1. How does a custom CNN compare to a fine-tuned MobileNetV2 model in terms of accuracy and training efficiency for metal defect classification?
2. Which data augmentation techniques yield the most significant improvement in model generalization?
3. Can Grad-CAM visualizations accurately highlight the physical defects on the metal surfaces that drive the model's decisions?

## 7. Literature Review
1. **Luo, Q., et al. (2020)** "Automated Visual Defect Detection for Flat Steel Surface: A Survey." IEEE Transactions on Instrumentation and Measurement. *Summary:* This paper provides a comprehensive review of traditional and deep learning methods for steel surface defect detection, highlighting the shift towards CNNs for improved accuracy.
2. **Song, K., et al. (2013)** "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects." Applied Surface Science. *Summary:* Discusses traditional feature extraction methods like LBP, establishing a baseline for the challenges posed by noise in industrial images.
3. **Yi, L., et al. (2017)** "End-to-end steel strip surface defects recognition based on convolutional neural networks." Steel Research International. *Summary:* Proposes an end-to-end CNN approach that outperforms traditional SVM and feature extraction methods on the NEU dataset.
4. **He, K., et al. (2016)** "Deep residual learning for image recognition." CVPR. *Summary:* Introduces ResNet, forming the foundational understanding of transfer learning and deep feature extraction used in modern defect detection systems.
5. **Sandler, M., et al. (2018)** "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR. *Summary:* Details the architecture of MobileNetV2, which is critical for understanding the lightweight transfer learning model proposed in this study.

## 8. Dataset Selection
* **Dataset Name:** NEU Metal Surface Defect Dataset
* **Dataset Source and Download Link:** [Kaggle - NEU Metal Surface Defects Data](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) (Note: Hosted on various Kaggle repositories and Northeastern University database).
* **Dataset Description:** The dataset contains grayscale images of hot-rolled steel strips with six typical surface defects.
* **Number of Images:** 1,800
* **Number of Classes:** 6
* **Class Names:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
* **Image Types and Resolution:** Grayscale BMP images, 200x200 pixels.

### Class Distribution Table
| Defect Class | Number of Images | Percentage |
| :--- | :--- | :--- |
| Crazing | 300 | 16.67% |
| Inclusion | 300 | 16.67% |
| Patches | 300 | 16.67% |
| Pitted Surface | 300 | 16.67% |
| Rolled-in Scale | 300 | 16.67% |
| Scratches | 300 | 16.67% |

## 9. Proposed Methodology

### Data Collection
The dataset will be downloaded and structured into a local directory (`data/raw/`) grouped by class folders.

### Data Preprocessing
* Resizing images to 224x224 to match the standard input sizes of common CNN architectures (like MobileNetV2).
* Normalizing pixel values to the range [0, 1].
* Splitting the dataset into 80% training and 20% validation sets.

### Data Augmentation
To prevent overfitting, the following augmentations will be applied to the training set:
* Rotation (up to 20 degrees)
* Horizontal and Vertical Flips
* Zoom (up to 20%)
* Brightness Adjustment (0.8 to 1.2)
* Width and Height Shifts (up to 10%)

### CNN Architecture Design
A Custom CNN will be constructed containing:
* Multiple Conv2D layers (32, 64, 128, 256 filters) with ReLU activation.
* Batch Normalization after convolutions to stabilize training.
* MaxPooling2D layers for spatial downsampling.
* Flattening layer followed by Dense layers (512, 256 neurons) with Dropout (0.5).
* Softmax output layer with 6 neurons.

### Transfer Learning Architecture
* **Base Model:** MobileNetV2 pre-trained on ImageNet.
* **Modification:** Base layers will initially be frozen. A GlobalAveragePooling2D layer will replace the fully connected layers, followed by a custom Dense classification head with Dropout.
* **Fine-Tuning:** After initial training, the top layers of the base model will be unfrozen and trained with a lower learning rate.

### Training Strategy
* **Optimizer:** Adam (Initial LR = 0.001)
* **Callbacks:** ModelCheckpoint (save best weights), EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.2).

### Evaluation Metrics
The models will be evaluated using Scikit-Learn metrics:
* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1 Score (Weighted)
* Confusion Matrix
* Classification Report

### Expected Results
It is expected that the MobileNetV2 model will achieve higher accuracy and converge faster due to its pre-learned feature representations. The custom CNN is expected to achieve >90% accuracy but may require more epochs.

### Expected Figures
* Training & Validation Accuracy Curves
* Training & Validation Loss Curves
* Confusion Matrix Heatmap
* Multiclass ROC Curves
* Grad-CAM Visualizations

### Expected Tables
* Model Comparison Table (Custom CNN vs MobileNetV2: Accuracy, Precision, Recall, F1, Time).

## 10. Project Timeline
| Phase | Task Description | Estimated Time |
| :--- | :--- | :--- |
| Phase 1 | Literature Review & Dataset Preparation | 1 Week |
| Phase 2 | Data Preprocessing & Augmentation | 1 Week |
| Phase 3 | Model Implementation (CNN & Transfer Learning) | 2 Weeks |
| Phase 4 | Model Training & Hyperparameter Tuning | 1 Week |
| Phase 5 | Evaluation, Analysis (ROC, Grad-CAM) & Comparison | 1 Week |
| Phase 6 | Documentation and Final Report Writing | 1 Week |

## 11. References
1. Luo, Q., Fang, X., Liu, L., Yang, C., & Sun, Y. (2020). Automated Visual Defect Detection for Flat Steel Surface: A Survey. *IEEE Transactions on Instrumentation and Measurement*, 69(9), 6261-6280.
2. Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. *Applied Surface Science*, 285, 858-864.
3. Yi, L., Li, G., & Jiang, M. (2017). End-to-end steel strip surface defects recognition based on convolutional neural networks. *Steel Research International*, 88(2), 1600068.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).
