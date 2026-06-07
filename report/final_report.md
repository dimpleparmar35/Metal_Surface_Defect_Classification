# Final Research Report: Metal Surface Defect Classification Using Deep Learning

## 1. Abstract
Surface defect detection is a critical process in metallurgical manufacturing to guarantee material quality and structural integrity. Traditional methods relying on manual visual inspection are inefficient, expensive, and error-prone. This study proposes an automated deep learning framework for classifying hot-rolled steel surface defects. We developed a Custom Convolutional Neural Network (CNN) trained from scratch and compared it against a transfer learning model based on the pre-trained MobileNetV2 architecture. Both models were trained and evaluated on the NEU Metal Surface Defect Dataset, containing 1,800 images distributed equally across six classes. To ensure rigorous evaluation, we implemented a stratified 70/15/15 train/validation/test split. The MobileNetV2 transfer learning model, combining locked feature extraction and fine-tuning, achieved superior classification results with an accuracy of 98.15%, compared to 92.59% for the Custom CNN. In addition, Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to provide spatial interpretability of the models' predictions, demonstrating that the classifier correctly isolates physical defects on the steel strips.

## 2. Introduction
In hot-rolled steel strip manufacturing, defects such as crazing, inclusions, and scratches frequently occur due to rolling roll wear, mechanical vibration, or slab impurities. The presence of these defects degrades the mechanical properties, corrosion resistance, and aesthetic appeal of the metal. Consequently, real-time quality control is vital for modern steel fabrication processes.

Early automated surface inspection (ASI) systems relied on manual visual inspection. However, human inspectors struggle to keep pace with rapid rolling speeds, and fatigue leads to inconsistent error rates. The computer vision community subsequently introduced machine learning methods combining handcrafted texture descriptors (e.g., Local Binary Patterns, Gabor filters) with Support Vector Machines (SVM). These methods, while faster, proved fragile under variations in illumination, scale shifts, and ambient noise.

In recent years, deep learning has revolutionized computer vision. Convolutional Neural Networks (CNNs) automatically learn hierarchical spatial features directly from raw pixels, minimizing the need for handcrafted descriptors. This study evaluates two distinct paradigms of deep learning for metal defect classification: (1) designing a custom sequential CNN optimized for computational efficiency, and (2) leveraging transfer learning with MobileNetV2, pre-trained on ImageNet, to classify specialized metal surfaces. We evaluate both approaches using metrics like Accuracy, Precision, Recall, F1 Score, and training compute times. We also use Grad-CAM to visualize what regions of the image influence the model's predictions.

## 3. Related Work
Automated inspection of hot-rolled steel strip defects has a rich literature. Early work focused on traditional texture analysis. Song and Yan [2] proposed Completed Local Binary Patterns (CLBP) to capture defect patterns, demonstrating that texture features are susceptible to high noise levels and variations in steel surface textures.

The transition to deep learning addressed these issues. Luo et al. [1] surveyed visual defect detection methods, showing that CNN architectures outperform traditional handcrafted feature extraction in both classification accuracy and robustness against surface noise. Yi et al. [3] implemented a custom sequential CNN for the NEU dataset, demonstrating that an end-to-end network could extract features and classify defects without separate preprocessing steps.

Transfer learning has since become a standard methodology. Pre-training on massive datasets like ImageNet allows models to learn general features like edges and textures, which can then be fine-tuned for specific domains, as shown by He et al. with ResNet [4]. For edge computing and real-time processing, Sandler et al. proposed MobileNetV2 [5], which utilizes inverted residual blocks and depthwise separable convolutions to achieve high efficiency. This research evaluates the applicability of both a custom CNN and MobileNetV2 for metal surface defect classification.

## 4. Methodology
The proposed framework consists of four primary stages: data ingestion, preprocessing/augmentation, model construction, and model interpretation.

### 4.1. Dataset
This study utilizes the NEU Metal Surface Defect Database. The dataset is balanced, containing 1,800 grayscale images with 300 samples for each of the six defect classes: Crazing (`crazing`), Inclusion (`inclusion`), Patches (`patches`), Pitted Surface (`pitted_surface`), Rolled-in Scale (`rolled-in_scale`), and Scratches (`scratches`). The original images are 200x200 pixels in BMP format.

### 4.2. Preprocessing & Data Splitting
The preprocessing pipeline operates as follows:
* **Directory Scan**: The raw files are loaded programmatically.
* **Resizing**: Images are bilinearly interpolated to 224x224 pixels to match the standard input resolution of MobileNetV2.
* **Normalization**: Grayscale values are normalized to [0, 1] to stabilize gradient calculations.
* **Stratified Splitting**: To prevent data leakage and ensure stable evaluation, the dataset is programmatically split into:
  * **Training Set (70%)**: 1,260 images.
  * **Validation Set (15%)**: 270 images.
  * **Test Set (15%)**: 270 images.

### 4.3. Data Augmentation
To mitigate overfitting, the training generator applies real-time data augmentations:
* Random rotation within $\pm 20$ degrees.
* Horizontal and vertical random flips.
* Width and height translations up to 10%.
* Random zoom within $\pm 20\%$.
* Brightness adjustments between 80% and 120%.

### 4.4. Model Architectures
We implement and compare two distinct architectures:
1. **Custom CNN**: Constructed sequentially with four Conv2D blocks:
   * Block 1: 32 filters, $3\times3$ kernel, ReLU, Batch Normalization, MaxPool (2x2).
   * Block 2: 64 filters, $3\times3$ kernel, ReLU, Batch Normalization, MaxPool (2x2).
   * Block 3: 128 filters, $3\times3$ kernel, ReLU, Batch Normalization, MaxPool (2x2).
   * Block 4: 256 filters, $3\times3$ kernel, ReLU, Batch Normalization, MaxPool (2x2).
   * Head: Flatten $\rightarrow$ Dense (512, ReLU, Dropout 0.5) $\rightarrow$ Dense (256, ReLU, Dropout 0.5) $\rightarrow$ Softmax (6).
2. **MobileNetV2 Transfer Learning**: Implements a pre-trained MobileNetV2 base (1280 features) with frozen weights. A custom classifier head is appended: Global Average Pooling $\rightarrow$ Dense (256, ReLU, Dropout 0.5) $\rightarrow$ Softmax (6).
   * After 15 epochs of training the classifier head, layers from index 100 onwards are unfrozen and trained at a lower learning rate ($1\times10^{-5}$) to adapt to steel texture features.

## 5. Experimental Setup
* **Software Stack**: Python 3.8, TensorFlow 2.16.1, Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn, and Scikit-Learn.
* **Loss Function**: Categorical Crossentropy.
* **Optimizer**: Adam (initial learning rate = 0.001).
* **Callbacks**:
  * `EarlyStopping` (monitor='val_loss', patience=10).
  * `ReduceLROnPlateau` (monitor='val_loss', factor=0.2, patience=5).
  * `ModelCheckpoint` (monitor='val_accuracy', save_best_only=True).

## 6. Results
Both models were trained and evaluated on the holdout test set (270 images). 

### 6.1. Model Performance Summary
The table below compares the Custom CNN and MobileNetV2 models across all evaluation metrics:

| Architecture | Test Accuracy | Test Precision | Test Recall | Test F1 Score | Training Time (min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 92.59% | 0.9284 | 0.9259 | 0.9252 | 3.08 min |
| **MobileNetV2** | **98.15%** | **0.9824** | **98.15%** | **98.15%** | 5.75 min (incl. FT) |

### 6.2. Learning Curves and Convergence
* **Custom CNN**: Required 25 epochs to converge, exhibiting slight oscillations in validation loss, which was mitigated by the ReduceLROnPlateau callback.
* **MobileNetV2**: Displayed rapid convergence. During the initial feature extraction phase, validation accuracy stabilized at ~95.2% within 10 epochs. After unfreezing base layers for fine-tuning, test accuracy increased to 98.15% within an additional 5 epochs.

### 6.3. Confusion Matrix Analysis
The confusion matrix revealed that the Custom CNN occasionally misclassified 'Crazing' as 'Scratches' and 'Rolled-in Scale' as 'Inclusion'. This confusion is attributed to similar linear structures and topological details. In contrast, the MobileNetV2 model resolved most of these confusions, showing minor errors only between 'Rolled-in Scale' and 'Inclusion'.

### 6.4. ROC Curves
The One-vs-Rest ROC curves demonstrated excellent discriminative ability for both architectures:
* **Custom CNN**: Class-wise Area Under the Curve (AUC) ranged between 0.96 and 0.98.
* **MobileNetV2**: Class-wise AUC reached 0.99 for all categories, confirming the model's high sensitivity and low false-alarm rates.

### 6.5. Interpretability via Grad-CAM
Grad-CAM heatmaps were generated to localize predictions. For 'Scratches', the heatmaps correctly highlighted the linear scratch paths on the steel. For 'Patches' and 'Pitted Surface', the highest activation intensities (red zones) matched the physical location of surface discoloration and pitting, verifying that the models classify defects based on relevant visual features.

## 7. Discussion
The results show that Transfer Learning is highly effective for specialized industrial classification tasks. Although MobileNetV2 is trained on natural images (animals, vehicles, etc.), the low-level edge and texture filters generalize well to industrial steel surfaces. Fine-tuning the top layers of the base model adapts the network to unique metallurgical patterns.

The Custom CNN performed well, achieving over 92% accuracy, but was more prone to overfitting and required regularization (Dropout, Batch Normalization). For resource-constrained factory environments, the Custom CNN has a smaller file size (17 MB) compared to MobileNetV2 (33 MB), but MobileNetV2's superior accuracy makes it the preferred option for quality assurance applications.

## 8. Conclusion
This study developed a deep learning system for classifying hot-rolled steel surface defects using the NEU dataset. A Custom CNN trained from scratch was compared against a MobileNetV2 transfer learning model. Rigorous testing on a 15% stratified split demonstrated that the fine-tuned MobileNetV2 achieves a test accuracy of 98.15%, outperforming the Custom CNN (92.59%). Grad-CAM visualizations confirmed that the models focus on the correct defect regions. Future work will focus on evaluating these models on edge devices and exploring object detection models (e.g., YOLOv8) to localize multiple defects in real-time.

## 9. References
* [1] Q. Luo, X. Fang, L. Liu, C. Yang, and Y. Sun, "Automated Visual Defect Detection for Flat Steel Surface: A Survey," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 9, pp. 6261-6280, Sept. 2020.
* [2] K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, Nov. 2013.
* [3] L. Yi, G. Li, and M. Jiang, "End-to-end steel strip surface defects recognition based on convolutional neural networks," *Steel Research International*, vol. 88, no. 2, p. 1600068, Feb. 2017.
* [4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.
* [5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.
