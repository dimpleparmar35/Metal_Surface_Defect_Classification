# Project Proposal: Industrial Defect Detection - Metal Surface Defect Classification Model

## 1. Project Title
Industrial Defect Detection: Metal Surface Defect Classification Model

## 2. Abstract
The automation of quality control in industrial manufacturing is essential to improve production efficiency and ensure product reliability. This project proposes a deep learning-based image classification system to automatically detect and classify defects on hot-rolled steel surfaces. Utilizing the NEU Metal Surface Defect Dataset, we design a Custom Convolutional Neural Network (CNN) and implement a Transfer Learning approach using MobileNetV2. Both models will be evaluated and compared on a holdout test split (15%) across key metrics, including Accuracy, Precision, Recall, F1 Score, and training computation time. Explainable AI via Gradient-weighted Class Activation Mapping (Grad-CAM) will be utilized to localize the defect predictions. The ultimate objective is to determine the optimal model balancing accuracy and real-time processing capabilities for factory-floor integration.

## 3. Background and Motivation
In the metallurgical industry, surface defects on hot-rolled steel strips severely impact the mechanical strength and visual appearance of the final product. Historically, quality control has relied on manual visual inspection by human inspectors. However, manual inspection is labor-intensive, slow, subjective, and highly prone to fatigue-induced errors, leading to inconsistent quality assurance. 

With recent advancements in computer vision and artificial intelligence, automated surface inspection (ASI) systems have emerged as a viable solution [1]. Using deep learning architectures, automated inspection systems can achieve high defect-detection accuracy and real-time processing speeds, making them suitable for modern automated production lines. Automating this process reduces scrap rates, prevents defective material from reaching customers, and provides real-time feedback to control the rolling mill process.

## 4. Problem Statement
The primary challenge is the automatic and accurate classification of various metal surface defects that vary in scale, shape, and contrast under varying industrial lighting conditions. The classification system must map a grayscale image of a metal surface to one of six predefined defect categories: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, or Scratches. The system needs to minimize false negatives (to prevent defective products from escaping) and false positives (to avoid unnecessary material rejection).

## 5. Project Objectives
* **Data Processing**: Preprocess and apply data augmentation to the NEU dataset, establishing a stratified training (70%), validation (15%), and testing (15%) split.
* **Custom Model Development**: Design, implement, and train a Custom CNN architecture from scratch using TensorFlow/Keras.
* **Transfer Learning**: Implement a transfer learning pipeline using a pre-trained MobileNetV2 architecture, including a custom classifier head and a fine-tuning phase.
* **Comparative Performance Analysis**: Benchmark both models on the test split and compile a comparison table based on Accuracy, Precision, Recall, F1 Score, and Training Duration.
* **Model Interpretability**: Apply Grad-CAM to visualize the region of interests (ROIs) that drive the models' predictions, ensuring physical defects align with class activations.

## 6. Research Questions
1. How does a custom-built CNN compare to a pre-trained, fine-tuned MobileNetV2 model in classification accuracy and computational training efficiency on hot-rolled steel images?
2. Which data augmentation techniques (e.g., rotation, translation, zoom, brightness adjustments) yield the most significant improvement in model generalization?
3. Can Grad-CAM heatmaps reliably localize and delineate the boundaries of physical metal surface defects?

## 7. Literature Review
* **[1] Q. Luo, X. Fang, L. Liu, C. Yang, and Y. Sun**, "Automated Visual Defect Detection for Flat Steel Surface: A Survey," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 9, pp. 6261-6280, Sept. 2020.
  * *Summary*: This survey reviews traditional image processing and modern deep learning methods for steel surface defect detection. It highlights the shift from manual feature descriptors (such as Gabor filters and Local Binary Patterns) to convolutional neural networks (CNNs), concluding that deep learning methods exhibit superior robustness to noise and lighting variations.
* **[2] K. Song and Y. Yan**, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, Nov. 2013.
  * *Summary*: The authors establish a classic baseline by applying Completed Local Binary Patterns (CLBP) for hot-rolled steel strip defect classification. Their findings highlight that traditional texture features struggle significantly with noise, scale shifts, and lighting imbalances, emphasizing the need for robust feature extraction.
* **[3] L. Yi, G. Li, and M. Jiang**, "End-to-end steel strip surface defects recognition based on convolutional neural networks," *Steel Research International*, vol. 88, no. 2, p. 1600068, Feb. 2017.
  * *Summary*: This paper proposes an end-to-end CNN classification framework evaluated on the NEU dataset. The network automatically extracts hierarchical features directly from raw pixels, outperforming support vector machine (SVM) classifiers and proving that custom CNNs can classify metal defects without manual feature engineering.
* **[4] K. He, X. Zhang, S. Ren, and J. Sun**, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.
  * *Summary*: This seminal work introduces residual connections (ResNet) to train deeper architectures. It establishes the foundations of transfer learning, demonstrating that representations learned on massive image databases (like ImageNet) generalize exceptionally well to specialized domain tasks.
* **[5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. C. Chen**, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.
  * *Summary*: The authors present MobileNetV2, which uses inverted residuals and linear bottlenecks to construct a highly efficient, lightweight CNN. This model is key for industrial edge deployment, where high inference speed and a small memory footprint are required.

## 8. Dataset Selection
* **Dataset Name**: NEU Metal Surface Defect Database
* **Dataset Source and Download Link**: [Kaggle - NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
* **Dataset Description**: The dataset contains grayscale images of hot-rolled steel strips, showing six typical kinds of surface defects.
* **Number of Images**: 1,800 images
* **Number of Classes**: 6 classes
* **Class Names**: Crazing (`crazing`), Inclusion (`inclusion`), Patches (`patches`), Pitted Surface (`pitted_surface`), Rolled-in Scale (`rolled-in_scale`), and Scratches (`scratches`)
* **Image Types and Resolution**: Grayscale BMP format, 200x200 pixel resolution.

### Class Distribution Table
| Defect Class | Number of Images | Percentage |
| :--- | :---: | :---: |
| Crazing | 300 | 16.67% |
| Inclusion | 300 | 16.67% |
| Patches | 300 | 16.67% |
| Pitted Surface | 300 | 16.67% |
| Rolled-in Scale | 300 | 16.67% |
| Scratches | 300 | 16.67% |
| **Total** | **1,800** | **100.0%** |

## 9. Proposed Methodology

### Data Collection
The NEU dataset will be structured locally in a `data/raw/` directory, organized into class subdirectories named after the defect types.

### Data Preprocessing
* **Resizing**: Images will be resized from their native 200x200 pixels to 224x224 pixels to match the standard input dimensions expected by pre-trained ImageNet architectures, specifically MobileNetV2.
* **Normalizing**: Grayscale values in [0, 255] will be scaled to [0.0, 1.0] to stabilize backpropagation gradients.
* **Splitting**: Programmatic stratified split to partition the 1,800 images into:
  * **Training Set**: 70% (1,260 images)
  * **Validation Set**: 15% (270 images)
  * **Test Set**: 15% (270 images)

### Data Augmentation
To prevent overfitting on the small dataset, the training subset will undergo real-time augmentation:
* Random rotation within $\pm 20$ degrees.
* Horizontal and vertical random flips (since steel defects have no semantic top-bottom orientation).
* Brightness adjustments between 80% and 120% of original values.
* Random zoom within $\pm 20\%$.
* Width and height translations of up to 10%.

### CNN Architecture Design
The Custom CNN is designed sequentially with the following layers:
* Four Convolutional Blocks:
  * Conv2D (32, 64, 128, 256 filters, $3\times3$ kernels, ReLU activation) to build hierarchical representation.
  * Batch Normalization after each convolution to reduce internal covariate shift.
  * MaxPooling2D ($2\times2$ stride) to downsample spatial dimensions.
* Classification Head:
  * Flatten layer to vectorize features.
  * Dense (512 units, ReLU) + Dropout (0.5).
  * Dense (256 units, ReLU) + Dropout (0.5).
  * Output Dense (6 units, Softmax) to output class posterior probabilities.

### Transfer Learning Architecture
The pre-trained MobileNetV2 [5] model will be integrated using a two-stage training scheme:
1. **Feature Extraction**: Freezing the base MobileNetV2 convolutional layers, appending a Global Average Pooling 2D layer, a Dense layer (256 units, ReLU), Dropout (0.5), and a Softmax output layer.
2. **Fine-Tuning**: Unfreezing layers from index 100 onwards and training the weights using a very low learning rate ($1\times10^{-5}$) to adapt the model to specialized steel textures.

### Training Strategy
* **Optimizer**: Adam (learning rate = 0.001 for training head, 0.00001 for fine-tuning).
* **Callbacks**:
  * `ModelCheckpoint`: Save the model with the highest validation accuracy.
  * `EarlyStopping`: Halt training if validation loss fails to decrease for 10 consecutive epochs.
  * `ReduceLROnPlateau`: Halve the learning rate if validation loss plateaus for 5 epochs.

### Evaluation Metrics
Performance on the holdout test set will be evaluated using:
* Accuracy: Total proportion of correct predictions.
* Precision, Recall, and F1 Score (Weighted): Accounts for multi-class classification performance.
* Confusion Matrix: Heatmap visualizing class confusion (e.g. Crazing vs. Scratches).
* ROC Curves and AUC: Measures true positive vs. false positive rate across decision thresholds.
* Training Time: Logs the computational training cost in minutes.

### Expected Results
It is expected that MobileNetV2 will achieve superior classification metrics (Accuracy $\ge 96\%$) due to the deep pre-trained visual primitives. The Custom CNN is expected to achieve acceptable metrics (Accuracy $\ge 90\%$) but might require more epochs to stabilize, and its training time per epoch is expected to be slightly higher than the frozen MobileNetV2 base phase.

### Expected Figures
* Training and validation loss/accuracy curves for both models.
* Confusion matrix heatmaps showing misclassification clusters.
* Multiclass ROC curves displaying One-vs-Rest AUC.
* Grad-CAM activation overlays on raw steel defect BMPs.
* Final comparative bar charts or table outputs.

### Expected Tables
* Performance Comparison Table comparing Custom CNN vs. MobileNetV2.

## 10. Project Timeline
| Phase | Task Description | Estimated Duration |
| :--- | :--- | :---: |
| Phase 1 | Literature Review, Dataset Download, Setup Repo | Week 1 |
| Phase 2 | Preprocessing Pipeline & Stratified Data Splitting | Week 2 |
| Phase 3 | Designing & Training Custom CNN Architecture | Week 3 |
| Phase 4 | Transfer Learning & Fine-tuning MobileNetV2 | Week 4 |
| Phase 5 | Evaluation, Metric Comparisons, and Grad-CAM Testing | Week 5 |
| Phase 6 | Report Writing & Final Code Review | Week 6 |

## 11. References
* [1] Q. Luo, X. Fang, L. Liu, C. Yang, and Y. Sun, "Automated Visual Defect Detection for Flat Steel Surface: A Survey," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 9, pp. 6261-6280, Sept. 2020.
* [2] K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, Nov. 2013.
* [3] L. Yi, G. Li, and M. Jiang, "End-to-end steel strip surface defects recognition based on convolutional neural networks," *Steel Research International*, vol. 88, no. 2, p. 1600068, Feb. 2017.
* [4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.
* [5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.
