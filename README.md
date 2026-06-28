# Industrial Defect Detection: Metal Surface Defect Classification

**Author:** Dimple Parmar  
**Course:** Machine Learning

## 📝 Phase 2 Submission Links
* **GitHub Repository:** [https://github.com/dimpleparmar35/Metal_Surface_Defect_Classification](https://github.com/dimpleparmar35/Metal_Surface_Defect_Classification)
* **Kaggle Notebook:** [https://www.kaggle.com/code/dimpleparmar23/industrial-defect-detection](https://www.kaggle.com/code/dimpleparmar23/industrial-defect-detection)

---

This repository contains a complete, end-to-end deep learning project designed to automate the quality control inspection of hot-rolled steel strips. Using a Custom Convolutional Neural Network (CNN) trained from scratch and a pre-trained MobileNetV2 utilizing transfer learning and fine-tuning, this system classifies metal surface defects into six distinct categories. To address the "black box" nature of deep learning, the pipeline integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize defect-specific network activations.

---

## Dataset Information
The project utilizes the **NEU Metal Surface Defect Database**:
* **Total Samples**: 1,800 grayscale images (300 images per class, perfectly balanced).
* **Resolution**: 200x200 pixels in BMP format.
* **Defect Classes (6)**: 
  * Crazing (`crazing`)
  * Inclusion (`inclusion`)
  * Patches (`patches`)
  * Pitted Surface (`pitted_surface`)
  * Rolled-in Scale (`rolled-in_scale`)
  * Scratches (`scratches`)
* **Kaggle Link**: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

---

## Folder Structure
```text
Metal_Surface_Defect_Classification/
├── data/                       # Dataset directory (download separately)
│   └── raw/                    # Place class folders (crazing, scratches, etc.) here
├── notebooks/
│   └── Metal_Surface_Defect_Classification.ipynb  # Self-contained Kaggle notebook
├── src/                        # Modular source code
│   ├── preprocessing.py        # Programmatic stratified split (70/15/15) & generators
│   ├── model.py                # Custom CNN & MobileNetV2 architecture builders
│   ├── train.py                # Training loop, callbacks, and execution time logging
│   ├── evaluate.py             # Computes metrics, plots ROC/CM, generates comparison table
│   ├── gradcam.py              # Gradient-weighted Class Activation Mapping (Grad-CAM) XAI
│   └── utils.py                # folder verification, seeding, and learning curves plotting
├── outputs/                    # Output directory (auto-created)
│   ├── figures/                # Learning curves, confusion matrices, ROC, comparison table
│   └── models/                 # Saved Keras model files (.keras) and training times
├── report/                     # Academic documentation
│   ├── proposal.md             # Formal Project Proposal (IEEE format references)
│   └── final_report.md         # Final Research Paper (IEEE format references)
├── README.md                   # Project documentation (this file)
└── requirements.txt            # Python environment dependencies
```

---

## Installation & Setup

1. **Clone this repository**:
   ```bash
   git clone <your-repository-url>
   cd Metal_Surface_Defect_Classification
   ```

2. **Set up a Python Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Acquire the Dataset**:
   * Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).
   * Unpack the archive and move the class subdirectories so that the images are placed in:
     `data/raw/` (e.g., `data/raw/crazing/crazing_1.bmp`).

---

## Usage Instructions

### 1. Model Training
To train the Custom CNN or MobileNetV2 model, run `src/train.py` from the project root. The scripts automatically create directories, seed the run for reproducibility, and log training time.

```bash
# Train the Custom CNN model
python src/train.py --model custom --epochs 30 --batch_size 32

# Train the MobileNetV2 model (Feature extraction + Fine-tuning)
python src/train.py --model mobilenetv2 --epochs 30 --batch_size 32
```
Trained model weights (`.keras` format) and training duration metrics are written to `outputs/models/`.

### 2. Model Evaluation
Evaluate the models on the holdout test split (15% of the total dataset) and generate confusion matrices, ROC curves, and the final model comparison table.

```bash
# Evaluate the Custom CNN model
python src/evaluate.py --model_path outputs/models/custom_cnn.keras

# Evaluate the MobileNetV2 model
python src/evaluate.py --model_path outputs/models/mobilenetv2.keras

# Auto-evaluate both models and compile the performance comparison table
python src/evaluate.py
```
Output charts are saved to `outputs/figures/`.

### 3. Model Explainability (Grad-CAM)
Visualize which region of interest (ROI) on a defect image drove the model's classification:

```bash
# Run Grad-CAM overlay on a specific defect image using a trained model
python src/gradcam.py --model_path outputs/models/mobilenetv2.keras --image_path data/raw/scratches/scratches_1.bmp
```
Heatmaps are saved to `outputs/figures/gradcam_mobilenetv2_scratches_1.bmp`.

---

## Experimental Results

The models were benchmarked on a stratified 15% holdout test set (270 images). Below is the performance summary:

| Architecture | Test Accuracy | Test Precision (Weighted) | Test Recall (Weighted) | Test F1 Score (Weighted) | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 16.67% | 0.0288 | 0.1667 | 0.0492 | ~3.14 minutes |
| **MobileNetV2** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | ~2.65 minutes |

### Visual Deliverables (`outputs/figures/`)
* **Learning Curves**: Show loss and accuracy optimization for training and validation splits.
* **Confusion Matrices**: Heatmap illustrating prediction distributions across classes.
* **ROC Curves**: One-vs-Rest ROC curves verifying AUC values $\ge 0.99$ for MobileNetV2.
* **Grad-CAM Activations**: Heatmap overlays showing model focus matches the actual physical coordinates of the defects.

---

## Future Improvements
* Deploy the models to edge computing platforms (e.g. Raspberry Pi or NVIDIA Jetson) for real-time inspection.
* Transition from image classification to object detection (e.g. YOLOv8) to localize and estimate sizing for multiple defects in a single sheet.
* Generate synthetic training data using Generative Adversarial Networks (GANs) to expand samples for rarer classes of defects.
* Build a Streamlit or Gradio dashboard for user-friendly factory quality control interface.
