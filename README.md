# Industrial Defect Detection: Metal Surface Defect Classification

## Project Overview
This project presents a Convolutional Neural Network (CNN) based image classification system designed to detect and classify defects on metal surfaces. Quality control in industrial manufacturing often relies on manual inspection, which can be prone to human error and inefficiency. By automating the detection of surface defects using deep learning architectures—specifically a Custom CNN and a pretrained MobileNetV2 using transfer learning—this project aims to enhance inspection accuracy and speed.

## Dataset Information
The project utilizes the **NEU Metal Surface Defect Dataset** (Northeastern University).
* **Total Images:** 1,800 grayscale images
* **Resolution:** 200x200 pixels
* **Classes:** 6 distinct defect types
* **Class Names:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
* **Distribution:** 300 images per class (perfectly balanced)

## Folder Structure
```text
Metal_Surface_Defect_Classification/
├── data/                       # Contains the raw NEU dataset
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   └── Metal_Surface_Defect_Classification.ipynb
├── src/                        # Source code for the deep learning pipeline
│   ├── preprocessing.py        # Data loading and augmentation
│   ├── model.py                # CNN and MobileNetV2 architectures
│   ├── train.py                # Model training scripts
│   ├── evaluate.py             # Evaluation metrics and reporting
│   ├── gradcam.py              # Explainable AI (Grad-CAM)
│   └── utils.py                # Helper functions for visualization
├── outputs/                    # Generated files
│   ├── figures/                # Accuracy/Loss curves, ROC, Confusion Matrix
│   └── models/                 # Saved model weights (.keras / .h5)
├── report/                     # Documentation and proposals
│   ├── proposal.md
│   └── final_report.md
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Installation Instructions
1. Clone this repository:
   ```bash
   git clone <your_repository_url>
   cd Metal_Surface_Defect_Classification
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the **NEU Metal Surface Defect Dataset** and place the organized class folders into the `data/raw/` directory.

## Training Instructions
To train the models from scratch, execute the `train.py` script from the `src` directory. You can specify the model type (either `custom` or `mobilenetv2`).

```bash
# Train Custom CNN
python src/train.py --model custom --epochs 30 --batch_size 32

# Train MobileNetV2 (Transfer Learning)
python src/train.py --model mobilenetv2 --epochs 30 --batch_size 32
```
Models will automatically be saved to `outputs/models/`.

## Evaluation Instructions
To evaluate a trained model and generate metrics (Accuracy, Precision, Recall, F1 Score, Classification Report, Confusion Matrix, and ROC curves), run:

```bash
# Evaluate a specific model
python src/evaluate.py --model_path outputs/models/custom_cnn.keras

# Generate Grad-CAM Visualizations
python src/gradcam.py --model_path outputs/models/custom_cnn.keras --image_path data/raw/Crazing/crazing_1.bmp
```
Visualizations and metrics will be saved in `outputs/figures/`.

## Results
*To be updated after training.*
A comparison study between the Custom CNN and MobileNetV2 architectures is conducted based on:
* Accuracy
* Precision
* Recall
* F1 Score
* Training Time

## Future Improvements
* Implementation of advanced architectures like EfficientNet or Vision Transformers (ViT).
* Real-time deployment using edge devices (e.g., Raspberry Pi or Jetson Nano).
* Expanding the dataset with synthetic data using Generative Adversarial Networks (GANs).
* Deployment of a web-based dashboard for real-time factory monitoring.
