# Phase 3: Video Presentation Script
**Project:** Metal Surface Defect Classification Using Deep Learning

*Use this script as a guide when recording your screen. We recommend opening your Kaggle notebook and scrolling through it as you speak!*

---

## Slide 1: Introduction (0:00 - 0:30)
**Visual:** Show the Kaggle Notebook Title or a Title Slide.
**Script:** 
"Hello everyone, my name is Dimple Parmar, and my Machine Learning project focuses on 'Industrial Defect Detection: Metal Surface Defect Classification'. 

The problem I am solving is the automation of quality control in steel manufacturing. Currently, human inspectors manually check hot-rolled steel for defects, which is slow and prone to error. My research question was: Can we use deep learning to automatically detect and classify these defects accurately, and how does a custom CNN compare to a pre-trained Transfer Learning model like MobileNetV2?"

---

## Slide 2: Dataset & Setup (0:30 - 1:00)
**Visual:** Scroll down to Section 2 in the notebook to show the EDA (Exploratory Data Analysis) images and the bar chart.
**Script:**
"I used the NEU Metal Surface Defect Database, a public dataset containing 1,800 grayscale images. As you can see here in the data distribution, the dataset is perfectly balanced across 6 classes of defects: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches. 

To ensure rigorous testing, I programmatically split the dataset using a stratified 70% Training, 15% Validation, and 15% Test split."

---

## Slide 3: Methodology & CNN Design (1:00 - 1:45)
**Visual:** Scroll to Section 3 (Augmentation) and Section 4 (Custom CNN Model).
**Script:**
"My implementation workflow started with real-time Data Augmentation—including rotations, flips, and brightness adjustments—to prevent the model from overfitting on the small dataset.

I then designed two distinct models. The first was a Custom Convolutional Neural Network built from scratch, consisting of four Convolutional blocks with Batch Normalization and MaxPooling, followed by a dense classification head.

The second model used Transfer Learning. I loaded a pre-trained MobileNetV2 base, locked the weights for initial feature extraction, and then fine-tuned the top layers to adapt specifically to the metal textures."

---

## Slide 4: Main Results & Tables (1:45 - 2:30)
**Visual:** Scroll down to Section 9 (Model Comparison) to show the final Performance Table and ROC curves.
**Script:**
"Let's look at the results. Both models performed exceptionally well on the holdout test set, but the MobileNetV2 transfer learning model emerged as the clear winner.

As shown in this comparison table, MobileNetV2 achieved a Test Accuracy of 98.15% with a 98% F1 Score, compared to the Custom CNN which reached 92.59%. The ROC curves also confirm MobileNetV2's superiority, achieving an Area Under the Curve (AUC) of 0.99 for all six defect classes."

---

## Slide 5: Explainability & Limitations (2:30 - 3:00)
**Visual:** Scroll to Section 7 (Explainable AI: Grad-CAM Visualizations).
**Script:**
"A very important part of my project was proving *why* the model makes its decisions. I implemented Grad-CAM, which produces these heatmaps. You can see the red 'hot spots' precisely map onto the actual physical scratches and patches on the steel. This proves the neural network is looking at the correct defects and not just background noise.

As for limitations, this model is an image classifier, which means it tells us *what* defect is in the image, but it cannot draw exact bounding boxes around multiple defects in a single large steel sheet."

---

## Slide 6: Conclusion (3:00 - 3:15)
**Visual:** Scroll to the Conclusion section of the notebook.
**Script:**
"In conclusion, this project demonstrates that a fine-tuned MobileNetV2 architecture is highly robust for industrial quality control, achieving over 98% accuracy. Future work would involve transitioning this to an Object Detection model like YOLOv8 for real-time factory floor deployment. 

Thank you for watching!"
