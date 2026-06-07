"""
model.py

This module defines the Deep Learning architectures used for classifying metal 
surface defects on the NEU Metal Surface Defect dataset. Two distinct architectures 
are implemented:

1. Custom CNN: A built-from-scratch convolutional neural network with alternating
   Conv2D, BatchNormalization, MaxPooling2D, and Dropout layers.
2. MobileNetV2 Transfer Learning: A lightweight transfer learning model utilizing 
   pre-trained ImageNet weights, coupled with a custom dense classification head.
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=6):
    """
    Builds and returns a custom Convolutional Neural Network (CNN).

    This model is built sequentially and uses multiple convolutional layers
    of increasing filters to capture hierarchical image features (from low-level 
    edges to high-level textures). Batch Normalization is placed after each 
    convolutional layer to accelerate convergence, and Dropout is applied 
    before dense layers to mitigate overfitting.

    Parameters:
    -----------
    input_shape : tuple of int, default (224, 224, 3)
        Dimensions of the input images (height, width, channels).
    num_classes : int, default 6
        Number of output defect categories (classes).

    Returns:
    --------
    model : tensorflow.keras.models.Sequential
        The compiled but untrained custom CNN model.
    """
    model = Sequential([
        # Explicit input layer for cleaner summary and visualization
        Input(shape=input_shape, name="Input_Layer"),
        
        # Block 1: Low-level feature extraction (edges, borders)
        Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv1"),
        BatchNormalization(name="BatchNorm1"), # Normalizes activations to stabilize training
        MaxPooling2D((2, 2), name="MaxPool1"), # Reduces spatial resolution by half (112x112)
        
        # Block 2: Mid-level feature extraction (textures, simple shapes)
        Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv2"),
        BatchNormalization(name="BatchNorm2"),
        MaxPooling2D((2, 2), name="MaxPool2"), # Reduces spatial resolution to 56x56
        
        # Block 3: Higher-level feature extraction (complex textures, patterns)
        Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv3"),
        BatchNormalization(name="BatchNorm3"),
        MaxPooling2D((2, 2), name="MaxPool3"), # Reduces spatial resolution to 28x28
        
        # Block 4: Deep feature extraction (defect-specific topological patterns)
        Conv2D(256, (3, 3), activation='relu', padding='same', name="Conv4"),
        BatchNormalization(name="BatchNorm4"),
        MaxPooling2D((2, 2), name="MaxPool4"), # Reduces spatial resolution to 14x14
        
        # Fully Connected (Dense) Head
        Flatten(name="Flatten"), # Convert 3D feature maps to 1D vector (14 * 14 * 256 = 50,176 values)
        
        # First Dense Layer
        Dense(512, activation='relu', name="Dense512"),
        Dropout(0.5, name="Dropout512"), # Randomly drops 50% of activations to prevent overfitting
        
        # Second Dense Layer
        Dense(256, activation='relu', name="Dense256"),
        Dropout(0.5, name="Dropout256"), # Additional dropout for generalization
        
        # Output classification layer
        Dense(num_classes, activation='softmax', name="Output_Layer") # Output class probabilities
    ], name="Custom_CNN_Defect_Classifier")
    
    return model

def build_mobilenetv2_transfer(input_shape=(224, 224, 3), num_classes=6):
    """
    Builds and returns a Transfer Learning model using pre-trained MobileNetV2.

    MobileNetV2 is a lightweight CNN designed for mobile and resource-constrained 
    devices, utilizing depthwise separable convolutions. We import it pre-trained 
    on the ImageNet dataset, freeze its feature-extraction base, and append a custom 
    Fully Connected classifier head.

    Parameters:
    -----------
    input_shape : tuple of int, default (224, 224, 3)
        Dimensions of the input images.
    num_classes : int, default 6
        Number of output classes.

    Returns:
    --------
    model : tensorflow.keras.models.Model
        The hybrid Keras Model combining the frozen base and trainable head.
    base_model : tensorflow.keras.applications.MobileNetV2
        The base MobileNetV2 model reference (needed later for unfreezing layers during fine-tuning).
    """
    # Load MobileNetV2 base pre-trained on ImageNet
    # We set include_top=False to discard the default 1000-class classifier head
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the base layers to retain pre-learned features (e.g. general textures/edges)
    # This speeds up training and prevents gradient destruction of pre-trained weights
    base_model.trainable = False
    
    # Custom classification head
    x = base_model.output
    
    # Global Average Pooling reduces spatial dimensions to 1D (7x7x1280 -> 1280)
    # This is less prone to overfitting than a standard Flatten layer
    x = GlobalAveragePooling2D(name="GlobalAvgPool")(x)
    
    # Custom intermediate Dense layer
    x = Dense(256, activation='relu', name="FC_Dense")(x)
    x = Dropout(0.5, name="FC_Dropout")(x) # Heavy dropout to reduce overfitting on small datasets
    
    # Final softmax output layer
    predictions = Dense(num_classes, activation='softmax', name="FC_Output")(x)
    
    # Define the final Keras functional Model
    model = Model(
        inputs=base_model.input, 
        outputs=predictions, 
        name="MobileNetV2_Transfer_Learning"
    )
    
    return model, base_model
