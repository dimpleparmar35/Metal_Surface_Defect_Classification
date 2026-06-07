"""
preprocessing.py

This module handles the data preprocessing pipeline for the Metal Surface Defect
Classification project. It walks the dataset directory, performs a stratified
split of the images into training (70%), validation (15%), and testing (15%) sets,
and returns three separate Keras ImageDataGenerator flow objects.

Data Augmentation is applied only to the training generator to prevent overfitting
and improve model generalization.
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224), seed: int = 42):
    """
    Scans the dataset directory, performs a stratified split, and returns 
    train, validation, and test generators.

    Parameters:
    -----------
    data_dir : str
        Path to the root folder of the dataset, which contains subdirectories 
        for each class (e.g., crazing, inclusion, patches, pitted_surface, etc.).
    batch_size : int, default 32
        Size of the batches of data.
    img_size : tuple of int, default (224, 224)
        The dimensions to which all images found will be resized.
    seed : int, default 42
        Random seed for reproducibility during splitting and shuffling.

    Returns:
    --------
    train_gen : DataFrameIterator
        Generator for training data with augmentation applied.
    val_gen : DataFrameIterator
        Generator for validation data (rescaled only).
    test_gen : DataFrameIterator
        Generator for test data (rescaled only).
    """
    # Verify directory existence
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Please ensure it exists."
        )

    # Walk directory to find all images
    # Supported formats: .bmp, .jpg, .jpeg, .png (NEU dataset uses grayscale .bmp)
    extensions = ('*.bmp', '*.jpg', '*.jpeg', '*.png', '*.BMP', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    
    # Traverse through class subdirectories
    for ext in extensions:
        pattern = os.path.join(data_dir, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))

    if not image_paths:
        raise ValueError(
            f"No image files found in '{data_dir}'. Make sure images are in class subfolders."
        )

    # Build file-to-label DataFrame
    data_records = []
    for path in image_paths:
        # Resolve class label as the name of the immediate parent folder
        label = os.path.basename(os.path.dirname(path))
        data_records.append({'filepath': os.path.abspath(path), 'label': label})

    df = pd.DataFrame(data_records)
    print(f"[Data Loader] Found {len(df)} images across {df['label'].nunique()} classes:")
    print(df['label'].value_counts())

    # Perform Stratified Split: Train (70%), Validation (15%), Test (15%)
    # Step 1: Split train and temporary test/val (70/30 split)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df['label']
    )

    # Step 2: Split temp into validation and test (50/50 of temp = 15/15 of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df['label']
    )

    print(f"[Split Summary] Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    # Set up augmentation parameters for training
    # Augmentation is critical to reduce overfitting on the small NEU dataset (1800 images)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,          # Rotate images up to 20 degrees
        width_shift_range=0.1,      # Shift images horizontally by 10%
        height_shift_range=0.1,     # Shift images vertically by 10%
        zoom_range=0.2,             # Zoom in/out by 20%
        brightness_range=[0.8, 1.2],# Adjust brightness between 80% and 120%
        horizontal_flip=True,       # Flip horizontally
        vertical_flip=True,         # Flip vertically (useful since steel strip defects are orientation agnostic)
        fill_mode='nearest'         # Strategy to fill newly created pixels
    )

    # Validation and testing generators should only scale pixels (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create flow iterators using flow_from_dataframe
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen, test_gen
