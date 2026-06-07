"""
utils.py

This module contains various helper functions and utilities for the Metal Surface 
Defect Classification pipeline. These utilities facilitate directories setup, 
reproducible seeding, and graphing training history curves (Accuracy & Loss).
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def ensure_dir(directory: str):
    """
    Checks if a directory path exists, and creates it recursively if not.

    Parameters:
    -----------
    directory : str
        The folder path to verify and create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[Utils] Created directory: {directory}")

def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and TensorFlow to guarantee 
    reproducible experimental results across separate training runs.

    Parameters:
    -----------
    seed : int, default 42
        The constant seed value to initialize random state generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configure TensorFlow to run deterministically (if possible on the hardware)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Utils] Seeding completed with seed = {seed}")

def plot_training_curves(history: dict, save_path: str = None):
    """
    Plots the training and validation Accuracy and Loss curves side-by-side 
    using Matplotlib and saves the figure to disk.

    Parameters:
    -----------
    history : dict
        A dictionary containing the training logs. Must have 'accuracy' and 'loss' keys,
        and optionally 'val_accuracy' and 'val_loss'.
    save_path : str, optional
        File path where the resulting plot will be saved. If None, plots will be 
        displayed on screen.
    """
    acc = history.get('accuracy')
    val_acc = history.get('val_accuracy')
    loss = history.get('loss')
    val_loss = history.get('val_loss')
    
    if acc is None or loss is None:
        raise KeyError("[Utils] History must contain 'accuracy' and 'loss' lists.")
        
    epochs = range(1, len(acc) + 1)

    # Set up subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left Plot: Accuracy Curves
    ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    if val_acc:
        ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Training and Validation Accuracy', fontsize=12, weight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right Plot: Loss Curves
    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Training and Validation Loss', fontsize=12, weight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure to disk if a destination path is provided
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Utils] Training curves saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()
