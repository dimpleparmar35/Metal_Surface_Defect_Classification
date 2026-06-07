"""
gradcam.py

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM),
an explainable AI (XAI) technique. Grad-CAM uses the gradients of any target 
class flowing into the final convolutional layer of a CNN to produce a coarse 
localization map highlighting the important regions in the image for prediction.

This script allows users to visualize what features of a metal surface defect 
(e.g., scratches, pits) the model is focusing on to make its decision.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import ensure_dir

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes the Grad-CAM heatmap for a given input image and model.

    Parameters:
    -----------
    img_array : numpy.ndarray
        Preprocessed image array of shape (1, height, width, channels).
    model : tensorflow.keras.Model
        The trained CNN model.
    last_conv_layer_name : str
        Name of the final convolutional layer in the network.
    pred_index : int, optional
        Index of the class to compute the heatmap for. If None, uses the 
        model's highest-probability predicted class.

    Returns:
    --------
    heatmap : numpy.ndarray
        A 2D normalized activation heatmap showing class importance.
    """
    # 1. Create a sub-model mapping the inputs to the last conv layer's outputs
    # and the model's final prediction outputs.
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Record operations for automatic differentiation to calculate gradients
    with tf.GradientTape() as tape:
        # Forward pass to retrieve last conv layer activations and predictions
        conv_outputs, predictions = grad_model(img_array)
        
        # If class index is not specified, select the top predicted class
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
            
        # Target class prediction score (y^c)
        class_channel = predictions[:, pred_index]

    # 3. Compute gradients of the score for class c (class_channel)
    # with respect to the feature map activations (A^k) of the last conv layer.
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Compute the neuron importance weights (alpha_k^c) by pooling the gradients
    # over the spatial dimensions (height and width).
    # Shape: (num_channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Compute the weighted sum of feature maps (L_GradCAM = ReLU(sum(alpha_k * A_k)))
    # Shape of conv_outputs[0]: (height, width, num_channels)
    # We matrix-multiply the conv outputs by the pooled gradients vector.
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (we only care about features that positively influence the class of interest)
    # and normalize the heatmap between 0 and 1.
    heatmap = tf.maximum(heatmap, 0.0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-10 # Prevent division by zero
    heatmap = heatmap / max_val
    
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Superimposes the Grad-CAM heatmap onto the original image and saves the result.

    Parameters:
    -----------
    img_path : str
        Path to the original input image.
    heatmap : numpy.ndarray
        2D normalized Grad-CAM activation heatmap (values in [0, 1]).
    cam_path : str, default "cam.jpg"
        Output destination file path for the overlaid image.
    alpha : float, default 0.4
        Opacity of the heatmap overlay (0.0 = only original image, 1.0 = only heatmap).
    """
    # Load original image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Original image not found at {img_path}")
        
    # Scale heatmap from [0, 1] to [0, 255]
    heatmap = np.uint8(255 * heatmap)

    # Apply Jet colormap (turns hot regions red, cold regions blue)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize the colormap to match original image dimensions
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the colored heatmap on the original image
    # Formula: overlaid_img = jet * alpha + original_img * (1 - alpha)
    superimposed_img = cv2.addWeighted(jet, alpha, img, 1.0 - alpha, 0)

    # Save the superimposed image to disk
    cv2.imwrite(cam_path, superimposed_img)

def run_gradcam(model_path: str, image_path: str, target_size=(224, 224)):
    """
    Loads model and image, detects the last conv layer, and generates/saves Grad-CAM.
    """
    figures_dir = os.path.join("outputs", "figures")
    ensure_dir(figures_dir)
    
    model_name = os.path.basename(model_path).split('.')[0].lower()
    print(f"[Grad-CAM] Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Search for the final convolutional layer dynamically
    last_conv_layer_name = None
    
    # First search for a direct Conv2D layer in reverse order
    for layer in reversed(model.layers):
        # If it's a Conv2D layer, grab its name
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    # If not found directly (e.g. if nested), search through model.layers for submodels
    if not last_conv_layer_name:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D) or 'conv' in sub_layer.name.lower():
                        last_conv_layer_name = sub_layer.name
                        break
            if last_conv_layer_name:
                break
                
    if not last_conv_layer_name:
        print("[Grad-CAM] Error: Could not find any convolutional layer in the network.")
        return

    print(f"[Grad-CAM] Identified last convolutional layer: '{last_conv_layer_name}'")
    
    # Load and preprocess the target image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
    img_array /= 255.0 # Scale pixels exactly as done in training/evaluation
    
    # Calculate Grad-CAM Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save the output image
    img_basename = os.path.basename(image_path)
    cam_path = os.path.join(figures_dir, f"gradcam_{model_name}_{img_basename}")
    
    save_and_display_gradcam(image_path, heatmap, cam_path)
    print(f"[Grad-CAM] Heatmap successfully saved to: {cam_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for defect detection")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the saved model file (.keras)"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True, 
        help="Path to the input image (.bmp/.jpg/etc.)"
    )
    
    args = parser.parse_args()
    
    if os.path.exists(args.model_path) and os.path.exists(args.image_path):
        run_gradcam(args.model_path, args.image_path)
    else:
        print("[Grad-CAM] Error: Model or Image path does not exist. Please check your arguments.")
