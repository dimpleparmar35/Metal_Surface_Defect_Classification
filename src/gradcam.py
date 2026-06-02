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
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize colormap to match original image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(jet, alpha, img, 1 - alpha, 0)

    # Save the superimposed image
    cv2.imwrite(cam_path, superimposed_img)

def run_gradcam(model_path, image_path, target_size=(224, 224)):
    figures_dir = os.path.join("outputs", "figures")
    ensure_dir(figures_dir)
    
    model_name = os.path.basename(model_path).split('.')[0]
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Find last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break
            
    # For MobileNetV2 wrapped in our Custom model, the conv layer is inside the base model
    if model_name == "mobilenetv2":
        # Extract the base MobileNetV2 model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        if base_model:
            for layer in reversed(base_model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    model = base_model # Use base model for heatmap extraction
                    break
    
    if not last_conv_layer_name:
        print("Could not find a convolutional layer. Grad-CAM requires a CNN.")
        return

    print(f"Using layer {last_conv_layer_name} for Grad-CAM.")
    
    # Prepare image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Rescale as done in training
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save output
    img_basename = os.path.basename(image_path)
    cam_path = os.path.join(figures_dir, f"gradcam_{model_name}_{img_basename}")
    
    save_and_display_gradcam(image_path, heatmap, cam_path)
    print(f"Grad-CAM visualization saved to {cam_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    
    args = parser.parse_args()
    
    if os.path.exists(args.model_path) and os.path.exists(args.image_path):
        run_gradcam(args.model_path, args.image_path)
    else:
        print("Model or Image path not found.")
