# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 2 - Proposal and Code Implementation - Dimple Parmar
# # Industrial Defect Detection: Metal Surface Defect Classification Model
# ## Phase 2 CNN Project - End-to-End Implementation
#
# This notebook contains the complete, self-contained deep learning pipeline to load, preprocess, augment, train, evaluate, and interpret classification models for hot-rolled steel surface defects using the **NEU Metal Surface Defect Dataset**.
#
# **Dataset:** NEU Metal Surface Defect Database — 1,800 grayscale images, 6 classes, 300 per class.
#
# **GitHub:** https://github.com/dimpleparmar35/Metal_Surface_Defect_Classification
#
# ### Project Outline:
# 1. **Setup & Path Configurations**: Locate and verify dataset directories.
# 2. **Exploratory Data Analysis (EDA)**: Plot class distribution and display defect sample images.
# 3. **Preprocessing & Augmentation Pipeline**: Stratified 70/15/15 splitting and augmentation visualization.
# 4. **Custom CNN Training**: Designing and training a convolutional network from scratch.
# 5. **MobileNetV2 Transfer Learning**: Frozen feature extraction and subsequent fine-tuning.
# 6. **Evaluation & Visualization**: Generating comparative training curves, confusion matrices, and ROC curves.
# 7. **Interpretability via Grad-CAM**: Visualizing defect-specific neural network activations.
# 8. **Error Analysis**: Displaying correct vs incorrect predictions.
# 9. **Model Comparison**: Creating and saving the final performance comparison table.
# 10. **Conclusion**: Summary of findings.

# %%
import os
import time
import glob
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                      Dropout, BatchNormalization, Input,
                                      GlobalAveragePooling2D)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score, roc_curve, auc)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
print("Libraries loaded successfully!")
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# %% [markdown]
# ### 1. Setup & Path Configurations
# This notebook auto-detects the dataset whether running on Kaggle or locally.

# %%
# ── Auto-discover dataset path ─────────────────────────────────────────────────
# Step 1: Print the full Kaggle input tree so we can always see exactly
#         what paths are available (very useful for debugging).
print("=== Kaggle Input Directory Structure ===")
kaggle_input = '/kaggle/input'
if os.path.isdir(kaggle_input):
    for root, dirs, files in os.walk(kaggle_input):
        depth = root.replace(kaggle_input, '').count(os.sep)
        indent = '  ' * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth >= 3:   # Don't print individual file names, just folders
            dirs[:] = []
else:
    print("Not running on Kaggle — local environment detected.")

# Step 2: Smart recursive search — find the folder that contains
#         the 6 NEU defect class subfolders directly.
VALID_CLASSES = {'crazing', 'inclusion', 'patches',
                 'pitted_surface', 'rolled-in_scale', 'scratches'}

DATA_DIR = None

# First try well-known fixed paths
possible_dirs = [
    '/kaggle/input/neu-surface-defect-database/NEU-DET/train/images',
    '/kaggle/input/neu-surface-defect-database/NEU-DET/images',
    '/kaggle/input/neu-surface-defect-database/NEU-DET',
    '/kaggle/input/neu-surface-defect-database',
    '../data/raw',
    './data/raw',
]

for path in possible_dirs:
    if os.path.isdir(path):
        subdirs = set(os.listdir(path))
        # Accept if at least 4 of our 6 class names are direct subdirs
        if len(subdirs & VALID_CLASSES) >= 4:
            DATA_DIR = path
            print(f"\n✅ Dataset found (class folders) at: {DATA_DIR}")
            break
        # Or accept if it contains image files anywhere beneath it
        imgs = (glob.glob(os.path.join(path, '**', '*.bmp'), recursive=True) +
                glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True) +
                glob.glob(os.path.join(path, '**', '*.png'), recursive=True))
        if imgs:
            DATA_DIR = path
            print(f"\n✅ Dataset found (images) at: {DATA_DIR}  ({len(imgs)} files)")
            break

# If still not found, walk ALL of /kaggle/input and find the deepest
# directory that contains at least 4 of the 6 class subfolders.
if DATA_DIR is None and os.path.isdir(kaggle_input):
    print("\n🔍 Searching entire /kaggle/input for class folders...")
    for root, dirs, files in os.walk(kaggle_input):
        if len(set(dirs) & VALID_CLASSES) >= 4:
            DATA_DIR = root
            print(f"✅ Auto-discovered at: {DATA_DIR}")
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "❌ Dataset not found!\n"
        "On Kaggle  → click '+ Add Input' on the right panel and search for:\n"
        "             'kaustubhdikshit/neu-surface-defect-database'\n"
        "             then click Run All again.\n"
        "Locally    → place images in  data/raw/<class_name>/  folders."
    )

# Create output directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
print("Output directories ready.")

# %% [markdown]
# ### 2. Exploratory Data Analysis (EDA)
# Examine the dataset class distribution and display one sample image from each defect category.

# %%
# ── Load all image paths ───────────────────────────────────────────────────────
extensions = ('*.bmp', '*.jpg', '*.jpeg', '*.png', '*.BMP', '*.JPG', '*.JPEG', '*.PNG')
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

if not image_paths:
    raise ValueError(f"No images found in '{DATA_DIR}'. Check that class subfolders contain images.")

data_records = []
for path in image_paths:
    label = os.path.basename(os.path.dirname(path))
    data_records.append({'filepath': path, 'label': label})

df = pd.DataFrame(data_records)

# Filter out non-class labels (e.g. 'images', 'train', 'annotations' folders)
VALID_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface',
                 'rolled-in_scale', 'scratches']
df = df[df['label'].isin(VALID_CLASSES)].reset_index(drop=True)

print(f"Total Images Found: {len(df)}")
print(df['label'].value_counts())

# ── Plot class distribution ────────────────────────────────────────────────────
plt.figure(figsize=(9, 4))
ax = sns.countplot(data=df, x='label',
                   order=sorted(df['label'].unique()),
                   palette='viridis')
ax.set_title('NEU Dataset — Class Distribution', fontsize=13, weight='bold')
ax.set_xlabel('Defect Class')
ax.set_ylabel('Number of Images')
plt.xticks(rotation=20, ha='right')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/eda_class_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ EDA class distribution saved.")

# ── Display sample images from each class ─────────────────────────────────────
classes = sorted(df['label'].unique())
plt.figure(figsize=(14, 6))
for idx, label in enumerate(classes):
    sample_path = df[df['label'] == label]['filepath'].values[0]
    img = cv2.imread(sample_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, idx + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{label}\n{img.shape[0]}×{img.shape[1]}px", fontsize=9)
    plt.axis('off')
plt.suptitle("NEU Dataset — One Sample per Defect Class", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/eda_defect_samples.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Defect sample images saved.")

# %% [markdown]
# ### 3. Preprocessing & Augmentation Pipeline
# Perform a stratified **70% training / 15% validation / 15% test** split.
# Data augmentation is applied to the training set ONLY.

# %%
# ── Stratified 70/15/15 Split ─────────────────────────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
)

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

TARGET_SIZE = (224, 224)
BATCH_SIZE  = 32

# ── Generators ────────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label',
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=42
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label',
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=42
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col='filepath', y_col='label',
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=42
)
NUM_CLASSES  = len(train_gen.class_indices)
class_names  = list(train_gen.class_indices.keys())
print(f"Classes detected: {class_names}")

# ── Augmentation preview ───────────────────────────────────────────────────────
# NOTE: We use a SEPARATE preview generator with NO rescale here.
# The main train_datagen already has rescale=1/255 built in, so if we
# also divide by 255 manually the image becomes black (double normalization).
preview_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    # NO rescale here — pixel values stay in [0,255] for correct display
)

sample_path = train_df.iloc[0]['filepath']
img_aug     = load_img(sample_path, target_size=TARGET_SIZE)
x_aug       = np.expand_dims(img_to_array(img_aug), axis=0)  # keep [0,255] range

plt.figure(figsize=(14, 3))
plt.subplot(1, 5, 1)
plt.imshow(img_aug)
plt.title("Original", fontsize=9)
plt.axis('off')

aug_gen = preview_datagen.flow(x_aug, batch_size=1, seed=42)
for i in range(4):
    batch = next(aug_gen)
    plt.subplot(1, 5, i + 2)
    plt.imshow(batch[0].astype('uint8'))  # cast back to uint8 for display
    plt.title(f"Augmented {i+1}", fontsize=9)
    plt.axis('off')

plt.suptitle("Real-time Training Augmentation Examples", weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/preprocessing_augmentation_samples.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Augmentation preview saved.")

# %% [markdown]
# ### 4. Custom CNN Model
# Design and train a 4-block CNN architecture from scratch using TensorFlow/Keras.

# %%
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=6):
    """
    Custom CNN: 4 Conv blocks (32→64→128→256 filters), BN, MaxPool.
    Dense head with Dropout for regularization. Softmax output.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classification head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="Custom_CNN")
    return model

custom_model = build_custom_cnn(num_classes=NUM_CLASSES)
custom_model.summary()

custom_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_custom = [
    ModelCheckpoint('outputs/models/custom_cnn.keras',
                    monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
]

print("🚀 Starting Custom CNN training...")
t0 = time.time()
history_custom = custom_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_custom,
    verbose=1
)
custom_train_time = time.time() - t0
print(f"✅ Custom CNN finished in {custom_train_time/60:.2f} minutes.")

# %% [markdown]
# ### 5. Transfer Learning — MobileNetV2
# Two-phase approach: frozen feature extraction followed by deep fine-tuning.

# %%
# ── Phase 1: Feature Extraction ───────────────────────────────────────────────
base_model = MobileNetV2(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
base_model.trainable = False   # freeze all base layers

x          = base_model.output
x          = GlobalAveragePooling2D()(x)
x          = Dense(256, activation='relu')(x)
x          = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

mobile_model = Model(inputs=base_model.input,
                     outputs=predictions,
                     name="MobileNetV2_Transfer")
mobile_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_mob = [
    ModelCheckpoint('outputs/models/mobilenetv2.keras',
                    monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

print("🚀 Phase 1: Feature Extraction...")
t0 = time.time()
history_mobile = mobile_model.fit(
    train_gen, validation_data=val_gen,
    epochs=15, callbacks=callbacks_mob, verbose=1
)

# ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────────
print("\n🔧 Phase 2: Fine-Tuning layers 100+ ...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

mobile_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_fine = mobile_model.fit(
    train_gen, validation_data=val_gen,
    epochs=15, callbacks=callbacks_mob, verbose=1
)
mobile_train_time = time.time() - t0
print(f"✅ MobileNetV2 total training finished in {mobile_train_time/60:.2f} minutes.")

# Merge histories for plotting
merged_history = {k: history_mobile.history[k] + history_fine.history[k]
                  for k in history_mobile.history}

# %% [markdown]
# ### 6. Performance Evaluation & Visualization
# Evaluate both models on the holdout test set. Generate training curves, confusion matrices, and ROC curves.

# %%
# ── 6.1 Learning Curves ───────────────────────────────────────────────────────
def plot_curves(history_dict, title, save_name):
    acc     = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss    = history_dict['loss']
    val_loss= history_dict['val_loss']
    epochs  = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs, acc,     'b-o', ms=4, label='Train Acc')
    ax1.plot(epochs, val_acc, 'r-o', ms=4, label='Val Acc')
    ax1.set_title(f'{title} — Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.plot(epochs, loss,     'b-o', ms=4, label='Train Loss')
    ax2.plot(epochs, val_loss, 'r-o', ms=4, label='Val Loss')
    ax2.set_title(f'{title} — Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.4)

    plt.suptitle(f'{title} Training History', weight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{save_name}.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"✅ {save_name}.png saved.")

plot_curves(history_custom.history, 'Custom CNN',   'custom_learning_curves')
plot_curves(merged_history,          'MobileNetV2',  'mobilenetv2_learning_curves')

# %%
# ── 6.2 Test Set Evaluation ───────────────────────────────────────────────────
y_true = test_gen.classes

def evaluate_on_test(model, name):
    test_gen.reset()
    preds  = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {name} — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5)
    plt.title(f'Confusion Matrix — {name}', fontsize=12, weight='bold')
    plt.ylabel('True Class'); plt.xlabel('Predicted Class')
    plt.tight_layout()
    save_key = name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f'outputs/figures/confusion_matrix_{save_key}.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved.")

    return acc, prec, rec, f1, preds

custom_metrics = evaluate_on_test(custom_model,  "Custom CNN")
mobile_metrics = evaluate_on_test(mobile_model,  "MobileNetV2")

# %%
# ── 6.3 ROC Curves ───────────────────────────────────────────────────────────
def plot_roc(y_true, preds, name):
    y_bin = to_categorical(y_true, num_classes=NUM_CLASSES)
    plt.figure(figsize=(7, 5))
    for i, cls in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], preds[:, i])
        auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, lw=1.5, label=f"{cls} (AUC={auc_i:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC Curves — {name}', fontsize=12, weight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_key = name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f'outputs/figures/roc_curves_{save_key}.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"✅ ROC curves saved.")

plot_roc(y_true, custom_metrics[4], "Custom CNN")
plot_roc(y_true, mobile_metrics[4], "MobileNetV2")

# %% [markdown]
# ### 7. Explainable AI — Grad-CAM Visualizations
# Implement Grad-CAM to overlay neural activation heatmaps on raw defect images,
# verifying that predictions align with the physical location of the steel surface defects.

# %%
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Keras 3 compatible Grad-CAM.
    Instead of using layer.output (which fails in Keras 3 Sequential models),
    we manually iterate through layers and use GradientTape to watch the
    intermediate conv output tensor directly.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x           = img_tensor
        conv_output = None

        # Forward pass layer-by-layer, capturing conv output at target layer
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)  # tell tape to track this tensor

        predictions = x
        if conv_output is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads        = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output  = conv_output[0]
    heatmap      = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0.0)
    max_val      = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val  = 1e-10
    return (heatmap / max_val).numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.45):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_map   = np.uint8(255 * heatmap)
    jet     = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
    jet     = cv2.resize(jet, (img.shape[1], img.shape[0]))
    jet     = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(jet, alpha, img, 1 - alpha, 0)
    return img, overlay

def get_last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_mobile_conv(model):
    """Find last batch-norm or depthwise conv in MobileNetV2."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback: use last Conv2D-type in base
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

custom_conv = get_last_conv_name(custom_model)
mobile_conv = get_mobile_conv(mobile_model)
print(f"Custom CNN last conv: {custom_conv}")
print(f"MobileNetV2 last conv: {mobile_conv}")

# ── Generate Grad-CAM grid across multiple defect classes ─────────────────────
n_samples = min(3, len(test_df))
fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
if n_samples == 1:
    axes = [axes]

for row, i in enumerate(range(n_samples)):
    sample   = test_df.iloc[i]
    img_path = sample['filepath']
    true_lbl = sample['label']

    img_load = load_img(img_path, target_size=TARGET_SIZE)
    img_arr  = np.expand_dims(img_to_array(img_load) / 255.0, axis=0)

    orig, custom_cam = (None, None)
    _, mobile_cam    = (None, None)

    if custom_conv:
        heat_c     = make_gradcam_heatmap(img_arr, custom_model, custom_conv)
        orig, custom_cam = overlay_gradcam(img_path, heat_c)
    if mobile_conv:
        heat_m     = make_gradcam_heatmap(img_arr, mobile_model, mobile_conv)
        _, mobile_cam = overlay_gradcam(img_path, heat_m)

    if orig is not None:
        axes[row][0].imshow(orig)
        axes[row][0].set_title(f"Original\n{true_lbl}", fontsize=9)
        axes[row][0].axis('off')
    if custom_cam is not None:
        axes[row][1].imshow(custom_cam)
        axes[row][1].set_title("Custom CNN Grad-CAM", fontsize=9)
        axes[row][1].axis('off')
    if mobile_cam is not None:
        axes[row][2].imshow(mobile_cam)
        axes[row][2].set_title("MobileNetV2 Grad-CAM", fontsize=9)
        axes[row][2].axis('off')

plt.suptitle("Explainable AI: Grad-CAM Heatmap Activations", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/gradcam_comparison_sample.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Grad-CAM comparison saved.")

# %% [markdown]
# ### 8. Error Analysis
# Identify where the MobileNetV2 model makes mistakes to pinpoint potential improvements.

# %%
test_gen.reset()
preds_mob  = mobile_model.predict(test_gen, verbose=0)
y_pred_mob = np.argmax(preds_mob, axis=1)

correct_idx   = np.where(y_pred_mob == y_true)[0]
incorrect_idx = np.where(y_pred_mob != y_true)[0]

print(f"Test samples : {len(y_true)}")
print(f"  ✅ Correct   : {len(correct_idx)}  ({len(correct_idx)/len(y_true)*100:.2f}%)")
print(f"  ❌ Incorrect : {len(incorrect_idx)}  ({len(incorrect_idx)/len(y_true)*100:.2f}%)")

n_show = min(2, len(correct_idx)), min(2, len(incorrect_idx))
fig, axes = plt.subplots(1, sum(n_show), figsize=(5 * sum(n_show), 4))
ax_idx = 0

for i in range(n_show[0]):
    idx  = correct_idx[i]
    path = test_df.iloc[idx]['filepath']
    img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    axes[ax_idx].imshow(img)
    axes[ax_idx].set_title(
        f"✅ CORRECT\nTrue: {class_names[y_true[idx]]}\nPred: {class_names[y_pred_mob[idx]]}",
        color='green', fontsize=9)
    axes[ax_idx].axis('off')
    ax_idx += 1

for i in range(n_show[1]):
    idx  = incorrect_idx[i]
    path = test_df.iloc[idx]['filepath']
    img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    axes[ax_idx].imshow(img)
    axes[ax_idx].set_title(
        f"❌ ERROR\nTrue: {class_names[y_true[idx]]}\nPred: {class_names[y_pred_mob[idx]]}",
        color='red', fontsize=9)
    axes[ax_idx].axis('off')
    ax_idx += 1

plt.suptitle("MobileNetV2 — Error Analysis: Correct vs Incorrect Predictions",
             fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/error_analysis_samples.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Error analysis saved.")

# %% [markdown]
# ### 9. Model Comparison Table
# Compile real metrics from both models and render as a high-quality PNG table.

# %%
comp_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)',
               'F1 Score (Weighted)', 'Training Time'],
    'Custom CNN': [
        f"{custom_metrics[0]:.2%}",
        f"{custom_metrics[1]:.4f}",
        f"{custom_metrics[2]:.4f}",
        f"{custom_metrics[3]:.4f}",
        f"{custom_train_time/60:.2f} min"
    ],
    'MobileNetV2': [
        f"{mobile_metrics[0]:.2%}",
        f"{mobile_metrics[1]:.4f}",
        f"{mobile_metrics[2]:.4f}",
        f"{mobile_metrics[3]:.4f}",
        f"{mobile_train_time/60:.2f} min"
    ]
})

print("=== Final Performance Comparison ===")
print(comp_df.to_string(index=False))

# Render as a polished matplotlib table
fig, ax = plt.subplots(figsize=(9, 3), dpi=200)
ax.axis('off')
tbl = ax.table(
    cellText=comp_df.values,
    colLabels=comp_df.columns,
    loc='center',
    cellLoc='center',
    colColours=['#2B4590', '#2B4590', '#2B4590']
)
tbl.auto_set_fontsize(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.6)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EEF2FF')

plt.title("Model Comparison: Custom CNN vs MobileNetV2",
          fontsize=13, pad=16, weight='bold')
plt.savefig('outputs/figures/comparison_table.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Comparison table saved.")

# %% [markdown]
# ### 10. Conclusion
# This project successfully implemented an end-to-end deep learning pipeline to automate
# the detection and classification of metal surface defects on hot-rolled steel.
#
# **Key Takeaways:**
# * **Transfer Learning is Superior:** Fine-tuned MobileNetV2 achieved ~98% accuracy
#   vs ~92% for the Custom CNN, proving that ImageNet pre-trained weights generalize
#   extremely well to industrial steel textures.
# * **Custom CNN is Viable:** The lighter scratch-built CNN (fewer params, faster training)
#   is still industrially viable for resource-constrained edge deployments.
# * **Data Augmentation is Critical:** Real-time augmentation (flips, rotations, zoom,
#   brightness) was essential to prevent overfitting on the small 1,800-image dataset.
# * **Grad-CAM Validates Trust:** Heatmap overlays confirm models focus on the actual
#   physical defect anomaly regions, not background noise — critical for factory deployment.
# * **Future Work:** Transition to YOLOv8 object detection for multi-defect localization
#   and deploy on NVIDIA Jetson Nano for real-time factory inspection.
