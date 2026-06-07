"""
train.py

This script trains the selected defect classification model (Custom CNN or MobileNetV2)
on the NEU Metal Surface Defect dataset. It handles:
1. Loading the train, val, and test generators from src/preprocessing.py.
2. Initializing the model with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
3. Compiling and running the training loop (including fine-tuning for MobileNetV2).
4. Tracking and logging the exact training execution time.
5. Saving the trained model weights and training time metrics to disk.
"""

import os
import argparse
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from preprocessing import get_data_generators
from model import build_custom_cnn, build_mobilenetv2_transfer
from utils import ensure_dir, plot_training_curves

def train_model(model_type: str, data_dir: str, epochs: int = 30, batch_size: int = 32):
    """
    Sets up the pipeline, instantiates the chosen model, runs the training loop,
    and writes performance history and training duration files.

    Parameters:
    -----------
    model_type : str
        Type of model to train. Choose 'custom' or 'mobilenetv2'.
    data_dir : str
        Path to the dataset directory containing class subfolders.
    epochs : int, default 30
        Maximum number of training epochs to run.
    batch_size : int, default 32
        Batch size used by data generators.
    """
    # Create required directories for outputs
    models_dir = os.path.join("outputs", "models")
    figures_dir = os.path.join("outputs", "figures")
    ensure_dir(models_dir)
    ensure_dir(figures_dir)
    
    # Load the 3 generators (Train, Val, Test) from preprocessing module
    # Notice that preprocessing now yields 3 generators. We will hold on to test_gen
    # just to verify its class definitions, but won't use it during training.
    train_gen, val_gen, test_gen = get_data_generators(
        data_dir, 
        batch_size=batch_size, 
        img_size=(224, 224)
    )
    num_classes = train_gen.num_classes
    
    # Instantiate the selected model architecture
    if model_type == 'custom':
        model = build_custom_cnn(num_classes=num_classes)
        model_name = "custom_cnn.keras"
    elif model_type == 'mobilenetv2':
        model, base_model = build_mobilenetv2_transfer(num_classes=num_classes)
        model_name = "mobilenetv2.keras"
    else:
        raise ValueError("Invalid model type. Choose 'custom' or 'mobilenetv2'")
        
    model.summary()
    
    # Define optimization criteria and compile the model
    # Adam optimizer is used with an initial learning rate of 0.001
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks configuration:
    # 1. ModelCheckpoint: Saves the best model weights based on validation accuracy.
    # 2. EarlyStopping: Terminate training if validation loss does not improve for 10 epochs.
    # 3. ReduceLROnPlateau: Reduce LR by a factor of 0.2 if validation loss plateaus for 5 epochs.
    checkpoint_path = os.path.join(models_dir, model_name)
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=1e-6, 
            verbose=1
        )
    ]
    
    print(f"\n[Trainer] Starting training for {model_type}...")
    start_time = time.time()
    
    # Run the main training loop
    history_callback = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    history_data = history_callback.history
    
    # Additional Fine-Tuning Phase for MobileNetV2 transfer learning
    if model_type == 'mobilenetv2':
        print("\n[Trainer] Starting fine-tuning phase for MobileNetV2...")
        # Unfreeze base model layers
        base_model.trainable = True
        
        # Keep early layers frozen (frozen up to layer 100) to preserve low-level features
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Compile model with a very low learning rate (1e-5) to prevent destroying pre-trained weights
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        fine_tune_callbacks = [
            ModelCheckpoint(
                checkpoint_path, 
                monitor='val_accuracy', 
                save_best_only=True, 
                verbose=1,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True, 
                verbose=1
            )
        ]
        
        ft_start_time = time.time()
        # Train for an additional 10 epochs specifically for fine-tuning
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=fine_tune_callbacks
        )
        ft_end_time = time.time()
        
        # Add fine-tuning time to the total training time
        elapsed_seconds += (ft_end_time - ft_start_time)
        
        # Merge training histories
        for key in history_data.keys():
            if key in history_fine.history:
                history_data[key].extend(history_fine.history[key])
                
    print(f"[Trainer] Training completed in {elapsed_seconds/60:.2f} minutes.")
    
    # Save the training time to a text file for evaluation and model comparison
    time_file_path = os.path.join(models_dir, f"{model_type}_training_time.txt")
    with open(time_file_path, 'w') as f:
        f.write(f"{elapsed_seconds:.2f}")
    print(f"[Trainer] Training time saved to {time_file_path}")
    
    # Generate and save the learning curves (Accuracy & Loss) to the figures directory
    curve_path = os.path.join(figures_dir, f"{model_type}_learning_curves.png")
    plot_training_curves(history_data, save_path=curve_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Metal Surface Defect Classification Model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="custom", 
        choices=["custom", "mobilenetv2"], 
        help="Type of model to train ('custom' or 'mobilenetv2')"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=os.path.join("data", "raw"), 
        help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    train_model(args.model, args.data_dir, args.epochs, args.batch_size)
