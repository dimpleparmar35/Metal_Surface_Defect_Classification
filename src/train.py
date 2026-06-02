import os
import argparse
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from preprocessing import get_data_generators
from model import build_custom_cnn, build_mobilenetv2_transfer
from utils import ensure_dir, plot_training_curves

def train_model(model_type, data_dir, epochs=30, batch_size=32):
    # Setup directories
    models_dir = os.path.join("outputs", "models")
    figures_dir = os.path.join("outputs", "figures")
    ensure_dir(models_dir)
    ensure_dir(figures_dir)
    
    # Get Data
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)
    num_classes = train_gen.num_classes
    
    # Build Model
    if model_type == 'custom':
        model = build_custom_cnn(num_classes=num_classes)
        model_name = "custom_cnn.keras"
    elif model_type == 'mobilenetv2':
        model, base_model = build_mobilenetv2_transfer(num_classes=num_classes)
        model_name = "mobilenetv2.keras"
    else:
        raise ValueError("Invalid model type. Choose 'custom' or 'mobilenetv2'")
        
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(models_dir, model_name)
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print(f"Starting training for {model_type}...")
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes.")
    
    # Fine-tuning for MobileNetV2 if requested
    if model_type == 'mobilenetv2':
        print("Unfreezing some layers of MobileNetV2 for fine-tuning...")
        base_model.trainable = True
        # Fine-tune from layer 100 onwards
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        model.compile(
            optimizer=Adam(learning_rate=1e-5), # Lower learning rate for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10, # Additional epochs for fine tuning
            callbacks=callbacks
        )
        
        # Append history
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
            
    # Plot and save curves
    curve_path = os.path.join(figures_dir, f"{model_type}_learning_curves.png")
    plot_training_curves(history, save_path=curve_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Metal Surface Defect Classification Model")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "mobilenetv2"], help="Model type to train")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "raw"), help="Path to raw dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.model, args.data_dir, args.epochs, args.batch_size)
