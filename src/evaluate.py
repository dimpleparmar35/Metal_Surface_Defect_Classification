import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.utils import to_categorical

def evaluate_model(model_path: str, data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224), results_dir: str = "outputs/figures"):
    """Evaluate a trained model and generate metrics + visualizations.

    Parameters
    ----------
    model_path: str
        Path to the saved Keras model (e.g., ``outputs/models/custom_cnn.keras``).
    data_dir: str
        Directory containing the validation data organized by class.
    batch_size: int, default 32
        Batch size for the evaluation generator.
    img_size: tuple, default (224, 224)
        Image resize dimensions.
    results_dir: str, default "outputs/figures"
        Directory where figures (confusion matrix, ROC curves) will be saved.
    """
    # Load model
    model = load_model(model_path)
    # Build validation generator (no augmentation, only rescaling)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    # Predict
    preds = model.predict(val_gen, verbose=1)
    y_true = val_gen.classes
    y_pred = np.argmax(preds, axis=1)
    class_names = list(val_gen.class_indices.keys())

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC curves (one‑vs‑rest, multiclass)
    y_true_bin = to_categorical(y_true, num_classes=len(class_names))
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc='lower right')
    roc_path = os.path.join(results_dir, 'roc_curves.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

    # Save textual metrics
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Metrics summary saved to {metrics_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a Keras model on validation data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (.keras/.h5)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to validation data folder")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.batch_size)
