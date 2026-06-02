import os
import matplotlib.pyplot as plt

def ensure_dir(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_curves(history, save_path=None):
    """Plot training and validation accuracy and loss curves."""
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve
    ax1.plot(epochs, acc, 'b-', label='Training Accuracy')
    if val_acc:
        ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss curve
    ax2.plot(epochs, loss, 'b-', label='Training Loss')
    if val_loss:
        ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    plt.close()
