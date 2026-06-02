import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224), validation_split: float = 0.2, seed: int = 42):
    """Create training and validation data generators.

    Parameters
    ----------
    data_dir: str
        Path to the dataset folder organized as ``class_name/*``.
    batch_size: int, default 32
        Number of samples per batch.
    img_size: tuple, default (224, 224)
        Target size for resizing images.
    validation_split: float, default 0.2
        Fraction of data to reserve for validation.
    seed: int, default 42
        Random seed for reproducibility.

    Returns
    -------
    train_gen, val_gen: ImageDataGenerator.flow_from_directory objects
        Ready‑to‑use generators for model.fit().
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please place the NEU dataset there.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=validation_split,
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        shuffle=False,
    )

    return train_gen, val_gen
