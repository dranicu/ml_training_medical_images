import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
IMG_HEIGHT = 150  # Same as used during training
IMG_WIDTH = 150   # Same as used during training
BATCH_SIZE = 1
MODEL_PATH = "./model/model.keras"  # Path to the saved model
DATA_DIR = "./data/validation"  # Validation directory

def validate_model(data_dir, model_path):
    """Validates a trained model on the test/validation dataset."""
    # Load the model
    model = load_model(model_path)

    # Prepare data generator for validation
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),  # Ensure resizing matches training
        batch_size=BATCH_SIZE,
        class_mode="binary",  # Use "categorical" if multi-class
        shuffle=False  # Ensure consistent ordering for evaluation
    )

    # Evaluate the model
    results = model.evaluate(test_generator)
    print(f"Validation Results - Loss: {results[0]}, Accuracy: {results[1]}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
    elif not os.path.exists(DATA_DIR):
        print(f"Error: Validation directory not found at {DATA_DIR}")
    else:
        validate_model(DATA_DIR, MODEL_PATH)
