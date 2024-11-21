import os
import oci
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "./model/model.keras"
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
NAMESPACE = "ocisateam"
BUCKET_NAME = "medical-image-bucket"
PREFIX = "model/"

# Functions

def prepare_data(train_dir, validation_dir):
    """Prepares data generators for training and validation."""
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",  # Change to "categorical" for multi-class
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    return train_generator, validation_generator


def build_model(input_shape):
    """Builds a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use `Dense(len(classes), activation='softmax')` for multi-class
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_generator, validation_generator, save_path, epochs):
    """Trains the model and saves the best version."""
    checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    # Save the final model
    model.save(save_path)
    print(f"Model saved at {save_path}")


def main():
    """Main function to orchestrate training."""
    # Verify directory structure
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALIDATION_DIR):
        print("Error: Training or validation directory does not exist.")
        return

    # Prepare data
    train_generator, validation_generator = prepare_data(TRAIN_DIR, VALIDATION_DIR)

    # Build model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = build_model(input_shape)

    # Train model
    train_model(model, train_generator, validation_generator, MODEL_SAVE_PATH, EPOCHS)

    # Upload to OCI Object Storage
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config = {}, signer=signer )
    object_name = os.path.join(PREFIX, "model.keras").replace("\\", "/")
    with open(MODEL_SAVE_PATH, "rb") as file_data:
        object_client.put_object(
                        NAMESPACE,
                        BUCKET_NAME,
                        object_name,
                        file_data
                    )


if __name__ == "__main__":
    main()
