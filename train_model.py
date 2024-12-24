import os
import oci
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
NAMESPACE = "<NAMESPACE>"  # Set your namespace
BUCKET_RAW = "medical-images-raw"
BUCKET_PROCESSED = "medical-images-processed"
BUCKET_MODEL = "trained-model"
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
MODEL_SAVE_PATH = "./model/model.keras"

# Augmentation setup
seq = iaa.Sequential([
    iaa.Fliplr(1),
    iaa.Rotate((-45, 45)),
    iaa.GaussianBlur(sigma=(0, 2.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.ElasticTransformation(alpha=(0, 10.0), sigma=1.0)
])

# Function to download images from OCI Object Storage
def download_images(namespace, bucket_name, download_dir):
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)

    os.makedirs(download_dir, exist_ok=True)
    objects = object_client.list_objects(namespace, bucket_name).data.objects
    
    for obj in objects:
        file_name = obj.name.split("/")[-1]
        if file_name.lower().endswith(('.jpg', '.png')):
            file_path = os.path.join(download_dir, file_name)
            response = object_client.get_object(namespace, bucket_name, obj.name)
            with open(file_path, 'wb') as file:
                file.write(response.data.content)

# Function to augment and distribute images
def augment_and_distribute(input_dir, output_dir):
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)

    os.makedirs(output_dir, exist_ok=True)
    train_class_dirs = [os.path.join(TRAIN_DIR, "class1"), os.path.join(TRAIN_DIR, "class2")]
    val_class_dirs = [os.path.join(VALIDATION_DIR, "class1"), os.path.join(VALIDATION_DIR, "class2")]

    for dir_path in train_class_dirs + val_class_dirs:
        os.makedirs(dir_path, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    mid = len(files) // 2

    for i, file_name in enumerate(files):
        image = Image.open(os.path.join(input_dir, file_name))
        image_np = np.array(image)
        augmented = seq(image=image_np)
        output_image = Image.fromarray(augmented)
        
        if i < mid:
            output_path = os.path.join(train_class_dirs[i % 2], f"aug_{file_name}")
            output_path = os.path.join(val_class_dirs[i % 2], f"aug_{file_name}")
        else:
            output_path = os.path.join(train_class_dirs[i % 2], f"aug_{file_name}")
            output_path = os.path.join(val_class_dirs[i % 2], f"aug_{file_name}")

        output_image.save(output_path)

        # Upload to OCI Object Storage
        with open(output_path, 'rb') as image_file:
            object_client.put_object(NAMESPACE, BUCKET_PROCESSED, output_path.replace(DATA_DIR + '/', ''), image_file)

# Prepare data generators
def prepare_data():
    train_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary")

    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        VALIDATION_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary")

    return train_gen, val_gen

# Model definition
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and save the model
def train_and_save_model(model, train_gen, val_gen):
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
    model.save(MODEL_SAVE_PATH)

# Upload model to OCI Object Storage
def upload_model():
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    with open(MODEL_SAVE_PATH, 'rb') as model_file:
        object_client.put_object(NAMESPACE, BUCKET_MODEL, "model.keras", model_file)

# Main execution
if __name__ == "__main__":
    raw_images_dir = "./raw_images"

    print("Downloading raw images...")
    download_images(NAMESPACE, BUCKET_RAW, raw_images_dir)

    print("Augmenting and distributing images...")
    augment_and_distribute(raw_images_dir, DATA_DIR)

    print("Preparing data...")
    train_generator, validation_generator = prepare_data()

    print("Building and training model...")
    cnn_model = build_model()
    train_and_save_model(cnn_model, train_generator, validation_generator)

    print("Uploading trained model...")
    upload_model()

    print("Process completed.")