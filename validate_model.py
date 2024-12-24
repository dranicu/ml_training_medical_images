import os
import oci
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
namespace = "ocisateam" # SET THE NAMESPACE
IMG_HEIGHT = 150  # Same as used during training
IMG_WIDTH = 150   # Same as used during training
BATCH_SIZE = 1
MODEL_PATH = "./model"  # Path to the saved model
DATA_DIR = "./data/validation"  # Validation directory

def download_images(namespace, prefix, bucket_name, download_dir):
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config = {}, signer=signer )
    
    # Ensure download directory exists
    os.makedirs(f"{download_dir}/class1", exist_ok=True)
    os.makedirs(f"{download_dir}/class2", exist_ok=True)

    # List objects in the bucket
    objects = object_client.list_objects(
        namespace,
        bucket_name,
        prefix=prefix,
    )
    count = 0
    for obj in objects.data.objects:
        file_name = obj.name.split("/")[-1]
        
        if file_name.lower().endswith(('.jpg', '.png')):  # Filter jpg/png files
            if float(count) < len(objects.data.objects)/2:
                file_path = os.path.join(f"{download_dir}/class1", file_name)
            else:
                file_path = os.path.join(f"{download_dir}/class2", file_name)
            print(f"Downloading {file_name}...")
            # Get object data
            get_obj_response = object_client.get_object(namespace, bucket_name, obj.name)
            # Save the file locally
            with open(file_path, 'wb') as file:
                file.write(get_obj_response.data.content)
            print(f"Downloaded: {file_path}")
        count += 1

def download_model(download_dir, bucket_name):
    model_name = "model.keras"
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config = {}, signer=signer )

    os.makedirs(f"{download_dir}", exist_ok=True)
    get_obj_response = object_client.get_object(namespace, bucket_name, model_name)
    model_path = os.path.join(f"{download_dir}", model_name)

    with open(model_path, 'wb') as file:
        file.write(get_obj_response.data.content)
        print(f"Downloaded: {model_path}")

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
    prefix = "validation"
    bucket_name_images = "medical-images"
    bucket_name_model = "trained-model"

    download_dir = DATA_DIR
    download_model(MODEL_PATH, bucket_name_model)
    download_images(namespace, prefix, bucket_name_images, download_dir)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
    elif not os.path.exists(DATA_DIR):
        print(f"Error: Validation directory not found at {DATA_DIR}")
    else:
        model_path = os.path.join(f"{MODEL_PATH}", "model.keras")
        validate_model(download_dir, model_path)
