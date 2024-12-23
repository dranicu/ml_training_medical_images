import os
import oci
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

namespace = "<NAMESPACE>" # SET THE NAMESPACE

# Set up augmentation parameters
seq = iaa.Sequential([
    iaa.Fliplr(1),  # horizontal flips
    iaa.Rotate((-45, 45)),  # random rotations (-45 to 45 degrees)
    iaa.GaussianBlur(sigma=(0, 2.0)),  # gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),  # gaussian noise
    iaa.Dropout(p=(0, 0.2)),  # random dropout
    iaa.Resize({"height": (0.5, 1.5), "width": (0.5, 1.5)}),  # random scaling
    iaa.Crop(percent=(0, 0.2)),  # random cropping
    iaa.ElasticTransformation(alpha=(0, 10.0), sigma=1.0),  # elastic transformation
    iaa.PiecewiseAffine(scale=(0.02, 0.1)),  # piecewise affine transformations
])

# Function to download images from OCI Object Storage using a PAR URL
def download_images(namespace, bucket_name, download_dir):
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config = {}, signer=signer )

    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # List objects in the bucket
    objects = object_client.list_objects(
        namespace,
        bucket_name,
    )

    for obj in objects.data.objects:
        file_name = obj.name.split("/")[-1]
        if file_name.lower().endswith(('.jpg', '.png')):  # Filter jpg/png files
            print(f"Downloading {file_name}...")
            # Get object data
            get_obj_response = object_client.get_object(namespace, bucket_name, obj.name)
            # Save the file locally
            file_path = os.path.join(download_dir, file_name)
            with open(file_path, 'wb') as file:
                file.write(get_obj_response.data.content)
            print(f"Downloaded: {file_path}")
    

# Function to upload processed images back to OCI Object Storage
def upload_images(namespace, bucket_name, upload_dir):
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    object_client = oci.object_storage.ObjectStorageClient(config = {}, signer=signer )
    nr_of_images = len([file for file in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, file))])
    count = 0
    for root, _, files in os.walk(upload_dir):
        for file_name in files:
            if float(count) < nr_of_images/2:
                object_name = os.path.join("train", file_name).replace("\\", "/")
            else:
                object_name = os.path.join("validation", file_name).replace("\\", "/")

            count += 1

            file_path = os.path.join(root, file_name)
                                   
            print(f"Uploading {file_path} to {object_name} in bucket {bucket_name}...")

            with open(file_path, "rb") as file_data:
                # Upload file
                object_client.put_object(
                    namespace,
                    bucket_name,
                    object_name,
                    file_data
                )
            print(f"Uploaded {file_name} successfully.")

# Function to apply augmentations to images
def augment_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_file in os.listdir(input_dir):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path)
            image_np = np.array(image)
            augmented_image_np = seq(image=image_np)  # Apply augmentations
            augmented_image = Image.fromarray(augmented_image_np)
            augmented_image.save(os.path.join(output_dir, f"aug_{image_file}"))

# Main execution
if __name__ == "__main__":
    bucket_name_up = "medical-images-processed"
    bucket_name_down = "medical-images-raw"
    download_dir = "./input_images"
    output_dir = "./augmented_images"
    
    # Step 1: Download images from OCI Object Storage
    print("Downloading images...")
    download_images(namespace, bucket_name_down, download_dir)
    
    # Step 2: Apply augmentations
    print("Applying augmentations...")
    augment_images(download_dir, output_dir)
    
    # Step 3: Upload augmented images back to OCI Object Storage
    print("Uploading augmented images...")
    upload_images(namespace, bucket_name_up, output_dir)
    
    print("Image augmentation and upload completed.")
