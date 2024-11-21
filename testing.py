import os
import requests

def download_images(par_url, download_dir):
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(par_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch file list: {response.status_code}, {response.text}")
    
    try:
        file_list = response.json()  # Expecting a JSON list of file URLs
    except ValueError:
        raise Exception("Response is not JSON. Check PAR URL response format.")
    
    for file_url in file_list:
        file_name = os.path.basename(file_url)
        file_path = os.path.join(download_dir, file_name)

        file_response = requests.get(file_url)
        if file_response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(file_response.content)
            print(f"Downloaded {file_name} to {file_path}")
        else:
            print(f"Failed to download {file_url}: {file_response.status_code}")

# Example usage
download_par_url = "https://objectstorage.eu-frankfurt-1.oraclecloud.com/p/6I1FGcBRAAk0rZIa0A3JjaDXziJymT0fkvfjxTmQlnIommXbhl4RbpYq5Ll__0R_/n/ocisateam/b/medical-image-bucket/o/non-cancer/"
download_dir = "./input_images"
download_images(download_par_url, download_dir)
