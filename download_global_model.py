"""
Example script for hospital nodes to download the global model package
This script downloads both the model and scaler from the central server
"""
import requests
import zipfile
import io
import os
from pathlib import Path

# Configuration
CENTRAL_SERVER_URL = "http://127.0.0.1:8000"
DOWNLOAD_DIR = "./"

def download_global_package(server_url=CENTRAL_SERVER_URL, output_dir=DOWNLOAD_DIR):
    """
    Download the complete global model package (model + scaler) from central server.
    
    Args:
        server_url: URL of the central server
        output_dir: Directory to extract files to
    """
    print(f"Connecting to central server: {server_url}")
    
    try:
        # Download the package
        print("Downloading global model package...")
        response = requests.get(f"{server_url}/global/package")
        response.raise_for_status()
        
        # Extract the zip file
        print("Extracting files...")
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(output_dir)
        
        # List extracted files
        extracted_files = zip_file.namelist()
        print("\nSuccessfully downloaded and extracted:")
        for file in extracted_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size:,} bytes)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: Failed to connect to central server")
        print(f"Details: {e}")
        return False
    except zipfile.BadZipFile:
        print("\nError: Downloaded file is not a valid zip file")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def download_files_separately(server_url=CENTRAL_SERVER_URL, output_dir=DOWNLOAD_DIR):
    """
    Alternative method: Download model and scaler separately.
    
    Args:
        server_url: URL of the central server
        output_dir: Directory to save files to
    """
    print(f"Connecting to central server: {server_url}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download model
        print("Downloading global model...")
        response = requests.get(f"{server_url}/global/model")
        response.raise_for_status()
        
        model_filename = response.headers.get('content-disposition', '').split('filename=')[-1].strip('"')
        if not model_filename:
            model_filename = "main_model.pkl"
        
        model_path = os.path.join(output_dir, model_filename)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"  - Saved: {model_filename} ({len(response.content):,} bytes)")
        
        # Download scaler
        print("Downloading global scaler...")
        response = requests.get(f"{server_url}/global/scaler")
        response.raise_for_status()
        
        scaler_path = os.path.join(output_dir, "global_scaler.pkl")
        with open(scaler_path, "wb") as f:
            f.write(response.content)
        print(f"  - Saved: global_scaler.pkl ({len(response.content):,} bytes)")
        
        print("\nDownload complete!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: Failed to connect to central server")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Global Model Download Utility")
    print("=" * 60)
    print("\nOptions:")
    print("1. Download complete package (recommended)")
    print("2. Download model and scaler separately")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        success = download_global_package()
    elif choice == "2":
        success = download_files_separately()
    else:
        print("Invalid option. Please run the script again.")
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Global model files are ready to use!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Please check the error messages above.")
        print("=" * 60)
