from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import os
import numpy as np
import cv2

register_heif_opener()

def test_heic_loading(file_path):
    print(f"Testing {file_path}...")
    try:
        image = Image.open(file_path)
        print(f"  Successfully opened {file_path}")
        image = ImageOps.exif_transpose(image)
        print(f"  Applied exif_transpose")
        image = image.convert('RGB')
        img_np = np.array(image)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        print(f"  Successfully converted to OpenCV format. Shape: {img.shape}")
        return True
    except Exception as e:
        print(f"  Failed to load {file_path}: {e}")
        return False

if __name__ == "__main__":
    heic_files = [f for f in os.listdir('.') if f.lower().endswith('.heic')]
    if not heic_files:
        print("No HEIC files found in current directory.")
    else:
        for f in heic_files[:3]:  # Test first 3
            test_heic_loading(f)
