import os
import subprocess
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import numpy as np
import cv2
from rembg import remove
from image_tools.process_background import match_color_metrics

register_heif_opener()

def convert_heic_to_jpg(heic_path):
    jpg_path = heic_path.rsplit('.', 1)[0] + ".jpg"
    if not os.path.exists(jpg_path):
        print(f"Converting {heic_path} to {jpg_path}...")
        subprocess.run(["sips", "-s", "format", "jpeg", heic_path, "--out", jpg_path], check=True)
    return jpg_path

def get_person_mask(image_path):
    # Use PIL to read image and handle EXIF orientation
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    img_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Remove background using rembg
    output = remove(img)
    
    # Ensure output matches input dimensions
    if output.shape[:2] != img.shape[:2]:
        output = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Ensure it's a writable copy
    output = output.copy()
    
    # output is a 4-channel image (BGRA) where the alpha channel is the mask
    mask = output[:, :, 3]
    # Create a binary mask (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return output, binary_mask

def main():
    # 1. Convert HEIC images
    img1_heic = "IMG_5699.HEIC"
    img2_heic = "IMG_5700.HEIC"
    
    img1_jpg = convert_heic_to_jpg(img1_heic)
    img2_jpg = convert_heic_to_jpg(img2_heic)

    # 2. Extract person from IMG_5699 (Source)
    print("Isolating people from IMG_5699 using rembg...")
    person_bgra, mask1 = get_person_mask(img1_jpg)
    
    # 3. Process IMG_5700 (Target)
    print("Processing target image IMG_5700...")
    # Use PIL to read image and handle EXIF orientation
    target_image = Image.open(img2_jpg)
    target_image = ImageOps.exif_transpose(target_image)
    target_image = target_image.convert('RGB')
    target_img_np = np.array(target_image)
    target_img = cv2.cvtColor(target_img_np, cv2.COLOR_RGB2BGR)

    # NEW: Color match person to target image
    print("Matching colors...")
    # Read original source for matching
    src_image = Image.open(img1_jpg)
    src_image = ImageOps.exif_transpose(src_image)
    src_image = src_image.convert('RGB')
    src_img_np = np.array(src_image)
    src_img_bgr = cv2.cvtColor(src_img_np, cv2.COLOR_RGB2BGR)
    
    img1_matched = match_color_metrics(src_img_bgr, mask1, target_img, strength=0.75)
    person_bgra[:, :, :3] = img1_matched
    
    h2, w2 = target_img.shape[:2]
    h1, w1 = person_bgra.shape[:2]

    # 4. Composite based on relative positioning
    print(f"Relocating people from {w1}x{h1} frame to {w2}x{h2} frame...")
    
    # NEW: Add padding to person_bgra to match target aspect ratio before resizing
    target_ratio = w2 / h2
    current_ratio = w1 / h1
    
    if abs(current_ratio - target_ratio) > 0.01:
        print(f"Adjusting aspect ratio from {current_ratio:.2f} to {target_ratio:.2f} with padding...")
        if current_ratio < target_ratio:
            # Source is narrower - pad width
            new_w = int(h1 * target_ratio)
            pad_x = (new_w - w1) // 2
            person_bgra = cv2.copyMakeBorder(person_bgra, 0, 0, pad_x, new_w - w1 - pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        else:
            # Source is wider - pad height
            new_h = int(w1 / target_ratio)
            pad_y = (new_h - h1) // 2
            person_bgra = cv2.copyMakeBorder(person_bgra, pad_y, new_h - h1 - pad_y, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    
    # Resize the entire person layer to match the target frame
    # This automatically preserves the relative position and scale of all people
    person_resized_bgra = cv2.resize(person_bgra, (w2, h2), interpolation=cv2.INTER_AREA)
    
    # Extract RGB and Alpha channels
    person_rgb = person_resized_bgra[:, :, :3]
    person_alpha = person_resized_bgra[:, :, 3] / 255.0
    person_alpha_3d = np.dstack([person_alpha]*3)

    # Blend
    result_img = (person_alpha_3d * person_rgb + (1 - person_alpha_3d) * target_img).astype(np.uint8)

    # Save result
    output_path = "new.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"Success! Result saved to {output_path}")

if __name__ == "__main__":
    main()
