import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from rembg import remove

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
    
    output = remove(img)
    # Ensure output matches input dimensions
    if output.shape[:2] != img.shape[:2]:
        output = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Ensure it's a writable copy
    output = output.copy()
    
    mask = output[:, :, 3]
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return output, binary_mask

def match_color_metrics(src_img, src_mask, tgt_img, strength=0.1):
    """
    Adjusts the Brightness, Contrast, and Saturation of the source image 
    to match the target image, using the HSV color space.
    """
    print("Adjusting person's contrast, saturation, and brightness to match the new background...")
    
    # Convert to HSV (Hue, Saturation, Value)
    src_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV).astype("float32")
    tgt_hsv = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2HSV).astype("float32")
    
    # Calculate metrics for the person, and for the target background
    src_mean, src_std = cv2.meanStdDev(src_hsv, mask=src_mask)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_hsv) # Target is the whole clean background
    
    h, s, v = cv2.split(src_hsv)
    
    # Blend the target metrics with original metrics to prevent extreme washing out
    target_std_v = src_std[2][0] * (1 - strength) + tgt_std[2][0] * strength
    target_mean_v = src_mean[2][0] * (1 - strength) + tgt_mean[2][0] * strength
    target_mean_s = src_mean[1][0] * (1 - strength) + tgt_mean[1][0] * strength
    
    # 1. Match Brightness (Mean) and Contrast (Std Dev) on the V channel
    v_adj = ((v - src_mean[2][0]) * (target_std_v / (src_std[2][0] + 1e-5))) + target_mean_v
    v_adj = np.clip(v_adj, 0, 255)
    
    # 2. Match Saturation on the S channel
    s_adj = s * (target_mean_s / (src_mean[1][0] + 1e-5))
    s_adj = np.clip(s_adj, 0, 255)
    
    # Re-merge channels (Hue remains untouched to preserve original clothing/skin tones)
    h_merged = np.asarray(h, dtype=np.float32)
    s_merged = np.asarray(s_adj, dtype=np.float32)
    v_merged = np.asarray(v_adj, dtype=np.float32)
    
    matched_hsv = cv2.merge([h_merged, s_merged, v_merged]).astype("uint8")
    return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)


def create_final_composition(result_img, source_img, target_img):
    """
    Creates a composition with the result image on top and 
    the two source images side-by-side at the bottom.
    Fits the final image to a standard screen size (1920x1080).
    """
    # 1. Prepare the bottom strip (sources)
    h_res, w_res = result_img.shape[:2]
    h_strip = h_res // 4
    
    def resize_h(img, target_h):
        h, w = img.shape[:2]
        new_w = int(w * (target_h / h))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    s1 = resize_h(source_img, h_strip)
    s2 = resize_h(target_img, h_strip)
    
    # Gap between source images
    gap = 40
    strip_w = s1.shape[1] + s2.shape[1] + gap
    strip = np.zeros((h_strip, strip_w, 3), dtype=np.uint8)
    strip[:, :s1.shape[1]] = s1
    strip[:, s1.shape[1]+gap:] = s2
    
    # Resize strip to match result width
    strip_h_final = int(h_strip * (w_res / strip_w))
    strip_resized = cv2.resize(strip, (w_res, strip_h_final), interpolation=cv2.INTER_AREA)
    
    # 2. Combine vertically with padding
    v_gap = 40
    padding = np.zeros((v_gap, w_res, 3), dtype=np.uint8)
    combined = cv2.vconcat([result_img, padding, strip_resized])
    
    # 3. Fit to screen size (1920x1080)
    max_w, max_h = 1920, 1080
    c_h, c_w = combined.shape[:2]
    scale = min(max_w / c_w, max_h / c_h)
    
    final_w = int(c_w * scale)
    final_h = int(c_h * scale)
    final_img = cv2.resize(combined, (final_w, final_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4)
    
    return final_img



