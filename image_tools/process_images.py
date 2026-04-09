import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pillow_heif import register_heif_opener

def convert_heic_to_jpg(heic_path):
    import os
    import subprocess
    jpg_path = heic_path.rsplit('.', 1)[0] + ".jpg"
    if not os.path.exists(jpg_path):
        print(f"Converting {heic_path} to {jpg_path}...")
        subprocess.run(["sips", "-s", "format", "jpeg", heic_path, "--out", jpg_path], check=True)
    return jpg_path

def get_person_mask(image_path):
    from rembg import remove
    register_heif_opener()
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

def merge_segments(p1_crop, p2_crop, extend_direction='bottom'):
    """
    Merges two segmented person crops.
    extend_direction: 'bottom' (default) or 'top'
    """
    h1, w1 = p1_crop.shape[:2]
    h2, w2 = p2_crop.shape[:2]

    # Scale P1 to match P2's width for alignment
    scale = w2 / float(w1)
    new_w1 = w2
    new_h1 = int(h1 * scale)
    p1_scaled = cv2.resize(p1_crop, (new_w1, new_h1))

    # Canvas size accommodates both layers
    canvas_w = w2
    canvas_h = max(h2, new_h1)
    
    p1_float = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    p1_float[0:new_h1, 0:new_w1] = p1_scaled.astype(np.float32) / 255.0

    p2_float = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    p2_float[0:h2, 0:w2] = p2_crop.astype(np.float32) / 255.0

    blend_zone = int(new_h1 * 0.05) # 15% smooth transition area
    
    if extend_direction == 'bottom':
        # Erase P2's top (shoulders, head, upper torso) so it doesn't overlap behind P1
        fade_start = new_h1 - blend_zone
        if fade_start > 0:
            p2_float[0:fade_start, :, 3] = 0.0 
            
        # Apply a gradient fade out to P1's bottom edge
        fade = np.ones((new_h1, 1), dtype=np.float32)
        if blend_zone > 0:
            fade[-blend_zone:] = np.linspace(1.0, 0.0, blend_zone)[:, None]
        fade_2d = np.tile(fade, (1, new_w1))
        p1_float[0:new_h1, 0:new_w1, 3] *= fade_2d
        
    elif extend_direction == 'top':
        # Erase P2's bottom
        fade_start = blend_zone
        p2_float[fade_start:, :, 3] = 0.0
        
        # Apply a gradient fade out to P1's top edge
        fade = np.ones((new_h1, 1), dtype=np.float32)
        if blend_zone > 0:
            fade[:blend_zone] = np.linspace(0.0, 1.0, blend_zone)[:, None]
        fade_2d = np.tile(fade, (1, new_w1))
        p1_float[0:new_h1, 0:new_w1, 3] *= fade_2d

    # Blend P1 over P2
    a1 = np.expand_dims(p1_float[:, :, 3], axis=-1)
    a2 = np.expand_dims(p2_float[:, :, 3], axis=-1)
    
    # Pre-multiplied Alpha Blending
    out_rgb = p1_float[:, :, :3] * a1 + p2_float[:, :, :3] * a2 * (1 - a1)
    out_alpha = a1 + a2 * (1 - a1)
    
    merged_float = np.concatenate([out_rgb, out_alpha], axis=-1)
    
    # Divide RGB by Alpha to get straight RGB colors for final pasting (avoids dark fringes)
    mask = out_alpha[:, :, 0] > 0
    merged_float[mask, :3] = merged_float[mask, :3] / out_alpha[mask]
    
    return (np.clip(merged_float, 0, 1) * 255).astype(np.uint8)

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


