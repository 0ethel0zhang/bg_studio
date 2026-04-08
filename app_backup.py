import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import os

register_heif_opener()
import tempfile
import subprocess
from rembg import remove

# Import functions from image_tools scripts
from image_tools.process_images import merge_segments
from image_tools.process_background import match_color_metrics

# Title
st.set_page_config(page_title="Image Processing Tools", layout="wide")

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

st.title("Image Processing Studio")
st.markdown("<p style='color: gray; font-size: 0.9em;'>Note: The first time a user uploads an image on the hosted version, the app will take a few seconds longer to process because rembg will download its ~170MB U2Net model to the cloud server. Subsequent images will be processed much faster.</p>", unsafe_allow_html=True)

# Sidebar for tool selection
tool = st.sidebar.selectbox("Select Tool", ["Merge/Extend People", "Change Background", "Relocate Person"])

# Helper function for image loading
def load_image(uploaded_file):
    if uploaded_file is None:
        return None
    
    try:
        image = Image.open(uploaded_file)
        # Fix orientation issues (especially for mobile uploads)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        img_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        st.error(f"Failed to load image {uploaded_file.name}: {e}")
        return None

def get_person_mask(img):
    """Returns (output_bgra, binary_mask)"""
    output = remove(img)
    
    # Ensure output matches input dimensions
    if output.shape[:2] != img.shape[:2]:
        output = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Ensure it's a writable copy
    output = output.copy()
    
    mask = output[:, :, 3]
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return output, binary_mask

def match_aspect_ratio(img, target_ratio, mode='crop'):
    """
    Adjusts img to have target_ratio (w/h) by either 'crop' or 'pad'.
    """
    h, w = img.shape[:2]
    current_ratio = w / h
    
    if abs(current_ratio - target_ratio) < 0.01:
        return img
    
    if mode == 'crop':
        if current_ratio > target_ratio:
            # Current is wider than target - crop width
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            return img[:, start_x:start_x+new_w]
        else:
            # Current is taller than target - crop height
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            return img[start_y:start_y+new_h, :]
    
    elif mode == 'pad':
        if current_ratio > target_ratio:
            # Current is wider than target - pad height
            new_h = int(w / target_ratio)
            pad_y = (new_h - h) // 2
            # Add black padding
            return cv2.copyMakeBorder(img, pad_y, new_h - h - pad_y, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            # Current is taller than target - pad width
            new_w = int(h * target_ratio)
            pad_x = (new_w - w) // 2
            # Add black padding
            return cv2.copyMakeBorder(img, 0, 0, pad_x, new_w - w - pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return img

# Initialize session state for the result image
if 'result_img' not in st.session_state:
    st.session_state.result_img = None
if 'result_filename' not in st.session_state:
    st.session_state.result_filename = "result.jpg"

# Main Layout
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.subheader("Source Images")
    source_file = st.file_uploader("Upload Source Image (Person 1)", type=["jpg", "jpeg", "png", "heic"])
    if source_file:
        source_img = load_image(source_file)
        if source_img is not None:
            h, w = source_img.shape[:2]
            st.write(f"Size: {w}x{h} (Aspect Ratio: {w/h:.2f})")
            st.image(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB), width='content', caption="Source Image 1")
    else:
        source_img = None

    target_file = st.file_uploader("Upload Target Image (Person 2 or Background)", type=["jpg", "jpeg", "png", "heic"])
    if target_file:
        target_img = load_image(target_file)
        if target_img is not None:
            h, w = target_img.shape[:2]
            st.write(f"Size: {w}x{h} (Aspect Ratio: {w/h:.2f})")
            st.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), width='stretch', caption="Source Image 2 / Background")
    else:
        target_img = None

with main_col2:
    st.subheader("Result")
    if st.session_state.result_img is not None:
        st.image(cv2.cvtColor(st.session_state.result_img, cv2.COLOR_BGR2RGB), width="stretch", caption="Generated Image")
        # Download button for the result image
        _, buffer = cv2.imencode(".jpg", st.session_state.result_img)
        st.download_button("Download Result Image", data=buffer.tobytes(), file_name=st.session_state.result_filename, mime="image/jpeg")
    else:
        st.info("The generated image will appear here once processed.")

# Processing Logic
if source_img is not None and target_img is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Processing Options")
    
    if tool == "Merge/Extend People":
        direction = st.sidebar.radio("Extend Direction", ["bottom", "top"], index=0)
        
        if st.sidebar.button("Process & Merge"):
            msg_placeholder = st.empty()
            if st.session_state.first_run:
                msg_placeholder.markdown("<p style='color: gray; font-size: 0.9em;'>Note: The first time a user uploads an image on the hosted version, the app will take a few seconds longer to process because rembg will download its ~170MB U2Net model to the cloud server. Subsequent images will be processed much faster.</p>", unsafe_allow_html=True)
            with st.spinner("Isolating people and merging..."):
                # Extract Person 1
                person1_bgra, mask1 = get_person_mask(source_img)
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours1:
                    c1 = max(contours1, key=cv2.contourArea)
                    x1, y1, w1, h1 = cv2.boundingRect(c1)
                    p1_crop = person1_bgra[y1:y1+h1, x1:x1+w1]
                    
                    # Extract Person 2
                    person2_bgra, mask2 = get_person_mask(target_img)
                    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours2:
                        c2 = max(contours2, key=cv2.contourArea)
                        x2, y2, w2, h2 = cv2.boundingRect(c2)
                        p2_crop = person2_bgra[y2:y2+h2, x2:x2+w2]
                        
                        # Inpaint Target Background
                        kernel = np.ones((15, 15), np.uint8)
                        mask2_dilated = cv2.dilate(mask2, kernel, iterations=1)
                        img2_inpainted = cv2.inpaint(target_img, mask2_dilated, 5, cv2.INPAINT_TELEA)
                        
                        # Merge
                        merged_person_uint8 = merge_segments(p1_crop, p2_crop, extend_direction=direction)
                        
                        # Paste back into scene
                        h_m, w_m = merged_person_uint8.shape[:2]
                        bg_h, bg_w = img2_inpainted.shape[:2]
                        
                        start_y, start_x = max(0, y2), max(0, x2)
                        end_y, end_x = min(bg_h, y2 + h_m), min(bg_w, x2 + w_m)
                        
                        paste_h, paste_w = end_y - start_y, end_x - start_x
                        
                        if paste_h > 0 and paste_w > 0:
                            src_start_y = 0 if y2 >= 0 else -y2
                            src_start_x = 0 if x2 >= 0 else -x2
                            src_roi = merged_person_uint8[src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w]
                            dst_roi = img2_inpainted[start_y:end_y, start_x:end_x]
                            
                            alpha = src_roi[:, :, 3] / 255.0
                            alpha_exp = np.expand_dims(alpha, axis=-1)
                            
                            blended = src_roi[:, :, :3] * alpha_exp + dst_roi * (1 - alpha_exp)
                            img2_inpainted[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
                            
                            st.session_state.result_img = img2_inpainted
                            st.session_state.result_filename = "merged_person.jpg"
                            st.rerun()
                        else:
                            st.error("Merge area out of bounds.")
                    else:
                        st.error("No person found in target image.")
                else:
                    st.error("No person found in source image.")

            msg_placeholder.empty()
            st.session_state.first_run = False

    elif tool == "Change Background":
        strength = st.sidebar.slider("Color Matching Strength", 0.0, 1.0, 0.75)
        match_mode = st.sidebar.radio("Match Aspect Ratio", ["Stretch (Default)", "Crop Target", "Pad Target"], index=2)
        
        if st.sidebar.button("Change Background"):
            msg_placeholder = st.empty()
            if st.session_state.first_run:
                msg_placeholder.markdown("<p style='color: gray; font-size: 0.9em;'>Note: The first time a user uploads an image on the hosted version, the app will take a few seconds longer to process because rembg will download its ~170MB U2Net model to the cloud server. Subsequent images will be processed much faster.</p>", unsafe_allow_html=True)
            with st.spinner("Processing background change..."):
                # Source Person
                img1_bgra, mask1 = get_person_mask(source_img)
                
                # Target Background (might have its own person to remove)
                h1, w1 = source_img.shape[:2]
                if match_mode == "Crop Target":
                    target_img_processed = match_aspect_ratio(target_img, w1/h1, mode='crop')
                elif match_mode == "Pad Target":
                    target_img_processed = match_aspect_ratio(target_img, w1/h1, mode='pad')
                else:
                    target_img_processed = target_img.copy()

                img2_bgra, mask2 = get_person_mask(target_img_processed)
                
                # Resize target to match source EXACTLY for pixel-perfect placement
                h2, w2 = target_img_processed.shape[:2]
                if (h1, w1) != (h2, w2):
                    target_img_resized = cv2.resize(target_img_processed, (w1, h1))
                    mask2_resized = cv2.resize(mask2, (w1, h1))
                else:
                    target_img_resized = target_img_processed.copy()
                    mask2_resized = mask2.copy()
                
                # Inpaint target if person exists
                contours, _ = cv2.findContours(mask2_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1000:
                    kernel = np.ones((15, 15), np.uint8)
                    mask2_dilated = cv2.dilate(mask2_resized, kernel, iterations=1)
                    clean_bg = cv2.inpaint(target_img_resized, mask2_dilated, 5, cv2.INPAINT_TELEA)
                else:
                    clean_bg = target_img_resized.copy()
                
                # Color match
                img1_matched = match_color_metrics(source_img, mask1, clean_bg, strength=strength)
                
                # Paste
                person1_pixels = mask1 > 127
                clean_bg[person1_pixels] = img1_matched[person1_pixels]
                
                st.session_state.result_img = clean_bg
                st.session_state.result_filename = "new_background.jpg"
                st.rerun()

            msg_placeholder.empty()
            st.session_state.first_run = False

    elif tool == "Relocate Person":
        match_mode = st.sidebar.radio("Match Aspect Ratio", ["Original", "Crop Target", "Pad Target"], index=1)
        if st.sidebar.button("Relocate"):
            msg_placeholder = st.empty()
            if st.session_state.first_run:
                msg_placeholder.markdown("<p style='color: gray; font-size: 0.9em;'>Note: The first time a user uploads an image on the hosted version, the app will take a few seconds longer to process because rembg will download its ~170MB U2Net model to the cloud server. Subsequent images will be processed much faster.</p>", unsafe_allow_html=True)
            with st.spinner("Relocating person..."):
                # Handle aspect ratio match if selected
                h1_orig, w1_orig = source_img.shape[:2]
                if match_mode == "Crop Target":
                    target_img_processed = match_aspect_ratio(target_img, w1_orig/h1_orig, mode='crop')
                elif match_mode == "Pad Target":
                    target_img_processed = match_aspect_ratio(target_img, w1_orig/h1_orig, mode='pad')
                else:
                    target_img_processed = target_img.copy()

                # Source Person
                person_bgra, mask1 = get_person_mask(source_img)
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours1:
                    c1 = max(contours1, key=cv2.contourArea)
                    x1, y1, w1_c, h1_c = cv2.boundingRect(c1)
                    person_crop = person_bgra[y1:y1+h1_c, x1:x1+w1_c]
                    
                    # Target Person (in the potentially resized/cropped/padded target)
                    _, mask2 = get_person_mask(target_img_processed)
                    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours2:
                        c2 = max(contours2, key=cv2.contourArea)
                        x2, y2, w2_c, h2_c = cv2.boundingRect(c2)
                        
                        # Inpaint
                        kernel = np.ones((15, 15), np.uint8)
                        mask2_dilated = cv2.dilate(mask2, kernel, iterations=1)
                        clean_bg = cv2.inpaint(target_img_processed, mask2_dilated, 5, cv2.INPAINT_TELEA)
                        
                        # Scale to match height
                        scale = h2_c / float(h1_c)
                        new_w = int(w1_c * scale)
                        new_h = h2_c
                        person_resized = cv2.resize(person_crop, (new_w, new_h))
                        
                        paste_x = x2 + (w2_c - new_w) // 2
                        paste_y = y2
                    else:
                        # Center Bottom
                        clean_bg = target_img_processed.copy()
                        new_h = int(target_img_processed.shape[0] * 0.8)
                        scale = new_h / float(h1_c)
                        new_w = int(w1_c * scale)
                        person_resized = cv2.resize(person_crop, (new_w, new_h))
                        paste_x = (target_img_processed.shape[1] - new_w) // 2
                        paste_y = target_img_processed.shape[0] - new_h
                    
                    # Composite (Simplified)
                    h_p, w_p = person_resized.shape[:2]
                    h_bg, w_bg = clean_bg.shape[:2]
                    
                    y1_p, y2_p = max(0, paste_y), min(h_bg, paste_y + h_p)
                    x1_p, x2_p = max(0, paste_x), min(w_bg, paste_x + w_p)
                    
                    # Source crop from person_resized
                    py1 = 0 if paste_y >= 0 else -paste_y
                    px1 = 0 if paste_x >= 0 else -paste_x
                    py2 = py1 + (y2_p - y1_p)
                    px2 = px1 + (x2_p - x1_p)
                    
                    if y2_p > y1_p and x2_p > x1_p:
                        src_roi = person_resized[py1:py2, px1:px2]
                        dst_roi = clean_bg[y1_p:y2_p, x1_p:x2_p]
                        
                        alpha = src_roi[:, :, 3] / 255.0
                        alpha_exp = np.expand_dims(alpha, axis=-1)
                        
                        blended = src_roi[:, :, :3] * alpha_exp + dst_roi * (1 - alpha_exp)
                        clean_bg[y1_p:y2_p, x1_p:x2_p] = blended.astype(np.uint8)
                        
                        st.session_state.result_img = clean_bg
                        st.session_state.result_filename = "relocated_person.jpg"
                        st.rerun()
                else:
                    st.error("No person found in source image.")
                    
            msg_placeholder.empty()
            st.session_state.first_run = False

