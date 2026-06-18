import cv2
import numpy as np
from config import YOLO_INPUT_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to improve visibility in low-light conditions.
    """
    # Convert image to LAB Color model
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def auto_gamma_correction(image):
    """
    Automatically adjust the gamma of an image based on its average brightness.
    This makes the pipeline adaptable to bright sunlight or dark night conditions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Target brightness (127 is mid-gray)
    target_brightness = 127.0
    
    # Calculate gamma
    # If image is dark (mean < 127), gamma < 1 (brightens image)
    # If image is bright (mean > 127), gamma > 1 (darkens image)
    # We clip gamma to prevent extreme washouts
    gamma = np.log(target_brightness / 255.0) / np.log(max(mean_brightness, 1) / 255.0)
    gamma = np.clip(gamma, 0.4, 2.5)
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def reduce_motion_blur(image):
    """
    Applies a sharpening filter to reduce the effects of motion blur,
    which is common in fast-moving traffic footage.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def preprocess_frame(frame, target_size=YOLO_INPUT_SIZE):
    """
    Full preprocessing pipeline for a single frame.
    """
    # 1. Resize for YOLO
    resized = cv2.resize(frame, target_size)

    # 2. Reduce Motion Blur (Sharpening)
    sharpened = reduce_motion_blur(resized)

    # 3. Auto-Gamma Correction (Adaptable Lighting)
    gamma_corrected = auto_gamma_correction(sharpened)

    # 4. Low-light enhancement (CLAHE)
    enhanced = apply_clahe(gamma_corrected)

    return enhanced, resized
