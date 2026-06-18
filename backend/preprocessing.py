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

def preprocess_frame(frame, target_size=YOLO_INPUT_SIZE):
    """
    Full preprocessing pipeline for a single frame.
    """
    # 1. Resize for YOLO
    resized = cv2.resize(frame, target_size)

    # 2. Low-light enhancement (CLAHE)
    enhanced = apply_clahe(resized)

    return enhanced, resized
