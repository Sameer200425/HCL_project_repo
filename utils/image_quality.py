"""
Image Quality Assessment (IQA) Utilities
=========================================
Real-time checks for document quality before processing.
Detects:
1. Glare / Reflection (Bright spots)
2. Darkness (Low luminance)
3. Blurriness (Out of focus)
4. Overexposure (Too bright overall)

Usage:
    checker = ImageQualityChecker()
    is_good, issues = checker.analyze(image_array)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict

class ImageQualityChecker:
    def __init__(self):
        # Thresholds (tunable based on deployment environment)
        self.BLUR_THRESHOLD = 50.0        # Laplacian variance (lowered to allow slightly fuzzy scans)
        self.DARKNESS_THRESHOLD = 20.0    # Mean brightness
        self.BRIGHTNESS_THRESHOLD = 254.0 # Raised significantly so digital white backgrounds don't trigger it
        self.GLARE_THRESHOLD = 254        # Pixel value for saturation
        self.GLARE_AREA_RATIO = 0.95      # Max allowed saturated area (allow up to 95% white for digital cheques)

    def analyze(self, image_np: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Analyze image quality.
        Args:
            image_np: Image as numpy array (RGB or BGR)
        Returns:
            passed (bool): True if quality is acceptable
            issues (List[str]): List of detected problems (e.g., "Too blurry")
        """
        issues = []
        
        if image_np is None:
            return False, ["Invalid image data"]
            
        # Convert to grayscale for analysis
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) # Assuming BGR from cv2.imdecode
        else:
            gray = image_np

        # 1. Check for Blurriness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.BLUR_THRESHOLD:
            issues.append("Image is too blurry. Please hold steady.")

        # 2. Check for Darkness
        mean_brightness = np.mean(gray)
        if mean_brightness < self.DARKNESS_THRESHOLD:
            issues.append("Image is too dark. Please improved lighting.")
        elif mean_brightness > self.BRIGHTNESS_THRESHOLD:
            issues.append("Image is overexposed (too bright).")

        # 3. Check for Glare / Direct Light Reflection
        # Look for areas that are fully saturated (near white)
        # Create a mask of saturated pixels
        _, bright_mask = cv2.threshold(gray, self.GLARE_THRESHOLD, 255, cv2.THRESH_BINARY)
        saturated_pixels = cv2.countNonZero(bright_mask)
        total_pixels = gray.shape[0] * gray.shape[1]
        saturation_ratio = saturated_pixels / total_pixels

        if saturation_ratio > self.GLARE_AREA_RATIO:
            issues.append("Glare detected — move away from direct light.")

        passed = len(issues) == 0
        return passed, issues
