import numpy as np
import cv2

def convert(image, screen_height, screen_width):
    image = cv2.resize(image, (screen_height, screen_width))
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    
    return image