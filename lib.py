import os
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from submodules.screen_capture.lib import capture

def describe_ui(windowId):
    # Capture screenshot of the selected window
    screenshot = capture(windowId)

    # Convert the PIL Image to a numpy array and normalize it
    screenshot_np = np.array(screenshot) / 255.0

    # Resize the image to dimensions divisible by 32
    height, width, _ = screenshot_np.shape
    height = (height // 32) * 32
    width = (width // 32) * 32
    screenshot_np = cv2.resize(screenshot_np, (width, height))

    # Convert the numpy array to a torch tensor
    screenshot_tensor = torch.from_numpy(screenshot_np).permute(2, 0, 1).unsqueeze(0).float()

    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8x.pt")

    # Use the model to detect objects in the screenshot
    results = model(screenshot_tensor)

    return results