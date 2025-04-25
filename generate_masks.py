from dataset import COCOSegmentation
import utils
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import torch
from visualize import visualize_prediction
from visualize import resize_colmap_images
import cv2
import numpy as np

from ultralytics import YOLO
model_path = '/Users/dragos/Licenta/Results/YOLO/best.pt'
model = YOLO(model_path)
device = torch.device('mps')
model.to(device)

def save_masks(path, first, last):
    for i in range(first, last):
        image_path = f'{path}/images/{i:04d}.jpg'
        print(image_path)
        results = model(image_path)
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data[0]
                masks = masks.cpu().numpy()
                mask_path = f'{path}/masks/{i:04d}.png'
                cv2.imwrite(mask_path, masks.astype(np.uint8))
                print(f"[+] Saved mask for image {i:04d} to {mask_path}")
            else:
                print(f"[!] No results for image {i:04d}")

# save_masks(1, 5)
