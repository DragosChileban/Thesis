from dataset import COCOSegmentation
import utils
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms as transforms
import torch
import cv2
import os

def visualize_prediction(result, masks_only = False, show_boxes = True):

    image = result.orig_img
    masks = result.masks.data
    boxes = result.boxes.xyxy

    pred_height, pred_width = masks.shape[1:3]
    if masks_only is False:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(rgb_image)
        resized_image = transforms.Resize((pred_height, pred_width))(image_tensor)
        print(resized_image.shape)
    else:
        resized_image = torch.zeros((3, pred_height, pred_width))

    masks = masks > 0
    print(masks.shape)
    image_with_masks = draw_segmentation_masks(resized_image, masks.squeeze(1), colors="pink", alpha=0.7)
    
    if show_boxes:
        labels = []
        for i in range(result.boxes.cls.shape[0]):
            class_idx = int(result.boxes.cls[i].item())
            labels.append(result.names[class_idx])
        height, width = result.boxes.orig_shape[0:2]
        scale_x = pred_width / width
        scale_y = pred_height / height
        boxes_scaled = boxes.clone()
        boxes_scaled[:, [0, 2]] *= scale_x  
        boxes_scaled[:, [1, 3]] *= scale_y  
        boxes = boxes_scaled
        image_with_boxes = draw_bounding_boxes(image_with_masks, boxes, colors="red", labels=labels, width=2)

        return image_with_boxes
    else:
        return image_with_masks
    
def resize_colmap_images(path, save_path, width, height):
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (width, height))
            cv2.imwrite(os.path.join(save_path, filename), resized_image)
            print(f"Resized {filename} to {width}x{height}")
        else:
            continue