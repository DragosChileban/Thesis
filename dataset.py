import torch
import os
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image
import numpy as np
import utils
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask

class COCOSegmentation(torch.utils.data.Dataset):
    def __init__(self, args, base_dir='/Users/dragos/Licenta/CarDD_release/CarDD_COCO', split='train', year='2017', my_transforms=None):
        super().__init__()
        ann_file = os.path.join(base_dir, f'annotations/instances_{split}{year}.json')
        self.img_dir = os.path.join(base_dir, f'{split}{year}')
        self.split = split
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())  # Get all image IDs
        self.args = args
        self.transforms = my_transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Generate masks & bounding boxes
        masks = []
        boxes = []
        labels = []
        
        for ann in anns:
            mask = coco.annToMask(ann)  # Convert segmentation to binary mask
            if mask.sum() == 0:  # Skip empty masks
                continue

            masks.append(mask)
            labels.append(ann["category_id"])

            # Convert bbox format to [x_min, y_min, x_max, y_max]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

        # Convert lists to tensors
        if len(masks) == 0:  # Handle images with no annotations
            masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
        else:
            masks = Mask(torch.as_tensor(masks, dtype=torch.uint8))
            boxes = BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(image.height, image.width))
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Ensure bounding boxes have correct shape
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes,  # Shape [N, 4]
            "labels": labels,  # Shape [N]
            "masks": masks,  # Shape [N, H, W]
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),  # Compute areas
        }

        if self.transforms: image, target = self.transforms(image, target)
                
        return image, target

    def __len__(self):
        return len(self.ids)