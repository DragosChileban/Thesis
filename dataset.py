import torch
import os
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image
import numpy as np
import utils
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask
import albumentations as A


class COCOSegmentation(torch.utils.data.Dataset):
    def __init__(self, base_dir='/Users/dragos/Licenta/CarDD_release/CarDD_COCO', split='train', year='2017', my_transforms=None):
        super().__init__()
        ann_file = os.path.join(base_dir, f'annotations/instances_{split}{year}.json')
        self.img_dir = os.path.join(base_dir, f'{split}{year}')
        self.split = split
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())  # Get all image IDs
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
            masks = torch.zeros((0, image.height, image.width), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
        else:
            masks = Mask(torch.as_tensor(np.array(masks), dtype=torch.float32))
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
        # Convert image to tensor
        masks = np.array(masks)
        image = np.array(image)
        boxes = np.array(boxes)
        labels = np.array(labels)

        # Apply transformations
        if self.transforms:
            height, width = image.shape[:2]
            if self.split == 'train':
                crop_size = min(height, width, 512)
                transforms = A.Compose([
                    # A.AtLeastOneBBoxRandomCrop(height=crop_size, width=crop_size, p=1.0),
                    A.Resize(height=512, width=512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # A.HorizontalFlip(p=0.5),
                ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
            if self.split == 'val':
                crop_size = min(height, width)
                transforms = A.Compose([
                    # A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
                    A.Resize(height=512, width=512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
            if self.split == 'test':
                crop_size = min(height, width)
                transforms = A.Compose([
                    # A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
                    A.Resize(height=512, width=512),
                ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
            transformed = transforms(
                image=image,
                masks=masks,
                bboxes=boxes,
                labels=labels
            )
            
            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to PyTorch tensors after transforms
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        masks = torch.as_tensor(np.array(masks), dtype=torch.float32)
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            "boxes": boxes,  # Shape [N, 4]
            "labels": labels-1,  # Shape [N]
            "masks": masks,  # Shape [N, H, W]
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Compute areas
        }
        
        # print("Image " + img_metadata["file_name"] + " has labesls: " + str(labels-1))

        return image, target

    def __len__(self):
        return len(self.ids)

# class COCOSegmentation(torch.utils.data.Dataset):
#     def __init__(self, base_dir='/Users/dragos/Licenta/CarDD_release/CarDD_COCO', split='train', year='2017', my_transforms=None):
#         super().__init__()
#         ann_file = os.path.join(base_dir, f'annotations/instances_{split}{year}.json')
#         self.img_dir = os.path.join(base_dir, f'{split}{year}')
#         self.split = split
#         self.coco = COCO(ann_file)
#         self.ids = list(self.coco.imgs.keys())
#         self.transforms = my_transforms

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         img_metadata = coco.loadImgs(img_id)[0]
#         path = img_metadata['file_name']
#         image = np.array(Image.open(os.path.join(self.img_dir, path)).convert('RGB'))

#         # Load annotations
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)

#         # Generate masks & bounding boxes
#         masks = []
#         boxes = []
#         labels = []
        
#         for ann in anns:
#             mask = coco.annToMask(ann)  # Convert segmentation to binary mask
#             if mask.sum() == 0:  # Skip empty masks
#                 continue

#             masks.append(mask)
#             # COCO classes are 1-indexed, subtract 1 for 0-indexing
#             labels.append(ann["category_id"]-1)  

#             # Convert bbox format to [x_min, y_min, x_max, y_max]
#             x, y, w, h = ann["bbox"]
#             boxes.append([x, y, x + w, y + h])

#         # Skip images without valid annotations
#         if len(masks) == 0:
#             return self.__getitem__((index + 1) % len(self.ids))

#         # Apply transformations
#         if self.transforms:
#             # Define appropriate transforms based on split
#             if self.split == 'train':
#                 transforms = A.Compose([
#                     A.Resize(height=512, width=512),
#                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                     # A.HorizontalFlip(p=0.5),
#                 ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
#             else:
#                 transforms = A.Compose([
#                     A.Resize(height=512, width=512),
#                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
            
#             # Apply transforms
#             transformed = transforms(
#                 image=image,
#                 masks=masks,
#                 bboxes=boxes,
#                 labels=labels
#             )
            
#             image = transformed['image']
#             masks = transformed['masks']
#             boxes = transformed['bboxes']
#             labels = transformed['labels']
        
#         # Convert everything to PyTorch tensors
#         image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
#         masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)  # Binary masks should be uint8
#         boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
#         labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        
#         # Create target dictionary
#         target = {
#             "boxes": boxes,
#             "labels": labels,
#             "masks": masks,  
#             "image_id": torch.tensor([img_id]),
#             "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#         }

#         return image, target

#     def __len__(self):
#         return len(self.ids)