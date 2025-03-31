import torch
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image
import json
from tqdm import trange
import torchvision.transforms as transforms
import torchvision
from dataset import COCOSegmentation
import utils

device = torch.device('mps')

num_classes = 6 
train_transforms, val_transforms = utils.get_transforms()
train_dataset = COCOSegmentation(args=None, split='train', my_transforms=train_transforms)
val_dataset = COCOSegmentation(args=None, split='val', my_transforms=val_transforms)

# define training and validation data loaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=utils.collate_fn
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

model = utils.get_model_instance_segmentation(num_classes)
model.to(device)
epochs=40
lr = 5e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                   max_lr=lr, 
                                                   total_steps=epochs*len(train_dataloader))

# construct an optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=0.005,
#     momentum=0.9,
#     weight_decay=0.0005
# )

# # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1
# )

checkpoint_path = '/Users/dragos/Licenta/Results/maskrcnn.pth'
utils.train_loop(model=model, 
           train_dataloader=train_dataloader,
           valid_dataloader=val_dataloader,
           optimizer=optimizer, 
           lr_scheduler=lr_scheduler, 
           device=torch.device(device), 
           epochs=epochs, 
           checkpoint_path=checkpoint_path,
           use_scaler=True)