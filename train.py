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


def start_training(exp_name = 'MaskRcnn', device='mps', data_path='/Users/dragos/Licenta/CardDD_COCO', checkpoint_path='/Users/dragos/Licenta/Results'):

  device = torch.device(device)
  num_classes = 6 
  train_dataset = COCOSegmentation(base_dir=data_path, split='train', my_transforms=True)
  subset = torch.utils.data.Subset(train_dataset, indices=list(range(20)))
  val_dataset = COCOSegmentation(base_dir=data_path, split='val', my_transforms=True)

  # define training and validation data loaders
  train_dataloader = torch.utils.data.DataLoader(
      subset,
      batch_size=4,
      shuffle=True,
      collate_fn=utils.collate_fn
  )

  val_dataloader = torch.utils.data.DataLoader(
      subset,
      batch_size=1,
      shuffle=False,
      collate_fn=utils.collate_fn
  )

  model = utils.get_model_instance_segmentation(num_classes)
  model.to(device)
  epochs=15
  # lr = 5e-4
  # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)  # start low
  # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
  #     optimizer,
  #     max_lr=5e-4,  # your peak
  #     steps_per_epoch=len(train_dataloader),
  #     epochs=epochs,
  #     pct_start=0.1,  # warmup
  #     div_factor=100,  # initial lr = max_lr/div_factor
  #     final_div_factor=100  # end small
  # )
  # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
  #                                                   max_lr=lr, 
  #                                                   total_steps=epochs*len(train_dataloader))

  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(
      params,
      lr=0.005,
      momentum=0.9,
      weight_decay=0.0005
  )

  # and a learning rate scheduler
  lr_scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer,
      step_size=3,
      gamma=0.1
  )

  utils.train_loop(model=model, 
            train_dataloader=train_dataloader,
            valid_dataloader=val_dataloader,
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            device=torch.device(device), 
            epochs=epochs, 
            checkpoint_path=checkpoint_path + '/' + exp_name,
            use_scaler=True)
