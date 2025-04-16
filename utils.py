import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.v2  as transforms
import torch
import math
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import sys
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.rpn import AnchorGenerator


def move_data_to_device(data, # Data to move to the device.
                        device:torch.device # The PyTorch device to move the data to.
                       ): # Moved data with the same structure as the input but residing on the specified device.
    """
    Recursively move data to the specified device.

    This function takes a data structure (could be a tensor, list, tuple, or dictionary)
    and moves all tensors within the structure to the given PyTorch device.
    """
    
    # If the data is a tuple, iterate through its elements and move each to the device.
    if isinstance(data, tuple):
        return tuple(move_data_to_device(d, device) for d in data)
    
    # If the data is a list, iterate through its elements and move each to the device.
    if isinstance(data, list):
        return list(move_data_to_device(d, device) for d in data)
    
    # If the data is a dictionary, iterate through its key-value pairs and move each value to the device.
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    
    # If the data is a tensor, directly move it to the device.
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    
    # If the data type is not a tensor, list, tuple, or dictionary, it remains unchanged.
    else:
        return data

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    print("Hidden layers ", hidden_layer)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    model.roi_heads.mask_loss_weight = 5.0  # Increase mask loss importance

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)


def plot_ultra_prediction(image, boxes, masks):
  if 'scores' in outputs:
    score_mask = outputs['scores'] > 0.5
    outputs = {
        'masks': outputs['masks'][score_mask],
        'boxes': outputs['boxes'][score_mask],
        'labels': outputs['labels'][score_mask]
    }
  classes = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  image_tensor = denormalize(images[0], mean, std)

  masks = outputs['masks'] > 0  # Convert masks to boolean
  image_with_masks = draw_segmentation_masks(image_tensor, masks.squeeze(1), colors="pink", alpha=0.7)

  boxes = outputs['boxes']
  labels = [classes[int(label)] for label in outputs['labels']]
  image_with_boxes = draw_bounding_boxes(image_with_masks, boxes, colors="red", labels=labels, width=2)

  return image_with_boxes

def plot_prediction(images, outputs):
  outputs = outputs[0]
  if 'scores' in outputs:
    score_mask = outputs['scores'] > 0.5
    outputs = {
        'masks': outputs['masks'][score_mask],
        'boxes': outputs['boxes'][score_mask],
        'labels': outputs['labels'][score_mask]
    }
  classes = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  image_tensor = denormalize(images[0], mean, std)

  masks = outputs['masks'] > 0  # Convert masks to boolean
  image_with_masks = draw_segmentation_masks(image_tensor, masks.squeeze(1), colors="pink", alpha=0.7)

  boxes = outputs['boxes']
  labels = [classes[int(label)] for label in outputs['labels']]
  image_with_boxes = draw_bounding_boxes(image_with_masks, boxes, colors="red", labels=labels, width=2)

  return image_with_boxes

def run_train_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, writer, plot_step=1):
    model.train()
    
    epoch_loss = 0

    lr_scheduler = None
    if epoch_id == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    for batch_id, (images, targets) in enumerate(dataloader):
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
      # Forward pass with Automatic Mixed Precision (AMP) context manager
      with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # losses = (
        #     1.0 * loss_dict['loss_classifier'] +
        #     1.0 * loss_dict['loss_box_reg'] +
        #     3.0 * loss_dict['loss_mask'] +
        #     1.0 * loss_dict['loss_objectness'] +
        #     1.0 * loss_dict['loss_rpn_box_reg']
        # )

      optimizer.zero_grad()
      if scaler is not None:
          scaler.scale(losses).backward()
          scaler.step(optimizer)
          scaler.update()
      else:
          losses.backward()
          optimizer.step()

      if lr_scheduler is not None:
          lr_scheduler.step()

      loss_item = losses.item()
      epoch_loss += loss_item
      
      if batch_id % plot_step == 0:
        print(f"Batch {batch_id}/{len(dataloader)}")
        print('Loss:  ', loss_item, '   Avg loss:  ', epoch_loss/(batch_id+1))
        print(loss_dict)
        writer.add_scalar('Train/Loss', loss_item, epoch_id*len(dataloader) + batch_id)
        writer.add_scalar('Train/Avg_Loss', epoch_loss/(batch_id+1), epoch_id*len(dataloader) + batch_id)
        # if batch_id % plot_step == 0:
        #   model.eval()
        #   with torch.no_grad():
        #     outputs = model(images)
        #   gt_plot = plot_prediction(images, targets)
        #   pred_plot = plot_prediction(images, outputs)
        #   writer.add_image('Train_samples/Ground_truth', gt_plot, epoch_id*len(dataloader) + batch_id)
        #   writer.add_image('Train_samples/Prediction', pred_plot, epoch_id*len(dataloader) + batch_id)
        #   model.train()

        # If loss is NaN or infinity, stop training
        stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
        assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    return epoch_loss / (batch_id + 1)


def run_val_epoch(model, dataloader, device, epoch_id, writer, plot_step=9):
    model.eval()
    
    epoch_loss = 0
    
    for batch_id, (images, targets) in enumerate(dataloader):
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
      
      outputs = model(images)
        
      if batch_id % plot_step == 0:
          gt_plot = plot_prediction(images, targets)
          pred_plot = plot_prediction(images, outputs)
          writer.add_image('Valid_samples/Ground_truth', gt_plot, epoch_id*len(dataloader) + batch_id)
          writer.add_image('Valid_samples/Prediction', pred_plot, epoch_id*len(dataloader) + batch_id)

    return epoch_loss


def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               use_scaler=False):
    
    print("Starting training loop...")
    writer = SummaryWriter(log_dir=checkpoint_path)

    scaler = torch.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf') 
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = run_train_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, writer)
  
        with torch.no_grad():
            valid_loss = run_val_epoch(model, valid_dataloader, device, epoch, writer)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(model.state_dict(), checkpoint_path + '/epoch' + str(epoch) + '.pth')

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()

def get_a_transforms():
    train_transforms = A.Compose([
        A.AtLeastOneBBoxRandomCrop(height=512, width=512, p=1.0),  
        A.HorizontalFlip(p=1.0),                     
        A.Normalize(mean=[0.485, 0.456, 0.406],      
                    std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))  

    val_transforms = A.Compose([
        A.Resize(height=512, width=512),             
        A.Normalize(mean=[0.485, 0.456, 0.406],     
                    std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))

    

    return train_transforms, val_transforms

# def get_transforms():
#     train_transforms = transforms.Compose(
#     [
#         transforms.ToImage(),
#         # transforms.RandomPhotometricDistort(p=1),
#         transforms.RandomIoUCrop(min_aspect_ratio=1.0, max_aspect_ratio=1.0),
#         transforms.Resize((256, 256)),
#         transforms.RandomHorizontalFlip(p=1),
#         transforms.SanitizeBoundingBoxes(),
#         transforms.ToDtype(torch.float32, scale=True),
#     ])

#     val_transforms = transforms.Compose(
#     [
#         transforms.ToImage(),
#         transforms.Resize((256, 256)),
#         # transforms.SanitizeBoundingBoxes(),
#         transforms.ToDtype(torch.float32, scale=True),
#     ])
    
#     return train_transforms, val_transforms