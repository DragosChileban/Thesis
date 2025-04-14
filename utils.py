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
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

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

    sizes = (
    (16,),  # P2
    (32,),  # P3
    (64,),  # P4
    (128,), # P5
    (256,)  # P6
    )

    aspect_ratios = (
        (0.5, 1.0, 2.0),  # P2
        (0.5, 1.0, 2.0),  # P3
        (0.5, 1.0, 2.0),  # P4
        (0.5, 1.0, 2.0),  # P5
        (0.5, 1.0, 2.0)   # P6
    )

    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    model.rpn.anchor_generator = anchor_generator

    model.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=28,   # try 28 or even 56
        sampling_ratio=2
    )

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)


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

def run_train_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, writer, plot_step=5):
    model.train()
    
    epoch_loss = 0
    
    for batch_id, (images, targets) in enumerate(dataloader):
        images = torch.stack(images).to(device)
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        # with autocast(torch.device(device)):
        img = images[0]
        losses = model(images.to(device), move_data_to_device(targets, device))
        # print("Loss dict", losses)
        # loss = sum([loss for loss in losses.values()]) 
        loss = (
            1.0 * losses['loss_classifier'] +
            1.0 * losses['loss_box_reg'] +
            10.0 * losses['loss_mask'] +
            1.0 * losses['loss_objectness'] +
            1.0 * losses['loss_rpn_box_reg']
        )
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                lr_scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        optimizer.zero_grad()

        loss_item = loss.item()
        epoch_loss += loss_item
        
        if batch_id % plot_step == 0:
          print(f"Batch {batch_id}/{len(dataloader)}")
          print('Loss:  ', loss_item, '   Avg loss:  ', epoch_loss/(batch_id+1))
          print(losses)
          writer.add_scalar('Train/Loss', loss_item, epoch_id*len(dataloader) + batch_id)
          writer.add_scalar('Train/Avg_Loss', epoch_loss/(batch_id+1), epoch_id*len(dataloader) + batch_id)
          if batch_id % plot_step == 0:
            model.eval()
            with torch.no_grad():
              outputs = model(images.to(device))
            gt_plot = plot_prediction(images, targets)
            pred_plot = plot_prediction(images, outputs)
            writer.add_image('Train_samples/Ground_truth', gt_plot, epoch_id*len(dataloader) + batch_id)
            writer.add_image('Train_samples/Prediction', pred_plot, epoch_id*len(dataloader) + batch_id)
            model.train()

        # If loss is NaN or infinity, stop training
        stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
        assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    return epoch_loss / (batch_id + 1)


def run_val_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, writer, plot_step=1):
    model.eval()
    
    epoch_loss = 0
    
    for batch_id, (images, targets) in enumerate(dataloader):
        images = torch.stack(images).to(device)
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        # with autocast(torch.device(device).type):
        with torch.no_grad():
          losses = model(images.to(device), move_data_to_device(targets, device))
          loss = sum([loss for loss in losses.values()]) 

        loss_item = loss.item()
        epoch_loss += loss_item
        
        if batch_id % plot_step == 0:
          print(f"Batch {batch_id}/{len(dataloader)}")
          print('Loss:  ', loss_item, '   Avg loss:  ', epoch_loss/(batch_id+1))
          writer.add_scalar('Valid/Loss', loss_item, epoch_id*len(dataloader) + batch_id)
          writer.add_scalar('Valid/Avg_Loss', epoch_loss/(batch_id+1), epoch_id*len(dataloader) + batch_id)
          if batch_id % plot_step == 0:
            with torch.no_grad():
                outputs = model(images.to(device))
            gt_plot = plot_prediction(images, targets)
            pred_plot = plot_prediction(images, outputs)
            writer.add_image('Valid_samples/Ground_truth', gt_plot, epoch_id*len(dataloader) + batch_id)
            writer.add_image('Valid_samples/Prediction', pred_plot, epoch_id*len(dataloader) + batch_id)

        # If loss is NaN or infinity, stop training
        stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
        assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    return epoch_loss / (batch_id + 1)


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
  
        # with torch.no_grad():
        #     valid_loss = run_val_epoch(model, valid_dataloader, None, None, device, scaler, epoch, writer)

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