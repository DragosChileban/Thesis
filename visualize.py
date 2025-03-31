from dataset import COCOSegmentation
import utils

train_trans, val_trans = utils.get_my_transforms()
dataset_test = COCOSegmentation(args=None, split='train', my_transforms=train_trans)
classes = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
sample = dataset_test[130]
print(type(sample))