import os
import torch
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
# import albumentations as A
import json

def get_classes():
    with open('/cifs/data/tserre_lrs/projects/prj_tpu_timm/timm_tpu/locnet/imagenet_sub_category.json', 'r') as file:
        data = json.load(file)
    return list(data.values())

def get_num_labels():
    with open('/cifs/data/tserre_lrs/projects/prj_tpu_timm/timm_tpu/locnet/IN_category_to_Human_category_idx.json', 'r') as file:
        data = json.load(file)
    return data

def get_image_transform():

    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])


def get_depth_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        
    ])


class LOCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, depth_path, image_transform=None, depth_transform = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.depth_path = depth_path
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.class_to_num = get_num_labels()
        self.class_names = sorted(get_classes())

        self.images = []
        self.depth_maps = []
        self.labels = []

        for cls_name in self.class_names:
            class_dir = os.path.join(root_dir, 'n02504458' )
            depth_map_dir = os.path.join(depth_path, 'n02504458')
            image_files = os.listdir(class_dir)

            for img_file in image_files:
                self.images.append(os.path.join(class_dir, img_file))
                depth_file = img_file.split('.')[0] + '_depth.JPEG'
                self.depth_maps.append(os.path.join(depth_map_dir, depth_file))
                self.labels.append(self.class_to_num[cls_name])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]
        depth_path = self.depth_maps[idx]

        image = Image.open(img_path).convert('RGB')
        depth = ImageOps.grayscale(Image.open(depth_path))
        depth = np.array(depth).astype(np.float32)
        depth /= 255.0


        
        if self.image_transform:
            image = self.image_transform(image)
            depth = self.depth_transform(depth)

        return image, depth, label