import os
import torch

# from skimage import io, transform
import numpy as np

# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandAugment
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps
from typing import Tuple, Dict, List, Optional

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import json


def get_classes():
    with open(
        "/cifs/data/tserre_lrs/projects/prj_tpu_timm/timm_tpu/locnet/imagenet_sub_category.json",
        "r",
    ) as file:
        data = json.load(file)
    return list(data.values())


def get_human_categories():
    with open(
        "/cifs/data/tserre_lrs/projects/prj_model_vs_human/LOCNet/data/imagenet_sub_category.json",
        "r",
    ) as file:
        data = json.load(file)
    return data


def get_num_labels():
    with open(
        "/cifs/data/tserre_lrs/projects/prj_tpu_timm/timm_tpu/locnet/IN_category_to_Human_category_idx.json",
        "r",
    ) as file:
        data = json.load(file)
    return data


def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        AlbumentationsRandAugment(num_ops=2, magnitude=9),
        ToTensorV2(),
    ])


class LOCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self, root_dir, depth_path, transforms=None
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.depth_path = depth_path
        self.transform = transforms
        self.class_to_num = get_num_labels()
        self.class_names = sorted(get_classes())

        self.images = []
        self.depth_maps = []
        self.labels = []

        for cls_name in self.class_names:
            class_dir = os.path.join(root_dir, cls_name)
            depth_map_dir = os.path.join(depth_path, cls_name)
            image_files = os.listdir(class_dir)

            for img_file in image_files:
                self.images.append(os.path.join(class_dir, img_file))
                depth_file = img_file.split(".")[0] + "_depth.JPEG"
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

        image = Image.open(img_path).convert("RGB")
        depth = ImageOps.grayscale(Image.open(depth_path))
        depth = np.array(depth).astype(np.float32)
        depth /= 255.0

        if self.transform:
            image, depth = self.transform(image, depth)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return image, depth, label


class GeiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, filter, image_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.class_to_num = get_num_labels()
        self.class_names = sorted(get_classes())
        self.human_categories = get_human_categories()

        self.images = []
        self.labels = []

        folder_style1 = [
            "colour",
            "contrast",
            "eidolonI",
            "eidolonII",
            "eidolonIII",
            "false-colour",
            "high-pass",
            "low-pass",
            "phase-scrambling",
            "power-equalisation",
            "rotation",
            "sketch",
            "stylized",
            "uniform-noise",
        ]

        if filter in folder_style1:
            root_dir = os.path.join(root_dir, filter, "dnn", "session-1")
            image_paths = os.listdir(root_dir)
            for img in image_paths:
                cls = img.split("_")[4]
                class_category = self.human_categories[cls]
                self.labels.append(self.class_to_num[class_category])
                self.images.append(os.path.join(root_dir, img))

        else:
            root_dir = os.path.join(root_dir, filter)
            class_names = os.listdir(root_dir)
            for cls in class_names:
                class_dir = os.path.join(root_dir, cls)
                images_paths = os.listdir(class_dir)
                for img in images_paths:
                    self.images.append(os.path.join(class_dir, img))
                    self.labels.append(self.class_to_num[self.human_categories[cls]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        return image, label


class AlbumentationsRandAugment(ImageOnlyTransform):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, p=1.0):
        super().__init__(always_apply=False, p=p)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.magnitude_scale = self.magnitude / self.num_magnitude_bins

        # Define operations with their magnitudes scaled appropriately
        self.operations = [
            ("Identity", A.NoOp()),
            ("ShearX", A.Affine(shear={'x': (-30 * self.magnitude_scale, 30 * self.magnitude_scale)}, p=1.0)),
            ("ShearY", A.Affine(shear={'y': (-30 * self.magnitude_scale, 30 * self.magnitude_scale)}, p=1.0)),
            ("TranslateX",
             A.Affine(translate_percent={"x": (-0.1 * self.magnitude_scale, 0.1 * self.magnitude_scale)}, p=1.0)),
            ("TranslateY",
             A.Affine(translate_percent={"y": (-0.1 * self.magnitude_scale, 0.1 * self.magnitude_scale)}, p=1.0)),
            ("Rotate", A.Rotate(limit=(-30 * self.magnitude_scale, 30 * self.magnitude_scale), p=1.0)),
        ]

    def apply(self, img, depth, **params):
        ops = np.random.choice(self.operations, size=self.num_ops, replace=False)
        for op_name, op in ops:
            img = op.apply(img, **params)
            depth = op.apply(depth, **params)
        return img, depth

    def get_transform_init_args_names(self):
        return ("num_ops", "magnitude", "num_magnitude_bins")
