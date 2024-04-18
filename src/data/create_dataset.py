import os
import torch

# from skimage import io, transform
import numpy as np

# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchvision import transforms as T
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


def get_shared_transform():
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        RandAffineAugment(),
        T.ToTensor(),
    ])
    return transforms


def get_img_transform():
    return RandColorAugment()


class LOCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self, root_dir, depth_path, shared_transforms=None, img_transform=None
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
        self.shared_transform = DualTransform(shared_transforms) if shared_transforms else None
        self.img_transform = img_transform if img_transform else None
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

        if self.shared_transform:
            image, depth = self.shared_transform(image, depth)
            image = self.img_transform(image) if self.img_transform else image
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


class DualTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, depth):
        seed = torch.random.initial_seed()
        torch.manual_seed(seed)
        if self.transform is not None:
            image = self.transform(image)
        torch.manual_seed(seed)
        if self.transform is not None:
            depth = self.transform(depth)
        return image, depth


class RandAffineAugment(RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        }


class RandColorAugment(RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
