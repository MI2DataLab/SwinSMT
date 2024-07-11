import glob
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandZoomd,
    ToTensord,
)


class TotalSegmentatorDataset(Dataset):
    """
    TotalSegmentatorDataset class.
    """

    def __init__(self,
                 data_path: str = None,
                 mode: str = None,
                 patch_size: list = (96, 96, 96)
                 ) -> None:
        """

        Args:
            data_path:
            mode:
        """
        self.data_path = data_path
        self.patch_size = patch_size
        assert mode in ["train", "valid", "test", None]
        self.mode = mode

        self.images = sorted(glob.glob(os.path.join(self.data_path, "*/*ct*.nii.gz"), recursive=True))
        self.labels = sorted(glob.glob(os.path.join(self.data_path, "*/*segmentations*.nii.gz"), recursive=True))
        assert len(self.images) == len(self.labels)

        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=1024,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"],
                                source_key="image",
                                k_divisible=self.patch_size),
                RandZoomd(keys=["image", "label"],
                          min_zoom=1.3, max_zoom=1.5,
                          mode=["area", "nearest"],
                          prob=0.3),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.1,
                    max_k=3,
                ),
                RandShiftIntensityd(keys=["image"],
                                    offsets=0.10,
                                    prob=0.20),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.valid_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=1024,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                CropForegroundd(keys=["image", "label"],
                                source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.test_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                ToTensord(keys=["image", "label"])
            ]
        )

    def __getitem__(self, x):
        """

        Args:
            x:

        Returns:

        """
        data_dict = {"image": self.images[x], "label": self.labels[x]}
        if self.mode == "train":
            data_dict = self.train_transform(data_dict)
        elif self.mode == "valid":
            data_dict = self.valid_transform(data_dict)
        elif self.mode == "test":
            data_dict = self.test_transform(data_dict)
        else:
            NotImplementedError("Please provide proper transformation!")

        return data_dict

    def __len__(self):
        return len(self.images)
