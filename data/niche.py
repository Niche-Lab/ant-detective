# pytorch imports
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# native imports
import os
import pandas as pd
import numpy as np
from abc import abstractmethod


class Niche_Loader(L.LightningDataModule):
    def __init__(self, dataname, shuffle=None, batch_size=32):
        super().__init__()
        self.param_dataset = dict(
            dataname=dataname,
            shuffle=shuffle,
        )
        self.batch_size = batch_size
        self.mu = None
        self.std = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # stage is
        # - None: fit, called on every GPU
        # - "fit": TrainerFn.FITTING: 'fit'
        # - "validate": TrainerFn.VALIDATING: 'validate'
        # - "test": TrainerFn.TESTING: 'test'
        # - "predict": TrainerFn.PREDICTING: 'predict'

        if stage == "fit" or stage is None:
            self.train = self._get_dataset(**self.param_dataset, setname="train")
            self.val = self._get_dataset(**self.param_dataset, setname="val")
            mu, std = self.compute_label_stats(self.train)
            self.train.set_target_transform(mu, std)
            self.val.set_target_transform(mu, std)

        elif stage == "test" or stage == "predict":
            self.train = self._get_dataset(**self.param_dataset, setname="train")
            self.test = self._get_dataset(**self.param_dataset, setname="test")
            mu, std = self.compute_label_stats(self.train)
            self.test.set_target_transform(mu, std)

        self.mu = mu
        self.std = std

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            # sampler=ImbalancedDatasetSampler(dataset) if is_train else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def compute_label_stats(self, dataset):
        labels = np.array(dataset.get_labels())
        mean = labels.mean()
        std = labels.std()
        return mean, std

    @abstractmethod
    def _get_dataset(self, **kwargs):
        pass


class Niche_Dataset(Dataset):
    def __init__(self, dir_dataset, shuffle, setname):
        """
        args:
            dir_dataset: str, path to the dataset root folder
            setname: str, "train", "val", or "test"
            shuffle: str, "train/val", "train/test", "random" or None
        """
        self.dir_dataset = dir_dataset
        self.annotations = self._load_annotations(dir_dataset)
        self.annotations = self._split_dataset(self.annotations, shuffle)
        self.annotations = self._set_split(self.annotations, setname)
        self.transform = None
        self.target_transform = None

    def __getitem__(self, idx):
        """
        return x, y
        """
        x = self._get_item_x(idx)
        y = self._get_item_y(idx)
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def set_target_transform(self, mu, std):
        self.target_transform = lambda x: (x - mu) / std

    @abstractmethod
    def _load_annotations(self, dir_dataset):
        """
        load annotations.csv
        ---
        args:
            dir_dataset: str, path to the dataset root folder
        return:
            pd.DataFrame
        """
        pass

    def _split_dataset(self, annotations, shuffle):
        """
        split the dataset into train/val/test
        ---
        args:
            annotations: pd.DataFrame
            shuffle: str, "train/val", "train/test", "random" or "default"
        """
        if shuffle == "train/val":
            # shuffle train/val (64/16)
            annotations = self._shuffle_train_val(annotations)
        elif shuffle == "train/test":
            # shuffle train/test (80/20) then train/val (64/16)
            annotations = self._shuffle_train_test(annotations)
            annotations = self._shuffle_train_val(annotations)
        elif shuffle == "random":
            # shuffle randomly (not by any experimental factor)
            annotations = self._shuffle_random(annotations)
        elif shuffle == "default":
            # default splits
            annotations = self._default_splits(annotations)
        # return the annotations
        return annotations

    @abstractmethod
    def _set_split(self, annotations, setname):
        """
        set the split
        ---
        args:
            annotations: pd.DataFrame
            setname: str, "train", "val", or "test"
        return:
            pd.DataFrame
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        return the length of the dataset
        """
        pass

    @abstractmethod
    def _get_item_x(self, idx):
        pass

    @abstractmethod
    def _get_item_y(self, idx):
        pass

    @abstractmethod
    def _shuffle_train_val(self, annotations):
        """
        shuffle train/val (64/16)
        ---
        args:
            annotations: pd.DataFrame
        return:
            pd.DataFrame
        """
        pass

    @abstractmethod
    def _shuffle_train_test(self, annotations):
        """
        shuffle train/test (80/20)
        ---
        args:
            annotations: pd.DataFrame
        return:
            pd.DataFrame
        """
        pass

    @abstractmethod
    def _shuffle_random(self, annotations):
        """
        shuffle randomly (not by any experimental factor)
        ---
        args:
            annotations: pd.DataFrame
        return:
            pd.DataFrame
        """
        pass

    @abstractmethod
    def _default_splits(self, annotations):
        """
        default splits
        ---
        args:
            annotations: pd.DataFrame
        return:
            pd.DataFrame
        """
        pass

    @abstractmethod
    def get_labels(self):
        """
        return the entire labels for standardization
        """
        pass
