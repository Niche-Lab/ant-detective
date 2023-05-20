# native imports
import os
import numpy as np
import pandas as pd
from PIL import Image

# pytorch imports
from torchvision import transforms

# local import
from niche import Niche_Loader, Niche_Dataset

# CONSTANTS
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(ROOT, "data", "raw")


class Ant_Loader(Niche_Loader):
    def __init__(self, dataname, traits, shuffle=None, batch_size=32):
        super().__init__(dataname, shuffle, batch_size)
        self.traits = traits

    def _get_dataset(self, dataname, shuffle, setname):
        dir_dataset = os.path.join(DIR_DATA, dataname)  # peptone_sucrose
        dataset = Ant_Dataset(
            dir_dataset=dir_dataset,
            shuffle=shuffle,
            setname=setname,
            traits=self.traits,
        )
        return dataset


class Ant_Dataset(Niche_Dataset):
    def __init__(self, dir_dataset, shuffle, setname, traits):
        """
        args:
            traits: list of str. Example: ["peptone", "sucrose"]
        """
        super().__init__(dir_dataset, shuffle, setname)
        self.traits = traits
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.ColorJitter(hue=0.3, brightness=0.05),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_annotations(self, dir_dataset):
        path_annotations = os.path.join(dir_dataset, "annotations.csv")
        annotations = pd.read_csv(path_annotations)
        return annotations

    def _set_split(self, annotations, setname):
        return annotations.query("split == %s" % setname)

    def __len__(self):
        return len(self.annotations)

    def _get_item_x(self, idx):
        img_path = os.path.join(
            self.dir_dataset, self.annotations.iloc[idx, "filename"]
        )
        image = Image.open(img_path)
        return image

    def _get_item_y(self, idx):
        return self.annotations.loc[idx, self.traits].values

    def _shuffle_train_val(self, annotations):
        # set all val and train (except test) to "train"
        annotations.loc[annotations.split != "test", "split"] = "train"
        # list trials in train (got 6 trials)
        trials = annotations.query("split == 'train'").trial.unique()
        # randomly select 1 as val
        val_trial = np.random.choice(trials, size=1, replace=False)[0]
        # set val_trial to "val"
        annotations.loc[annotations.trial == val_trial, "split"] = "val"
        # return
        return annotations

    def _shuffle_train_test(self, annotations):
        # set all  split to "train"
        annotations.loc[:, "split"] = "train"
        # get all trials
        trials = annotations.trial.unique()
        # select 3 trials as test
        test_trials = np.random.choice(trials, size=3, replace=False)
        # set test_trials to "test"
        annotations.loc[annotations.trial.isin(test_trials), "split"] = "test"
        # return
        return annotations

    def _shuffle_random(self, annotations):
        n = len(annotations)
        n_train = int(n * 0.64)
        n_val = int(n * 0.16)
        n_test = n - n_train - n_val
        # set all split to "train"
        annotations.loc[:, "split"] = "train"
        # randomly select n_val as val
        val_idx = np.random.choice(n, size=n_val, replace=False)
        annotations.loc[val_idx, "split"] = "val"
        # randomly select n_test as test
        test_idx = np.random.choice(
            np.setdiff1d(np.arange(n), val_idx), size=n_test, replace=False
        )
        annotations.loc[test_idx, "split"] = "test"
        # return
        return annotations

    def _default_splits(self, annotations):
        # 1-5: train
        annotations.loc[annotations.trial.isin(range(1, 6)), "split"] = "train"
        # 6: val
        annotations.loc[annotations.trial == 6, "split"] = "val"
        # 7-9: test
        annotations.loc[annotations.trial.isin(range(7, 10)), "split"] = "test"
        # return
        return annotations

    def get_labels(self):
        return self.annotations.loc[:, self.traits].values
