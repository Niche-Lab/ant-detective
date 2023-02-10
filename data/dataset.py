# native imports
import os
import pandas as pd
from PIL import Image

# pytorch imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# local import
from sampler import ImbalancedDatasetSampler


class PepSuc_Dataset(Dataset):
    """
    File structure:
    data/
        train/
            1.jpg
            2.jpg
            ...
            annotation.txt
        test/
            1.jpg
            2.jpg
            ...
            annotation.txt

    annotation.txt:
    # filename label
    1.jpg 3
    2.jpg 5
    3.jpg 0
    ...
    """

    def __init__(self, dir_img, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(dir_img, "annotation.txt"))
        self.dir_img = dir_img
        self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.ColorJitter(hue=0.3, brightness=0.05),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # get image matrix
        img_path = os.path.join(self.dir_img, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        # get label: sucrose and peptone
        label = self.img_labels.loc[idx, ["sucrose", "peptone"]].values
        label = label.astype("float").reshape(-1)
        # data transformation
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_loader(name_data: str, batch_size: int = 32) -> dict:
    # get root directory
    ROOT = os.path.dirname(os.path.abspath(__file__))
    # create dataloader
    loader = dict({})
    for setname in ["train", "test", "val"]:
        is_train = setname == "train"  # only shuffle train set

        # init dataset
        if name_data == "peptone_sucrose":
            dataset = PepSuc_Dataset(os.path.join(ROOT, "peptone_sucrose", setname))
        elif name_data == "peptone_sucrose_demo":
            dataset = PepSuc_Dataset(
                os.path.join(ROOT, "peptone_sucrose", "demo", setname)
            )

        # init dataloader
        loader[setname] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4,
            batch_sampler=ImbalancedDatasetSampler(dataset) if is_train else None,
        )
    # return
    return loader
