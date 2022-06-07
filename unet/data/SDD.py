"""Semantic Drone Dataset (SDD)"""
from enum import Enum
import os
import cv2
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as Alb
from torch.utils.data import Dataset


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


"""Semantic Drone Dataset (SDD)"""
import logging
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as Alb


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


"""Semantic Drone Dataset (SDD)"""
import logging
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as Alb
from albumentations.pytorch import ToTensorV2


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


class SegmentationPixelValues(Enum):
    person = 15
    car = 17
    dog = 16
    roof = 9

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


"""Semantic Drone Dataset (SDD)"""
from enum import Enum
import os
import cv2
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as Alb
from torch.utils.data import Dataset


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


"""Semantic Drone Dataset (SDD)"""
import logging
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as Alb


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


"""Semantic Drone Dataset (SDD)"""
import logging
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as Alb
from albumentations.pytorch import ToTensorV2


def kaggle_auth(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(destination):
    """download semantic drone dataset into destination directory"""
    # import of kaggle makes authentication call straight away
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="bulentsiyah/semantic-drone-dataset", path=destination
    )


class SegmentationPixelValues(Enum):
    person = 15
    car = 17
    dog = 16
    roof = 9

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


# experimental augmentation
BASIC_AUGMENTATION = Alb.Compose(
    [
        Alb.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
        Alb.HorizontalFlip(),
        Alb.VerticalFlip(),
        Alb.GridDistortion(p=0.2),
        Alb.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        Alb.GaussNoise(),
    ]
)


class SemanticDroneDataset(Dataset):
    """dataset module from https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset"""

    LABEL_FOLDER_NAME = "label_images_semantic"
    IMAGE_FOLDER_NAME = "original_images"

    def __init__(self, data_path: Path, transforms=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_samples = self.read_samples()
        self.df = self.create_dataframe_from_dataset()
        # run augmentation transforms on each image file read
        if transforms is not None:
            print("applying extra augmentations..")
            self.transform_pipeline = Alb.Compose([transforms, ToTensorV2()])

        else:
            print("convert to tensor")
            self.transform_pipeline = ToTensorV2()

    def create_dataframe_from_dataset(self):
        return pd.DataFrame(
            {"id": self.train_samples}, index=np.arange(0, len(self.train_samples))
        )

    def read_samples(self):
        train_samples = self.data_path / self.IMAGE_FOLDER_NAME
        logging.info(f"reading foleder: {train_samples}")
        train_samples = train_samples.glob("*.jpg")

        return list(train_samples)

    def get_corresponding_label(self, sample_path):
        name = Path(sample_path).stem
        return self.data_path / self.LABEL_FOLDER_NAME / "{}.png".format(name)

    def pixel_to_channel(self, seg: np.array):
        """one-hot each pixel value to seperate image black and white scale channel"""
        mask = np.zeros((len(SegmentationPixelValues), *seg.shape))
        for idx, segment_class in enumerate(SegmentationPixelValues.list()):
            pixels = np.where(seg == segment_class)
            mask[idx][pixels] = 1

        return mask

    @staticmethod
    def read_rgb(path):
        return cv2.imread(str(path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_gray(path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, index):
        train_sample_path = self.df.iloc[index]["id"]
        train_image = self.read_rgb(train_sample_path)
        logging.info(f"reading train sample: {train_sample_path}")

        label_segment_path = self.get_corresponding_label(train_sample_path)
        label_segment = self.read_gray(label_segment_path)
        label_segment = self.pixel_to_channel(label_segment)
        logging.info(f"reading label sample: {label_segment}")

        transformed = self.transform_pipeline(image=train_image, mask=label_segment)
        return transformed["image"], transformed["mask"]
