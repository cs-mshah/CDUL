import os
from typing import Dict, List
from torch import Tensor
import torch
import numpy as np
from torchvision.datasets import VOCDetection
import lovely_tensors as lt
lt.monkey_patch()


def get_categories(labels_dir: str) -> List[str]:
    """
    Get the object categories for Pascal VOC 2012

    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a sorted list to maintain consistency
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories: List[str] = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])
        categories.sort()
        return categories


class VOCLabelTransform:
    def __init__(self, object_categories: str | None = None, exclude_difficult: bool = False, transform_type: str = 'one_hot', saved_dir: str = None):
        """VOC Label Transform
        Args:
            object_categories (list): List of object categories.
            exclude_difficult (bool): If True, exclude objects labeled as difficult (for one-hot).
            transform_type (VOCLabelTransformType): Type of label transform to apply.
            saved_dir (str): Directory to fetch the saved label tensors.
        """
        self.object_categories = object_categories
        self.exclude_difficult = exclude_difficult
        self.transform_type = transform_type
        self.saved_dir = saved_dir

    def __call__(self, target: Dict):
        """
        return label transform depending on transform_type

        Args:
            target: xml tree file
        Returns:
            target with transform type applied
        """
        if self.transform_type == 'one_hot':
            return self.one_hot(target)
        elif self.transform_type == 'filename':
            return self.filename_label(target)
        elif self.transform_type == 'global':
            return self.global_label(target)
        elif self.transform_type == 'aggregate':
            return self.aggregated_label(target)
        elif self.transform_type == 'final':
            return self.final_label(target)
        

    def one_hot(self, target) -> Tensor:
        """
        return one hot encoded labels
        """
        ls = target['annotation']['object']

        j = []
        if type(ls) == dict:
            if int(ls['difficult']) == 0:
                j.append(self.object_categories.index(ls['name']))
            elif int(ls['difficult']) == 1 and self.exclude_difficult == False:
                j.append(self.object_categories.index(ls['name']))
        elif type(ls) == list:
            for i in range(len(ls)):
                if int(ls[i]['difficult']) == 0:
                    j.append(self.object_categories.index(ls[i]['name']))
                elif int(ls[i]['difficult']) == 1 and self.exclude_difficult == False:
                    j.append(self.object_categories.index(ls[i]['name']))
        else:
            raise TypeError

        k = np.zeros(len(self.object_categories), dtype=int)
        k[j] = 1 # object present

        return torch.from_numpy(k)
    
    def filename_label(self, target) -> str:
        """
        return one hot encoded label
        """
        return target['annotation']['filename']

    
    def global_label(self, target) -> Tensor:
        """returns soft global saved vector
        """
        tensor_location = os.path.join(self.saved_dir, 'global', self.filename_label(target).split('.')[0] + '.pt')
        return torch.load(tensor_location, map_location=torch.device('cpu'))
    
    def aggregated_label(self, target) -> Tensor:
        """returns soft aggregation saved vector
        """
        tensor_location = os.path.join(self.saved_dir, 'aggregate', self.filename_label(target).split('.')[0] + '.pt')
        return torch.load(tensor_location, map_location=torch.device('cpu'))
    
    def final_label(self, target) -> Tensor:
        """returns initial pseudo label
        """
        return 0.5 * (self.global_label(target) + self.aggregated_label(target))


def get_label_txt(label: torch.Tensor, object_categories: List) -> List[str]:
    """
    Get Label text from one hot encoded Tensor

    Args:
        label: 0/1 encoded label
    Returns:
        List of  categories string
    """
    k = label.numpy()
    labeltext=[]
    for i in range(len(k)):
      if k[i] >= 1:
        labeltext.append(object_categories[i])

    return labeltext


def download_dataset():
    """Download Pascal VOC 2012 dataset
    """
    _ = VOCDetection(os.environ['DATASETS_ROOT'], year='2012', image_set='val', download=True)


if __name__ == '__main__':
    download_dataset()