import os
from typing import Dict, List
from torch import Tensor
import torch
import numpy as np
from torchvision.datasets import VOCDetection
from loguru import logger as log
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
    def __init__(self, object_categories: List | None = None, exclude_difficult: bool = False, transform_type: str = 'onehot', 
                 global_cache_dir: str = None, 
                 aggregate_cache_dir: str = None,
                 pseudo_cache_dir: str = None):
        """VOC Label Transform
        Args:
            object_categories (list): List of object categories.
            exclude_difficult (bool): If True, exclude objects labeled as difficult (for one-hot).
            transform_type (VOCLabelTransformType): Type of label transform to apply.
            global_cache_dir (str): Directory to fetch the global cached tensors.
            aggregate_cache_dir (str): Directory to fetch the aggregate cached tensors.
            pseudo_cache_dir (str): Directory to fetch the pseudo label vectors during training. Set to None for all other cases.
        """
        self.object_categories = object_categories
        self.exclude_difficult = exclude_difficult
        self.transform_type = transform_type
        self.global_cache_dir = global_cache_dir
        self.aggregate_cache_dir = aggregate_cache_dir
        self.pseudo_cache_dir = pseudo_cache_dir

    def __call__(self, target: Dict):
        """
        return label transform depending on transform_type

        Args:
            target: xml tree file
        Returns:
            target with transform type applied
        """
        
        if '_' in self.transform_type:
            transform_types = self.transform_type.split('_')
            return tuple(getattr(self, f"{t}_label")(target) for t in transform_types)
        
        label_function_name = f"{self.transform_type}_label"
        
        if hasattr(self, label_function_name):
            label_function = getattr(self, label_function_name)
            return label_function(target)

        raise ValueError(f"Unsupported transform_type: {self.transform_type}")
        

    def onehot_label(self, target) -> Tensor:
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
    
    def get_tensor_location(self, cache_dir: str, target) -> str:
        """returns the tensor location based on the target
        """
        return os.path.join(cache_dir, self.filename_label(target).split('.')[0] + '.pt')

    def global_label(self, target) -> Tensor:
        """returns soft global saved vector
        """
        cache = self.pseudo_label(target)
        if cache is not None:
            return cache
        tensor_location = self.get_tensor_location(self.global_cache_dir, target)
        return torch.load(tensor_location, map_location=torch.device('cpu'))

    def aggregated_label(self, target) -> Tensor:
        """returns soft aggregation saved vector
        """
        cache = self.pseudo_label(target)
        if cache is not None:
            return cache
        tensor_location = self.get_tensor_location(self.aggregate_cache_dir, target)
        return torch.load(tensor_location, map_location=torch.device('cpu'))
    
    def final_label(self, target) -> Tensor:
        """returns initial pseudo label
        """
        cache = self.pseudo_label(target)
        if cache is not None:
            return cache
        return 0.5 * (self.global_label(target) + self.aggregated_label(target))
    
    def pseudo_label(self, target) -> Tensor:
        """returns pseudo label stored during training
        """
        if self.pseudo_cache_dir is not None:
            tensor_location = self.get_tensor_location(self.pseudo_cache_dir, target)
            if os.path.exists(tensor_location):
                try:
                    return torch.load(tensor_location, map_location=torch.device('cpu'))
                except Exception as e:
                    log.error(f"Error loading pseudo label for {tensor_location}: {e}")


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
    _ = VOCDetection(os.environ['DATASETS_ROOT'], year='2012', image_set='train', download=True)


if __name__ == '__main__':
    download_dataset()