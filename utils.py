import os
from typing import Dict, List
import math
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_categories(labels_dir: str) -> List[str]:
    """
    Get the object categories for Pascal VOC 2012

    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])

        return categories

    
class VOCLabelTransform:
    def __init__(self, object_categories: List, exclude_difficult: bool = True):
        """VOC Label Transform
        Args:
            object_categories (list): List of object categories in the desired order.
            exclude_difficult (bool): If True, exclude objects labeled as difficult.
        """
        self.object_categories = object_categories
        self.exclude_difficult = exclude_difficult

    def __call__(self, target: Dict):
        """
        Encode multiple labels using 1/0 encoding. Index ordering is as per the provided "object_categories".

        Args:
            target: xml tree file
        Returns:
            torch tensor encoding labels as 1/0 vector
        """
        ls = target['annotation']['object']

        j = []
        if type(ls) == dict:
            if int(ls['difficult']) == 0:
                j.append(self.object_categories.index(ls['name']))
            elif int(ls['difficult']) == 1 and self.exclude_difficult == False:
                j.append(self.object_categories.index(ls['name']))
        else:
            for i in range(len(ls)):
                if int(ls[i]['difficult']) == 0:
                    j.append(self.object_categories.index(ls[i]['name']))
                elif int(ls[i]['difficult']) == 1 and self.exclude_difficult == False:
                    j.append(self.object_categories.index(ls[i]['name']))

        k = np.zeros(len(self.object_categories), dtype=int)
        k[j] = 1 # object present

        return torch.from_numpy(k)


def get_label_txt(label: torch.Tensor, object_categories: List) -> List[str]:
    """
    Get Label text from numpy array encoded as 0/1 vector

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
    
    