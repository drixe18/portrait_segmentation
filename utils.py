import pydoc
from typing import List
import torch
import numpy as np
import random
import albumentations as albu
from albumentations.pytorch import ToTensor
import albumentations.augmentations.functional as F
from copy import deepcopy

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = dict(d.copy())    
    object_type = kwargs.pop("type")

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)

"""Helpers for pytorch lightning."""
def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def set_determenistic(seed=666, precision=10):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def get_transform(conf):
    conf_augmentation = deepcopy(conf)
    def get_object(trans):
        if trans.type in {'albumentations.Compose', 'albumentations.OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.contain]
            _ = trans.pop("contain")
            return object_from_dict(trans, transforms=augs_tmp)
        
        return object_from_dict(trans)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]

    return albu.Compose(augs)