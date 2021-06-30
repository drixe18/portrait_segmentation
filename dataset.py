from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import numpy as np
import cv2 as cv
import albumentations as albu
import torch
from torch.utils.data import Dataset
import os
from glob import glob


def get_aisegment_paths(path_folder='aisegment'):
    image_dir = os.path.join(path_folder, 'clip_img')
    mask_dir = os.path.join(path_folder, 'matting')
    
    images = sorted(glob(os.path.join(image_dir, '*/*/*.jpg')))
    masks = sorted(glob(os.path.join(mask_dir, '*/*/*.png')))
    
    # fix
    try:
        masks.remove(os.path.join(mask_dir, '1803201916/matting_00000000/1803201916-00000117.png'))
    except:
        pass
    
    print(f'Найдено {len(images)} изображений, {len(masks)} масок')
    
    samples = [(Path(i[0]), Path(i[1])) for i in zip(images, masks)]
    return samples


def get_fixed_aisegment_paths(path_folder='aisegment'):
    image_dir = os.path.join(path_folder, 'clip_img')
    mask_dir = 'aisegment_by_ter'

    images = sorted(glob(os.path.join(image_dir, '*/*/*.jpg')))
    # masks = sorted(glob(os.path.join(mask_dir, '*.png')))
    masks = sorted(glob(os.path.join(mask_dir, '*.jpg')))

    # fix
    # masks.remove(os.path.join(mask_dir, '1803201916-00000117.png'))

    print(f'Найдено {len(images)} изображений, {len(masks)} масок')

    samples = [(Path(i[0]), Path(i[1])) for i in zip(images, masks)]
    return samples


def get_eg1800_paths(path_folder='eg1800/EG1800'):
    image_dir = os.path.join(path_folder, 'images_data_crop')
    mask_dir = os.path.join(path_folder, 'GT_png')

    images = sorted(glob(os.path.join(image_dir, '*.jpg')))
    masks = sorted(glob(os.path.join(mask_dir, '*.png')))
    print(f'Найдено {len(images)} изображений, {len(masks)} масок')

    # fix matching
    masks = [i for i in masks if i.split('/')[-1].split('_')[0] in [i.split('/')[-1].split('.')[0] for i in images]]
    print(f'Загружено {len(images)} изображений, {len(masks)} масок')
    return [(Path(i[0]), Path(i[1])) for i in zip(images, masks)]


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        length: int = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]

        image = cv.cvtColor(cv.imread(str(image_path)), cv.COLOR_BGR2RGB)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image = image / 127.5
        image -= 1
        
        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": image_path.stem,
            "features": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "masks": torch.unsqueeze(mask, 0).float(),
        }


if __name__ == "__main__":
    samples = get_eg1800_paths()
    
    samples = get_fixed_aisegment_paths()
    print(samples[-1])
    print(len(samples))
    dataset = SegmentationDataset(samples=samples, transform=None)
    sample = dataset[0]
    print(sample)
    print(sample['features'].shape, sample['features'].min(), sample['features'].max())
    print(sample['masks'].shape, sample['masks'].min(), sample['masks'].max())
