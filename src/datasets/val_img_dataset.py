import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

from .data_utils import get_tensor, logical_or_reduce


class ValImgDataset(BaseDataset):
    def __init__(self, data_dir, use_blur=True, blur_scale=10, *args, **kwargs):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        self._img_dir = ROOT_PATH / data_dir
        self._mask_dir = ROOT_PATH / data_dir / "BiSeNet_mask"
        self._data_dir = ROOT_PATH / "data"
        self._data_dir.mkdir(exist_ok=True)
        index = self._get_or_load_index()
        self.use_blur = use_blur
        self.blur_scale = blur_scale

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "val_img.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        for i in range(7):
            index.append(
                {
                    "target_img_path": str(self._img_dir / "target" / f"{i}.png"),
                    "source_img_path": str(self._img_dir / "source" / f"{i}.png"),
                    "target_mask_path": str(self._mask_dir / "target" / f"{i}.png"),
                    "source_mask_path": str(self._mask_dir / "source" / f"{i}.png"),
                }
            )
        return index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]

        target_img_path = data_dict["target_img_path"]
        source_img_path = data_dict["source_img_path"]
        target_mask_path = data_dict["target_mask_path"]
        source_mask_path = data_dict["source_mask_path"]

        target_img = Image.open(target_img_path).convert("RGB").resize((512, 512))
        source_img = Image.open(source_img_path).convert("RGB").resize((512, 512))

        target_mask = Image.open(target_mask_path).convert("L").resize((512, 512))
        source_mask = Image.open(source_mask_path).convert("L").resize((512, 512))

        target_img = get_tensor()(target_img)
        source_img = get_tensor()(source_img)

        target_mask = np.array(target_mask)
        source_mask = np.array(source_mask)
        target_mask = logical_or_reduce(
            *[target_mask == item for item in [1, 2, 3, 5, 6, 7, 9]]
        ).astype(float)
        source_mask = logical_or_reduce(
            *[source_mask == item for item in [1, 2, 3, 5, 6, 7, 9]]
        ).astype(float)
        target_mask = torch.from_numpy(target_mask)
        source_mask = torch.from_numpy(source_mask)

        masked_img = target_img * (1 - target_mask)

        masked_source_img = source_img * source_mask
        # masked_source_img = transforms.Grayscale(3)(masked_source_img)

        if self.use_blur:
            blur_image = torch.nn.functional.interpolate(
                target_img.unsqueeze(0), (self.blur_scale, self.blur_scale)
            )
            blur_image = torch.nn.functional.interpolate(
                blur_image, (512, 512)
            ).squeeze()
        else:
            blur_image = None

        return {
            "only_source_img": source_img,
            "target_img": target_img,
            "inpaint_img": masked_img,
            "mask": target_mask,
            "corrupt_img": blur_image,
            "source_img": masked_source_img,
        }
