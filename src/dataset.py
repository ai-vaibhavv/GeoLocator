"""Image loading, augmentation, and the per-row tensors the model trains on."""

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class GeoDataset(Dataset):
    """dataset row -> (image, lat, lng[, coarse cell][, fine cell])."""

    AUG_DEFAULT = {
        "crop": True,
        "rotation": True,
        "perspective": True,
        "jitter": True,
        "erasing": True,
        "hflip": False,
    }

    def __init__(
        self,
        rows,
        img_dir,
        image_size,
        augment=False,
        return_cell=False,
        return_cell2=False,
        aug_spec=None,
    ):
        """Build the dataset and the transform pipeline its rows go through.

        Args:
            rows: row dicts carrying filename, lat, lng and optionally cell/cell2.
            img_dir: directory the filenames are relative to.
            image_size: side length the images are resized to.
            augment: whether to apply the training augmentations.
            return_cell: also return the coarse cell index.
            return_cell2: also return the fine cell index (implies return_cell).
            aug_spec: overrides for AUG_DEFAULT; ignored when augment is False.
        """
        self.rows = rows
        self.img_dir = img_dir
        self.return_cell = return_cell or return_cell2
        self.return_cell2 = return_cell2

        a = dict(self.AUG_DEFAULT, **(aug_spec or {})) if augment else {}
        if a.get("crop"):
            steps = [
                T.Resize((image_size + 16, image_size + 16)),
                T.RandomCrop(image_size),
            ]
        else:
            steps = [T.Resize((image_size, image_size))]
        if a.get("hflip"):
            steps.append(T.RandomHorizontalFlip(p=0.5))
        if a.get("rotation"):
            steps.append(T.RandomRotation(15))
        if a.get("perspective"):
            steps.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))
        if a.get("jitter"):
            steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        steps += [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
        if a.get("erasing"):
            steps.append(T.RandomErasing(p=0.25, scale=(0.02, 0.1)))
        self.transform = T.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(f"{self.img_dir}/{r['filename']}").convert("RGB")
        out = [
            self.transform(img),
            torch.tensor(float(r["lat"])),
            torch.tensor(float(r["lng"])),
        ]
        if self.return_cell:
            out.append(torch.tensor(r["cell"], dtype=torch.long))
        if self.return_cell2:
            out.append(torch.tensor(r["cell2"], dtype=torch.long))
        return tuple(out)
