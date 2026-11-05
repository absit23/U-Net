import os

import cv2
from torch.utils.data import Dataset, DataLoader

from .copy_paste import CopyPaste
from .transforms import get_transforms

IMG_EXTS = (".png", ".jpg", ".jpeg")


def resolve_split_dirs(data_dir, split):
    """Return (images_dir, masks_dir) for a split, or None if the split is absent.

    Supported layouts:
      A) <data_dir>/<split>/images and <data_dir>/<split>/masks
      B) Roboflow flat export: <data_dir>/<split>/xxx.jpg + xxx_mask.png
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        return None
    img_dir, mask_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "masks")
    if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
        return img_dir, mask_dir
    return split_dir, split_dir


class ConcreteDamageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, copy_paste=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.copy_paste = None
        self.image_filenames = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith(IMG_EXTS)
            and not os.path.splitext(f)[0].endswith("_mask")  # skip masks in flat layout
        )
        if copy_paste:  # dict(classes=[...], prob=..., max_pastes=..., min_area=...) -- train split only
            self.copy_paste = CopyPaste(self, **copy_paste)
            print(f"Copy-paste on: donor images per class {self.copy_paste.summary()} | "
                  f"prob={copy_paste.get('prob', 0.5)}")

    def _find_mask_path(self, img_name):
        stem = os.path.splitext(img_name)[0]
        candidates = [img_name]
        candidates += [stem + ext for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")]
        candidates += [f"{stem}_mask{ext}" for ext in (".png", ".jpg", ".jpeg")]
        for name in candidates:
            path = os.path.join(self.masks_dir, name)
            # In the flat layout the image itself must not be mistaken for its mask.
            if os.path.exists(path) and not (self.masks_dir == self.images_dir and name == img_name):
                return path
        return None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_path = self._find_mask_path(img_name)
        if mask_path is None:
            raise FileNotFoundError(f"Mask for '{img_name}' not found in '{self.masks_dir}'.")

        image = cv2.imread(os.path.join(self.images_dir, img_name))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        if self.copy_paste is not None:
            image, mask = self.copy_paste(image, mask, idx)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return {"image": image, "mask": mask.long()}


def build_loader(data_dir, split, img_size, batch_size, num_workers, is_train, pin_memory=False, copy_paste=None):
    dirs = resolve_split_dirs(data_dir, split)
    if dirs is None:
        raise FileNotFoundError(f"Split '{split}' not found under '{data_dir}'.")
    dataset = ConcreteDamageDataset(*dirs, transform=get_transforms(img_size, is_train),
                                    copy_paste=copy_paste if is_train else None)
    if len(dataset) == 0:
        raise ValueError(f"No images found for split '{split}' in '{dirs[0]}'.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                      num_workers=num_workers, pin_memory=pin_memory)
