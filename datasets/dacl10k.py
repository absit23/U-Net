"""dacl10k loader: Supervisely polygon annotations -> 19-channel multi-label masks.

dacl10k is a MULTI-LABEL segmentation dataset: a single pixel can carry several
classes at once (Spalling with Exposed Rebars covered by Rust). So the target is
not one integer per pixel but 19 independent binary channels, and the model needs
a sigmoid per channel rather than a softmax over classes.

Expected layout (the Dataset Ninja / Supervisely export):

    <root>/train/img/*.jpg      <root>/train/ann/*.jpg.json
    <root>/val/img/*.jpg        <root>/val/ann/*.jpg.json
    <root>/test/img/*.jpg       <root>/test/ann/*.jpg.json   (annotations are EMPTY)
    <root>/meta.json

The official test annotations were never released: those json files carry no
objects, so a model can only be scored on test by submitting to CodaLab. Use
`val` (975 images) as the local test set.

Polygons are rasterised directly at `img_size` by scaling the point coordinates,
instead of rasterising at full resolution and then resizing. Source images run up
to 5184x3456, so this is the difference between milliseconds and seconds per item.
"""
import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = (".jpg", ".jpeg", ".png")

# meta.json order (alphabetical). Keep this fixed: channel index == position here.
DACL_CLASSES = [
    "alligator crack",
    "bearing",
    "cavity",
    "crack",
    "drainage",
    "efflorescence",
    "expansion joint",
    "exposed rebars",
    "graffiti",
    "hollowareas",
    "joint tape",
    "protective equipment",
    "restformwork",
    "rockpocket",
    "rust",
    "spalling",
    "washouts/concrete corrosion",
    "weathering",
    "wetspot",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(DACL_CLASSES)}
NUM_CLASSES = len(DACL_CLASSES)

# The dataset's own grouping, from the meta.json tags.
CONCRETE_DEFECTS = ["crack", "alligator crack", "efflorescence", "rockpocket",
                    "washouts/concrete corrosion", "hollowareas", "cavity",
                    "spalling", "restformwork"]
GENERAL_DEFECTS = ["wetspot", "rust", "graffiti", "weathering"]
OBJECT_PARTS = ["exposed rebars", "bearing", "expansion joint", "drainage",
                "protective equipment", "joint tape"]

CONCRETE_IDX = [CLASS_TO_IDX[c] for c in CONCRETE_DEFECTS]
GENERAL_IDX = [CLASS_TO_IDX[c] for c in GENERAL_DEFECTS]
OBJECT_IDX = [CLASS_TO_IDX[c] for c in OBJECT_PARTS]


def verify_meta(root):
    """Check the shipped meta.json against DACL_CLASSES; raise on a mismatch."""
    path = os.path.join(root, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        titles = sorted(c["title"] for c in json.load(f)["classes"])
    if titles != sorted(DACL_CLASSES):
        raise ValueError(
            "meta.json classes do not match DACL_CLASSES.\n"
            f"  only in meta.json: {sorted(set(titles) - set(DACL_CLASSES))}\n"
            f"  only in code:      {sorted(set(DACL_CLASSES) - set(titles))}")
    return titles


def polygons_to_mask(ann, out_h, out_w):
    """Supervisely annotation dict -> (out_h, out_w, 19) uint8, one channel per class.

    Interior rings are holes and are punched back out after the exterior is filled.
    Returns the mask and the number of objects whose class was not recognised.
    """
    # Filled as [C,H,W] so each layer is contiguous: cv2.fillPoly cannot write
    # into the strided view mask[:, :, c] of an [H,W,C] array.
    planes = np.zeros((NUM_CLASSES, out_h, out_w), dtype=np.uint8)
    src_h = int(ann.get("size", {}).get("height") or 0)
    src_w = int(ann.get("size", {}).get("width") or 0)
    if not src_h or not src_w:
        return np.ascontiguousarray(planes.transpose(1, 2, 0)), 0
    sx, sy = out_w / src_w, out_h / src_h
    unknown = 0

    for obj in ann.get("objects", []):
        idx = CLASS_TO_IDX.get(obj.get("classTitle"))
        if idx is None:
            unknown += 1
            continue
        pts = obj.get("points") or {}
        exterior = pts.get("exterior") or []
        if len(exterior) < 3:
            continue
        layer = planes[idx]
        ring = np.array(exterior, dtype=np.float64)
        ring[:, 0] *= sx
        ring[:, 1] *= sy
        cv2.fillPoly(layer, [np.round(ring).astype(np.int32)], 1)
        for hole in pts.get("interior") or []:
            if len(hole) < 3:
                continue
            h = np.array(hole, dtype=np.float64)
            h[:, 0] *= sx
            h[:, 1] *= sy
            cv2.fillPoly(layer, [np.round(h).astype(np.int32)], 0)

    return np.ascontiguousarray(planes.transpose(1, 2, 0)), unknown


class Dacl10kDataset(Dataset):
    """One item = {"image": float tensor [3,S,S], "mask": float tensor [19,S,S]}."""

    def __init__(self, root, split, img_size=512, transform=None, strict=True):
        self.img_dir = os.path.join(root, split, "img")
        self.ann_dir = os.path.join(root, split, "ann")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"No image folder at '{self.img_dir}'.")
        self.img_size = int(img_size)
        self.transform = transform
        self.strict = strict
        self.files = sorted(f for f in os.listdir(self.img_dir)
                            if f.lower().endswith(IMG_EXTS))
        if not self.files:
            raise ValueError(f"No images found in '{self.img_dir}'.")
        self.unknown_classes = 0

    def __len__(self):
        return len(self.files)

    def ann_path(self, img_name):
        """Supervisely names the annotation '<image file name>.json'."""
        direct = os.path.join(self.ann_dir, img_name + ".json")
        if os.path.exists(direct):
            return direct
        stem = os.path.splitext(img_name)[0] + ".json"
        alt = os.path.join(self.ann_dir, stem)
        return alt if os.path.exists(alt) else None

    def __getitem__(self, idx):
        name = self.files[idx]
        image = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)

        path = self.ann_path(name)
        if path is None:
            if self.strict:
                raise FileNotFoundError(f"Annotation for '{name}' not found in "
                                        f"'{self.ann_dir}'.")
            mask = np.zeros((self.img_size, self.img_size, NUM_CLASSES), np.uint8)
        else:
            with open(path, encoding="utf-8") as f:
                ann = json.load(f)
            mask, unknown = polygons_to_mask(ann, self.img_size, self.img_size)
            self.unknown_classes += unknown

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return {"image": image, "mask": mask.float(), "name": name}


def get_dacl_transforms(img_size=512, is_train=True):
    """Augmentation for multi-label masks.

    transpose_mask=True is what makes ToTensorV2 return [C,H,W] instead of [H,W,C].
    No Resize here: the image and the polygons are already rasterised at img_size.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tail = [A.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)]
    if not is_train:
        return A.Compose(tail)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625),
                 rotate=(-45, 45), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        *tail,
    ])


def build_dacl_loader(root, split, img_size=512, batch_size=8, num_workers=2,
                      is_train=None, pin_memory=False, shuffle=None):
    if is_train is None:
        is_train = split == "train"
    if shuffle is None:
        shuffle = is_train
    dataset = Dacl10kDataset(root, split, img_size,
                             transform=get_dacl_transforms(img_size, is_train),
                             strict=(split != "test"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
