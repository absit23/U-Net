"""Class definitions. Edit class names / weights HERE only.

Mask IDs follow the order below (Background = 0). This order MUST match the
mask values of your Roboflow export (Roboflow sorts class names alphabetically).
"""

# 7-class variant = 6 base damage classes + Efflorescence
CLASSES_7 = [
    "Background", "Delamination", "Efflorescence", "Microcrack",
    "Minor spalling", "Moderate spalling", "Rebar corrosion", "Structural crack",
]

# 6-class variant = same, without Efflorescence
CLASSES_6 = [c for c in CLASSES_7 if c != "Efflorescence"]

# Merged-severity variant: Delamination + Rebar corrosion -> Severe spalling.
# Masks must come from make_merged_dataset.py (IDs are baked into the mask files).
CLASSES_M = [
    "Background", "Efflorescence", "Microcrack", "Minor spalling",
    "Moderate spalling", "Severe spalling", "Structural crack",
]

# Original Mendeley/Roboflow dataset, untouched (8 damage classes, no Efflorescence).
CLASSES_O = [
    "Background", "Concrete crushing", "Delamination", "Microcrack",
    "Minor spalling", "Moderate spalling", "Rebar corrosion",
    "Structural crack", "Structural deformation",
]

# crackmerged = 7class with Microcrack + Structural crack merged into one "Crack"
# class. The severity split is recovered afterwards from the predicted crack
# width (reconstruct_eval.py) instead of being forced on the network. No new
# masks are needed: mask value 7 is remapped onto 3, which is renamed "Crack".
CLASSES_CRACKMERGED = [
    "Background", "Delamination", "Efflorescence", "Crack",
    "Minor spalling", "Moderate spalling", "Rebar corrosion",
]

VARIANTS = {"7class": CLASSES_7, "6class": CLASSES_6, "merged6": CLASSES_M,
            "orig8": CLASSES_O, "crackmerged": CLASSES_CRACKMERGED}

# Mask-value remapping applied at load time when a variant's IDs differ from the
# 7class export on disk. {variant: {source_id: target_id}}
REMAPS = {"crackmerged": {7: 3}}


def get_mask_remap(variant):
    """Return a 256-entry lookup table for cv2.LUT, or None if no remap is needed."""
    mapping = REMAPS.get(variant)
    if not mapping:
        return None
    import numpy as np
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut

# Hand-set starting weights (by name, so both variants use the same values).
# Prefer `--class_weights auto` to derive them from the training masks.
_MANUAL_WEIGHTS = {
    "Background": 0.2, "Delamination": 1.0, "Efflorescence": 2.5,
    "Microcrack": 6.0, "Minor spalling": 12.0, "Moderate spalling": 2.0, "Severe spalling": 0.5, "Concrete crushing": 2.0, "Structural deformation": 2.0,
    "Rebar corrosion": 0.5, "Structural crack": 1.0, "Crack": 1.0,
}


def get_class_names(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}.")
    return list(VARIANTS[variant])


def default_class_weights(class_names):
    return [_MANUAL_WEIGHTS[name] for name in class_names]
