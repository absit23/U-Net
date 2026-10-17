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

VARIANTS = {"7class": CLASSES_7, "6class": CLASSES_6}

# Hand-set starting weights (by name, so both variants use the same values).
# Prefer `--class_weights auto` to derive them from the training masks.
_MANUAL_WEIGHTS = {
    "Background": 0.2, "Delamination": 1.0, "Efflorescence": 2.5,
    "Microcrack": 6.0, "Minor spalling": 12.0, "Moderate spalling": 2.0,
    "Rebar corrosion": 0.5, "Structural crack": 1.0,
}


def get_class_names(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}.")
    return list(VARIANTS[variant])


def default_class_weights(class_names):
    return [_MANUAL_WEIGHTS[name] for name in class_names]
