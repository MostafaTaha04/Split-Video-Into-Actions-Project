"""
labels.py
=========
The activities the system recognises, and how they are drawn.

Every moment of a video gets exactly one of these 13 activity labels, or
BACKGROUND when none of them is happening (reaching, looking, pausing).
The video is cut only where the label changes.
"""

ACTIVITIES = [
    "CLOSE", "COVER", "EXTRACT", "FIXING_FAN", "INSERT", "OPEN", "PLUG",
    "PLUG_FAN", "SCREW", "UNCOVER", "UNPLUG", "UNPLUG_FAN", "UNSCREW",
]
BACKGROUND = "BACKGROUND"
CLASSES = [BACKGROUND] + ACTIVITIES          # index 0 = background
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Six activities come in opposite pairs; FIXING_FAN has no opposite. The pairs
# matter for evaluation: confusing SCREW with UNSCREW is a different (and
# harder) mistake than confusing SCREW with PLUG.
OPPOSITES = {
    "OPEN": "CLOSE", "COVER": "UNCOVER", "INSERT": "EXTRACT",
    "PLUG": "UNPLUG", "PLUG_FAN": "UNPLUG_FAN", "SCREW": "UNSCREW",
}
OPPOSITES.update({b: a for a, b in list(OPPOSITES.items())})

# One colour per label; each pair shares a hue (dark = putting together,
# light = taking apart) so related activities read as related on the timeline.
LABEL_COLORS = {
    "SCREW": "#1f4e9c", "UNSCREW": "#7fa7e0",
    "PLUG": "#1e7a3c", "UNPLUG": "#86c79a",
    "PLUG_FAN": "#0f7c80", "UNPLUG_FAN": "#7cc8c9",
    "INSERT": "#c25a00", "EXTRACT": "#f2a65a",
    "COVER": "#6a3d9a", "UNCOVER": "#b9a0dc",
    "CLOSE": "#a8202a", "OPEN": "#ee8b92",
    "FIXING_FAN": "#7a5230",
    BACKGROUND: "#9aa3ad",
}


def normalise(label: str) -> str:
    """'unplug fan' / 'Unplug-Fan' -> 'UNPLUG_FAN'; raises on an unknown label."""
    key = str(label).strip().upper().replace("-", "_").replace(" ", "_")
    if key in ("", "NONE", "BG", "OTHER", "NULL", "SIL", "BACKGROUND"):
        return BACKGROUND
    if key not in CLASS_INDEX:
        raise ValueError(f"unknown activity label {label!r}; expected one of {CLASSES}")
    return key
