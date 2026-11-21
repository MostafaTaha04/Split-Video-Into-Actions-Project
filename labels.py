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

# Which activities happen when a PC is put together and when it is taken apart.
# OPEN and CLOSE (a latch or lever) can happen in both.
TASKS = ("assembly", "disassembly")
TASK_ACTIVITIES = {
    "assembly": ["COVER", "INSERT", "PLUG", "PLUG_FAN", "SCREW", "FIXING_FAN", "OPEN", "CLOSE"],
    "disassembly": ["UNCOVER", "EXTRACT", "UNPLUG", "UNPLUG_FAN", "UNSCREW", "OPEN", "CLOSE"],
}

# Each pair without its direction. Frame by frame, screwing and unscrewing (or
# plugging and unplugging) look almost the same; whether the whole video is an
# assembly or a disassembly is much easier to see. The model can therefore learn
# the action without its direction and add the direction from the video's task.
ACTION_TYPE = {
    "SCREW": "SCREWING", "UNSCREW": "SCREWING",
    "PLUG": "CABLE", "UNPLUG": "CABLE",
    "PLUG_FAN": "FAN_CABLE", "UNPLUG_FAN": "FAN_CABLE",
    "INSERT": "PART", "EXTRACT": "PART",
    "COVER": "SIDE_COVER", "UNCOVER": "SIDE_COVER",
    "FIXING_FAN": "FIXING_FAN", "OPEN": "OPEN", "CLOSE": "CLOSE",
    BACKGROUND: BACKGROUND,
}


def with_direction(action_type: str, task: str) -> str:
    """'SCREWING' + 'disassembly' -> 'UNSCREW'; labels without a direction are returned as they are."""
    for label in TASK_ACTIVITIES.get(task, []):
        if ACTION_TYPE[label] == action_type:
            return label
    # BACKGROUND, or an action that does not belong to this task (e.g. FIXING_FAN while taking a PC apart)
    return BACKGROUND


def task_action_types(task: str) -> list:
    """The direction-free actions that can happen in this task, plus BACKGROUND."""
    return [BACKGROUND] + sorted({ACTION_TYPE[a] for a in TASK_ACTIVITIES[task]})

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
