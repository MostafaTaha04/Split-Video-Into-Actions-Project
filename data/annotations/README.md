# Annotations

One JSON file per video, listing every activity with its start and end time.
The video `data/videos/<name>.mp4` is paired with `data/annotations/<name>.json`.

## Format

```json
{
  "video": "pc_assembly_01.mp4",
  "task": "assembly",
  "duration_s": 573.6,
  "provenance": { "...": "who made the file and whether it has been checked" },
  "segments": [
    {"start": 19.0, "end": 32.0, "label": "INSERT",
     "note": "Motherboard lowered into the case."}
  ]
}
```

- `start` and `end` are in seconds from the start of the video.
- `label` must be one of the 13 activities in `labels.py`.
- Any time not covered by a segment is **BACKGROUND** (setting up, putting
  gloves on, looking around, reaching for the next part).
- `note` (optional) describes what happens in the segment.

## What each label means

| Label | Meaning in these videos |
|---|---|
| COVER / UNCOVER | Put the case side cover back on / take it off |
| INSERT / EXTRACT | Put a part in / take it out (motherboard, CPU, RAM stick, power supply, drive) |
| PLUG / UNPLUG | Connect / disconnect a power or data cable |
| PLUG_FAN / UNPLUG_FAN | Connect / disconnect a fan cable on the motherboard |
| FIXING_FAN | Mount a fan (CPU cooler or case fan) in its place |
| SCREW / UNSCREW | Drive screws in / take them out with the screwdriver |
| OPEN / CLOSE | Open / close a latch or lever (RAM clip, CPU socket lever) |

## Current files

| File | Segments |
|---|---|
| `pc_assembly_01.json` | 11 |
| `pc_assembly_02.json` | 11 |
| `pc_assembly_03.json` | 11 |
| `pc_assembly_04.json` | 10 (the video ends before the cover goes back on) |
| `pc_disassembly_01.json` | 14 |
| `pc_disassembly_02.json` | 13 |
| `pc_disassembly_03.json` | 13 |
| `pc_disassembly_04.json` | 11 |

(`pc_assembly_05.mp4` was a byte-for-byte copy of `pc_assembly_01.mp4` and was
removed, so the same video isn't counted twice.)

How they were made: a first draft was made with an AI tool from the video
frames; Mostafa Taha and Osama Najjar then checked every file against the video
and corrected it. Boundaries are accurate to about 1 second.

The assembly videos all follow the same order:
1. Motherboard `INSERT` → `SCREW`.
2. CPU `INSERT`.
3. CPU cooler `FIXING_FAN`.
4. Rear fan `FIXING_FAN` → `SCREW`.
5. RAM `INSERT`.
6. Power supply `INSERT` → `SCREW`.
7. Cables `PLUG`.
8. `COVER`.

The disassembly videos all follow the same order:
1. `UNCOVER`.
2. Cables `UNPLUG`, then the cooler's fan cable `UNPLUG_FAN`.
3. CPU cooler `UNSCREW` → `EXTRACT`.
4. A drive `EXTRACT` (not seen in `pc_disassembly_04`).
5. Rear fan `UNSCREW` → `EXTRACT`.
6. Power supply `UNSCREW` → `EXTRACT`.
7. Motherboard `UNSCREW` → `EXTRACT`.

`OPEN`/`CLOSE` (CPU socket lever, RAM clips) and `PLUG_FAN` are not marked
separately in these videos: they are very short and happen inside the CPU, RAM
and cable steps.

## Checking a file in the player

```
python viewer.py --video data/videos/pc_disassembly_01.mp4 --segments data/annotations/pc_disassembly_01.json --no-clips
```

Open the page it prints. Each step is coloured on the timeline under the video.
Use the arrow keys to move from step to step and `L` to loop the current step.
If a time or a label is wrong, edit the JSON file and run the command again.
