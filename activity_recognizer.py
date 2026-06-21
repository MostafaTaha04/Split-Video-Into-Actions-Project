from dataclasses import dataclass
from typing import Iterable, Set


@dataclass
class ActivityDecision:
    """Human-readable activity classification for a segment."""
    label: str
    reason: str
    confidence: float


class ActivityRecognizer:
    """
    Rule-based activity recognizer for PC/hardware assembly videos.

    V3 update:
    - Removed strong relative-time rules.
    - Activity labels are based mainly on detected objects, ROIs, interactions, and motion.
    - Labels are descriptive assistance, not the main evaluated task.
    """

    def __init__(self):
        self.cpu_terms = {"cpu", "processor", "computer processor"}
        self.socket_terms = {
            "cpu socket",
            "socket retention lever",
            "socket retention bracket",
            "cpu socket region",
            "cpu_socket_region",
        }
        self.screw_terms = {"screw", "screwdriver"}
        self.fan_terms = {"fan", "cooling fan"}
        self.ram_terms = {"ram", "ram stick"}
        self.cable_terms = {"cable", "connector"}
        self.thermal_terms = {"thermal paste", "heatsink"}
        self.workspace_terms = {
            "motherboard",
            "motherboard workspace",
            "motherboard_workspace",
            "active motion region",
            "active_motion_region",
        }

    @staticmethod
    def _norm(items: Iterable[str]) -> Set[str]:
        return {
            str(x).strip().lower().replace("_", " ")
            for x in items
            if str(x).strip()
        }

    @staticmethod
    def _has(tools: Set[str], terms: Set[str]) -> bool:
        norm_terms = {t.replace("_", " ") for t in terms}
        return any(term in tools for term in norm_terms)

    def describe_segment(
        self,
        segment,
        total_duration: float,
        segment_index: int,
        num_segments: int,
    ) -> ActivityDecision:
        real_objects = self._norm(getattr(segment, "real_objects_used", []))
        rois = self._norm(getattr(segment, "heuristic_regions", []))
        all_items = real_objects | rois

        interactions = self._norm(getattr(segment, "interaction_types", []))
        motion = float(getattr(segment, "avg_motion_energy", 0.0))
        activity = float(getattr(segment, "avg_activity_level", 0.0))

        has_interaction = bool(interactions - {"none"})
        high_motion = motion >= 3.0 or activity >= 0.38
        medium_motion = motion >= 1.5 or activity >= 0.25
        low_motion = motion < 1.5 and activity < 0.25

        has_cpu = self._has(all_items, self.cpu_terms)
        has_socket = self._has(all_items, self.socket_terms)
        has_screw = self._has(real_objects, self.screw_terms)
        has_fan = self._has(real_objects, self.fan_terms)
        has_ram = self._has(real_objects, self.ram_terms)
        has_cable = self._has(real_objects, self.cable_terms)
        has_thermal = self._has(real_objects, self.thermal_terms)
        has_workspace = self._has(all_items, self.workspace_terms)

        # Real object-driven rules first.
        if has_screw and high_motion:
            return ActivityDecision(
                "Screw operation",
                "screw or screwdriver detected with strong hand/motion signal",
                0.86,
            )

        if has_screw:
            return ActivityDecision(
                "Prepare screw/screwdriver",
                "screw or screwdriver detected",
                0.74,
            )

        if has_fan and has_screw:
            return ActivityDecision(
                "Cooling fan screw installation",
                "fan and screw-related object detected",
                0.86,
            )

        if has_fan:
            return ActivityDecision(
                "Cooling fan positioning",
                "fan/cooling fan detected",
                0.76,
            )

        if has_ram and has_interaction:
            return ActivityDecision(
                "RAM module insertion/pressing",
                "RAM detected with hand-object interaction",
                0.84,
            )

        if has_ram:
            return ActivityDecision(
                "RAM module positioning",
                "RAM detected",
                0.72,
            )

        if has_cable and has_interaction:
            return ActivityDecision(
                "Cable connection",
                "cable or connector detected with hand interaction",
                0.84,
            )

        if has_cable:
            return ActivityDecision(
                "Cable positioning",
                "cable or connector detected",
                0.72,
            )

        if has_thermal and high_motion:
            return ActivityDecision(
                "Thermal/heatsink preparation",
                "thermal paste or heatsink detected with motion",
                0.78,
            )

        if has_thermal:
            return ActivityDecision(
                "Prepare thermal paste or heatsink",
                "thermal paste/heatsink detected",
                0.70,
            )

        # CPU/socket rules are intentionally more general now.
        if has_cpu and has_socket and has_interaction and medium_motion:
            return ActivityDecision(
                "CPU/socket manipulation",
                "CPU/processor and socket region detected with hand interaction",
                0.86,
            )

        if has_cpu and has_socket:
            if low_motion:
                return ActivityDecision(
                    "CPU/socket inspection",
                    "CPU/processor and socket detected with low motion",
                    0.76,
                )

            return ActivityDecision(
                "CPU/socket alignment or seating",
                "CPU/processor and socket detected with motion",
                0.80,
            )

        if has_cpu:
            return ActivityDecision(
                "CPU handling",
                "CPU/processor detected",
                0.74,
            )

        if has_socket and has_interaction and high_motion:
            return ActivityDecision(
                "Socket mechanism manipulation",
                "socket ROI detected with interaction and strong motion",
                0.76,
            )

        if has_socket and has_interaction:
            return ActivityDecision(
                "Socket area interaction",
                "socket ROI detected with hand interaction",
                0.72,
            )

        if has_socket and medium_motion:
            return ActivityDecision(
                "Socket area manipulation",
                "socket ROI active with motion",
                0.68,
            )

        if has_socket:
            return ActivityDecision(
                "Socket area inspection",
                "socket ROI detected with low motion",
                0.62,
            )

        # Generic fallback.
        if has_workspace and high_motion:
            return ActivityDecision(
                "Workspace manipulation",
                "workspace ROI active with strong motion",
                0.60,
            )

        if has_workspace and medium_motion:
            return ActivityDecision(
                "Workspace adjustment",
                "workspace ROI active with moderate motion",
                0.58,
            )

        if segment_index == num_segments - 1 and low_motion:
            return ActivityDecision(
                "Final inspection or pause",
                "last segment with low motion",
                0.56,
            )

        if medium_motion:
            return ActivityDecision(
                "Hand movement / transition",
                "motion detected but no specific component recognized",
                0.52,
            )

        return ActivityDecision(
            "Inspection or pause",
            "low motion and no specific recognized component",
            0.50,
        )