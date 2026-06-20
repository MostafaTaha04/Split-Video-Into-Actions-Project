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

    The object detector answers: "what is visible?"
    The temporal segmenter answers: "when does the action change?"
    This class combines detected tools/components + interactions + motion + time order
    to produce a practical human-readable activity label.
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
        tools = self._norm(getattr(segment, "tools_used", []))
        interactions = self._norm(getattr(segment, "interaction_types", []))
        motion = float(getattr(segment, "avg_motion_energy", 0.0))
        activity = float(getattr(segment, "avg_activity_level", 0.0))
        duration = float(getattr(segment, "duration", 0.0))

        if total_duration <= 0:
            total_duration = max(duration, 1.0)

        mid_time = 0.5 * (float(segment.start_time) + float(segment.end_time))
        rel = mid_time / total_duration

        has_cpu = self._has(tools, self.cpu_terms)
        has_socket = self._has(tools, self.socket_terms)
        has_screw = self._has(tools, self.screw_terms)
        has_fan = self._has(tools, self.fan_terms)
        has_ram = self._has(tools, self.ram_terms)
        has_cable = self._has(tools, self.cable_terms)
        has_thermal = self._has(tools, self.thermal_terms)
        has_workspace = self._has(tools, self.workspace_terms)

        has_interaction = bool(interactions - {"none"})
        high_motion = motion >= 3.0 or activity >= 0.38
        medium_motion = motion >= 1.5 or activity >= 0.25

        # Object-specific activities first.
        if has_screw and high_motion:
            return ActivityDecision(
                "Tighten or loosen screw",
                "screw/screwdriver detected with strong local motion",
                0.86,
            )

        if has_screw:
            return ActivityDecision(
                "Prepare screw or screwdriver",
                "screw/screwdriver detected",
                0.74,
            )

        if has_fan and has_screw:
            return ActivityDecision(
                "Install cooling fan with screws",
                "cooling fan and screw-related object detected",
                0.86,
            )

        if has_fan:
            return ActivityDecision(
                "Position cooling fan",
                "fan/cooling fan detected",
                0.76,
            )

        if has_ram and has_interaction:
            return ActivityDecision(
                "Insert or press RAM module",
                "RAM detected with hand interaction",
                0.84,
            )

        if has_ram:
            return ActivityDecision(
                "Position RAM module",
                "RAM detected",
                0.72,
            )

        if has_cable and has_interaction:
            return ActivityDecision(
                "Connect cable to connector",
                "cable/connector detected with hand interaction",
                0.84,
            )

        if has_cable:
            return ActivityDecision(
                "Position cable near connector",
                "cable/connector detected",
                0.72,
            )

        if has_thermal and high_motion:
            return ActivityDecision(
                "Apply or spread thermal paste / heatsink preparation",
                "thermal paste or heatsink detected with motion",
                0.78,
            )

        if has_thermal:
            return ActivityDecision(
                "Prepare heatsink or thermal paste",
                "thermal/heatsink object detected",
                0.70,
            )

        if has_cpu and has_socket and has_interaction:
            if rel < 0.45:
                return ActivityDecision(
                    "Position CPU into socket",
                    "CPU and socket detected with hand interaction",
                    0.86,
                )
            return ActivityDecision(
                "Seat or adjust CPU in socket",
                "CPU/socket interaction after initial positioning",
                0.84,
            )

        if has_cpu and has_socket:
            return ActivityDecision(
                "Align CPU with socket",
                "CPU and socket detected",
                0.80,
            )

        if has_cpu:
            return ActivityDecision(
                "Handle CPU component",
                "CPU/processor detected",
                0.74,
            )

        if has_socket and medium_motion:
            if rel < 0.35:
                return ActivityDecision(
                    "Prepare CPU socket area",
                    "socket region active early in the video",
                    0.70,
                )

            if rel < 0.70:
                return ActivityDecision(
                    "Adjust socket frame or retention bracket",
                    "socket region active in middle phase",
                    0.72,
                )

            return ActivityDecision(
                "Close or lock CPU socket mechanism",
                "socket region active in late phase",
                0.74,
            )

        if has_socket:
            return ActivityDecision(
                "Inspect CPU socket area",
                "socket region detected with low motion",
                0.64,
            )

        # Generic fallback from workspace + motion + time order.
        if has_workspace and high_motion:
            if rel < 0.20:
                return ActivityDecision(
                    "Prepare motherboard workspace",
                    "workspace active at the beginning",
                    0.62,
                )

            if rel < 0.55:
                return ActivityDecision(
                    "Perform main assembly manipulation",
                    "workspace active with strong motion",
                    0.62,
                )

            if rel < 0.85:
                return ActivityDecision(
                    "Secure or adjust component on motherboard",
                    "workspace active late in video",
                    0.62,
                )

            return ActivityDecision(
                "Final inspection of motherboard area",
                "workspace active near end",
                0.62,
            )

        if medium_motion:
            return ActivityDecision(
                "Hand movement / transition between assembly actions",
                "motion detected but no specific component recognized",
                0.52,
            )

        if segment_index == num_segments - 1 or rel > 0.85:
            return ActivityDecision(
                "Final inspection or pause",
                "low motion near the end",
                0.55,
            )

        return ActivityDecision(
            "Inspection or pause",
            "low motion and no specific recognized component",
            0.50,
        )