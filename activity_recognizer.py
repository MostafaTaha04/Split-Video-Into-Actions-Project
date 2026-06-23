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
    """

    def __init__(self):
        self.cpu_terms = {"cpu", "processor", "computer processor"}
        self.real_socket_terms = {"cpu socket", "socket retention lever", "socket retention bracket"}
        self.cooler_terms = {
            "cpu cooler", "computer cpu cooler", "air cooler", "cooling fan",
            "cooler fan", "computer fan", "fan blades", "heatsink",
        }
        self.cooler_mount_terms = {
            "mounting clip", "retention clip", "mounting bracket",
            "cooler bracket", "bracket", "clip",
        }
        self.screw_terms = {"screw", "screwdriver"}
        self.ram_terms = {"ram", "ram stick", "memory module", "ram slot"}
        self.ssd_terms = {"ssd", "m.2 ssd", "nvme ssd"}
        self.cable_terms = {"cable", "fan cable", "power cable", "connector", "plug", "header"}
        self.thermal_terms = {"thermal paste"}
        self.roi_terms = {
            "motherboard workspace", "motherboard_workspace",
            "cpu socket region", "cpu_socket_region",
            "active motion region", "active_motion_region",
        }

    @staticmethod
    def _norm(items: Iterable[str]) -> Set[str]:
        return {
            str(x).strip().lower().replace("_", " ")
            for x in items
            if str(x).strip()
        }

    @staticmethod
    def _has(items: Set[str], terms: Set[str]) -> bool:
        terms_norm = {t.strip().lower().replace("_", " ") for t in terms}
        return any(term in items for term in terms_norm)

    def describe_segment(
        self,
        segment,
        total_duration: float,
        segment_index: int,
        num_segments: int,
    ) -> ActivityDecision:
        real_objects = self._norm(getattr(segment, "real_objects_used", []))
        rois = self._norm(getattr(segment, "heuristic_regions", []))

        interactions = self._norm(getattr(segment, "interaction_types", []))
        motion = float(getattr(segment, "avg_motion_energy", 0.0))
        activity = float(getattr(segment, "avg_activity_level", 0.0))

        # Hand-motion summary (drives labels when no object is detected).
        grip = float(getattr(segment, "grip_ratio", 0.0))
        hand_vel = float(getattr(segment, "avg_hand_velocity", 0.0))
        two_hand = float(getattr(segment, "two_hand_ratio", 0.0))
        curvature = float(getattr(segment, "avg_curvature", 0.0))
        hands_ratio = float(getattr(segment, "hands_present_ratio", 0.0))

        has_interaction = bool(interactions - {"none"})
        high_motion = motion >= 3.0 or activity >= 0.38
        medium_motion = motion >= 1.5 or activity >= 0.25
        low_motion = motion < 1.5 and activity < 0.25

        has_cpu = self._has(real_objects, self.cpu_terms)
        has_real_socket = self._has(real_objects, self.real_socket_terms)
        has_cooler = self._has(real_objects, self.cooler_terms)
        has_cooler_mount = self._has(real_objects, self.cooler_mount_terms)
        has_screw = self._has(real_objects, self.screw_terms)
        has_ram = self._has(real_objects, self.ram_terms)
        has_ssd = self._has(real_objects, self.ssd_terms)
        has_cable = self._has(real_objects, self.cable_terms)
        has_thermal = self._has(real_objects, self.thermal_terms)

        has_rois = bool(rois)

        # 1. Cooler / fan activity
        if has_cooler and has_screw and high_motion:
            return ActivityDecision("CPU cooler screw operation", "cooler/fan and screw-related object detected with strong motion", 0.88)
        if has_cooler and has_cooler_mount and has_interaction:
            return ActivityDecision("Attach or lock cooler mounting mechanism", "cooler and mounting clip/bracket detected with hand interaction", 0.88)
        if has_cooler and has_interaction and high_motion:
            return ActivityDecision("CPU cooler installation", "cooler/fan detected with hand interaction and motion", 0.86)
        if has_cooler and medium_motion:
            return ActivityDecision("Position or align CPU cooler", "cooler/fan detected with motion", 0.80)
        if has_cooler:
            return ActivityDecision("CPU cooler inspection", "cooler/fan detected with low motion", 0.72)

        # 2. Cable activity
        if has_cable and has_interaction and medium_motion:
            return ActivityDecision("Cable connection", "cable/connector detected with hand interaction", 0.84)
        if has_cable and medium_motion:
            return ActivityDecision("Cable positioning", "cable/connector detected with motion", 0.76)
        if has_cable:
            return ActivityDecision("Cable inspection", "cable/connector detected with low motion", 0.68)

        # 3. RAM activity
        if has_ram and has_interaction and high_motion:
            return ActivityDecision("Press RAM module into slot", "RAM detected with hand interaction and strong motion", 0.86)
        if has_ram and medium_motion:
            return ActivityDecision("Align or insert RAM module", "RAM detected with motion", 0.80)
        if has_ram:
            return ActivityDecision("RAM module inspection", "RAM detected with low motion", 0.70)

        # 4. SSD activity
        if has_ssd and has_screw:
            return ActivityDecision("Secure M.2 SSD", "SSD and screw/screwdriver detected", 0.86)
        if has_ssd and has_interaction and medium_motion:
            return ActivityDecision("Insert M.2 SSD", "SSD detected with hand interaction", 0.84)
        if has_ssd:
            return ActivityDecision("M.2 SSD positioning", "SSD detected", 0.72)

        # 5. CPU/socket activity
        if has_cpu and has_real_socket and has_interaction and medium_motion:
            return ActivityDecision("CPU/socket manipulation", "CPU and real socket-related object detected with interaction", 0.86)
        if has_cpu and has_real_socket:
            return ActivityDecision("CPU/socket alignment or seating", "CPU and real socket-related object detected", 0.80)
        if has_cpu and medium_motion:
            return ActivityDecision("CPU handling", "CPU/processor detected with motion", 0.76)
        if has_cpu:
            return ActivityDecision("CPU inspection", "CPU/processor detected with low motion", 0.68)

        # 6. Screw/tool generic activity
        if has_screw and has_interaction and high_motion:
            return ActivityDecision("Screwdriver/screw operation", "screw or screwdriver detected with hand interaction", 0.84)
        if has_screw:
            return ActivityDecision("Prepare screw/screwdriver", "screw or screwdriver detected", 0.72)

        # 7. Thermal activity
        if has_thermal and high_motion:
            return ActivityDecision("Thermal paste / heatsink preparation", "thermal paste detected with motion", 0.76)
        if has_thermal:
            return ActivityDecision("Thermal paste inspection", "thermal paste detected", 0.66)

        # 8. No real objects detected: describe the HAND MOTION PHASE.
        gripping = grip >= 0.45
        twisting = curvature >= 0.6
        two_handed = two_hand >= 0.45
        fast_hand = hand_vel >= 12.0 or high_motion
        some_hand = hands_ratio >= 0.25

        if some_hand and gripping and twisting:
            return ActivityDecision("Rotating / tightening motion", "hand gripping with curved/twisting trajectory (screw- or twist-like)", 0.66)
        if some_hand and gripping and two_handed:
            return ActivityDecision("Two-handed part manipulation", "both hands present with sustained grip", 0.64)
        if some_hand and gripping and (medium_motion or fast_hand):
            return ActivityDecision("Grip and move a component", "sustained hand grip with motion", 0.62)
        if some_hand and gripping:
            return ActivityDecision("Hold part / fine adjustment", "sustained hand grip with low motion", 0.58)
        if some_hand and two_handed and (medium_motion or fast_hand):
            return ActivityDecision("Two-handed positioning", "both hands present with motion, no firm grip", 0.58)
        if some_hand and fast_hand:
            return ActivityDecision("Reach / reposition hand", "fast hand movement without sustained grip", 0.56)
        if some_hand and medium_motion:
            return ActivityDecision("Workspace adjustment", "moderate hand motion in the workspace", 0.54)
        if not some_hand and low_motion:
            if segment_index == num_segments - 1:
                return ActivityDecision("Final inspection or pause", "last segment, no hands in view and low motion", 0.52)
            return ActivityDecision("Inspection / pause (hands out of view)", "no hands in view and low motion", 0.50)
        if has_rois and low_motion:
            return ActivityDecision("Workspace inspection", "hands resting in workspace with low motion", 0.50)

        # 9. Complete generic fallback
        if high_motion:
            return ActivityDecision("Unspecified hand activity", "motion detected but no reliable object detected", 0.48)
        if medium_motion:
            return ActivityDecision("Unspecified adjustment", "moderate motion but no reliable object detected", 0.46)
        return ActivityDecision("Inspection or pause", "low motion and no reliable object detected", 0.44)