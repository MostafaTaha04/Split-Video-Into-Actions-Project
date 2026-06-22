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

    Important design decision:
    - Real object detections drive object-specific labels.
    - Heuristic ROIs only produce generic workspace labels.
    - We do NOT call something "CPU/socket" unless a real CPU/socket object is detected.
    """

    def __init__(self):
        self.cpu_terms = {
            "cpu",
            "processor",
            "computer processor",
        }

        self.real_socket_terms = {
            "cpu socket",
            "socket retention lever",
            "socket retention bracket",
        }

        self.cooler_terms = {
            "cpu cooler",
            "computer cpu cooler",
            "air cooler",
            "cooling fan",
            "cooler fan",
            "computer fan",
            "fan blades",
            "heatsink",
        }

        self.cooler_mount_terms = {
            "mounting clip",
            "retention clip",
            "mounting bracket",
            "cooler bracket",
            "bracket",
            "clip",
        }

        self.screw_terms = {
            "screw",
            "screwdriver",
        }

        self.ram_terms = {
            "ram",
            "ram stick",
            "memory module",
            "ram slot",
        }

        self.ssd_terms = {
            "ssd",
            "m.2 ssd",
            "nvme ssd",
        }

        self.cable_terms = {
            "cable",
            "fan cable",
            "power cable",
            "connector",
            "plug",
            "header",
        }

        self.thermal_terms = {
            "thermal paste",
        }

        self.roi_terms = {
            "motherboard workspace",
            "motherboard_workspace",
            "cpu socket region",
            "cpu_socket_region",
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
        """
        Generate a descriptive activity label.

        This is not the main evaluated task. The main task is temporal segmentation.
        """
        real_objects = self._norm(getattr(segment, "real_objects_used", []))
        rois = self._norm(getattr(segment, "heuristic_regions", []))

        interactions = self._norm(getattr(segment, "interaction_types", []))
        motion = float(getattr(segment, "avg_motion_energy", 0.0))
        activity = float(getattr(segment, "avg_activity_level", 0.0))

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

        # ------------------------------------------------------------
        # 1. Cooler / fan activity
        # ------------------------------------------------------------
        if has_cooler and has_screw and high_motion:
            return ActivityDecision(
                "CPU cooler screw operation",
                "cooler/fan and screw-related object detected with strong motion",
                0.88,
            )

        if has_cooler and has_cooler_mount and has_interaction:
            return ActivityDecision(
                "Attach or lock cooler mounting mechanism",
                "cooler and mounting clip/bracket detected with hand interaction",
                0.88,
            )

        if has_cooler and has_interaction and high_motion:
            return ActivityDecision(
                "CPU cooler installation",
                "cooler/fan detected with hand interaction and motion",
                0.86,
            )

        if has_cooler and medium_motion:
            return ActivityDecision(
                "Position or align CPU cooler",
                "cooler/fan detected with motion",
                0.80,
            )

        if has_cooler:
            return ActivityDecision(
                "CPU cooler inspection",
                "cooler/fan detected with low motion",
                0.72,
            )

        # ------------------------------------------------------------
        # 2. Cable activity
        # ------------------------------------------------------------
        if has_cable and has_interaction and medium_motion:
            return ActivityDecision(
                "Cable connection",
                "cable/connector detected with hand interaction",
                0.84,
            )

        if has_cable and medium_motion:
            return ActivityDecision(
                "Cable positioning",
                "cable/connector detected with motion",
                0.76,
            )

        if has_cable:
            return ActivityDecision(
                "Cable inspection",
                "cable/connector detected with low motion",
                0.68,
            )

        # ------------------------------------------------------------
        # 3. RAM activity
        # ------------------------------------------------------------
        if has_ram and has_interaction and high_motion:
            return ActivityDecision(
                "Press RAM module into slot",
                "RAM detected with hand interaction and strong motion",
                0.86,
            )

        if has_ram and medium_motion:
            return ActivityDecision(
                "Align or insert RAM module",
                "RAM detected with motion",
                0.80,
            )

        if has_ram:
            return ActivityDecision(
                "RAM module inspection",
                "RAM detected with low motion",
                0.70,
            )

        # ------------------------------------------------------------
        # 4. SSD activity
        # ------------------------------------------------------------
        if has_ssd and has_screw:
            return ActivityDecision(
                "Secure M.2 SSD",
                "SSD and screw/screwdriver detected",
                0.86,
            )

        if has_ssd and has_interaction and medium_motion:
            return ActivityDecision(
                "Insert M.2 SSD",
                "SSD detected with hand interaction",
                0.84,
            )

        if has_ssd:
            return ActivityDecision(
                "M.2 SSD positioning",
                "SSD detected",
                0.72,
            )

        # ------------------------------------------------------------
        # 5. CPU/socket activity
        # Only use CPU/socket wording when real CPU/socket objects are detected.
        # Do not use cpu_socket_region ROI for these labels.
        # ------------------------------------------------------------
        if has_cpu and has_real_socket and has_interaction and medium_motion:
            return ActivityDecision(
                "CPU/socket manipulation",
                "CPU and real socket-related object detected with interaction",
                0.86,
            )

        if has_cpu and has_real_socket:
            return ActivityDecision(
                "CPU/socket alignment or seating",
                "CPU and real socket-related object detected",
                0.80,
            )

        if has_cpu and medium_motion:
            return ActivityDecision(
                "CPU handling",
                "CPU/processor detected with motion",
                0.76,
            )

        if has_cpu:
            return ActivityDecision(
                "CPU inspection",
                "CPU/processor detected with low motion",
                0.68,
            )

        # ------------------------------------------------------------
        # 6. Screw/tool generic activity
        # ------------------------------------------------------------
        if has_screw and has_interaction and high_motion:
            return ActivityDecision(
                "Screwdriver/screw operation",
                "screw or screwdriver detected with hand interaction",
                0.84,
            )

        if has_screw:
            return ActivityDecision(
                "Prepare screw/screwdriver",
                "screw or screwdriver detected",
                0.72,
            )

        # ------------------------------------------------------------
        # 7. Thermal activity
        # ------------------------------------------------------------
        if has_thermal and high_motion:
            return ActivityDecision(
                "Thermal paste / heatsink preparation",
                "thermal paste detected with motion",
                0.76,
            )

        if has_thermal:
            return ActivityDecision(
                "Thermal paste inspection",
                "thermal paste detected",
                0.66,
            )

        # ------------------------------------------------------------
        # 8. ROI-only fallback
        # This is the most important fix for the cooling fan video.
        # If real objects are not detected, do not pretend it is socket work.
        # ------------------------------------------------------------
        if has_rois and has_interaction and high_motion:
            return ActivityDecision(
                "Workspace manipulation",
                "only heuristic ROIs detected, with hand interaction and strong motion",
                0.62,
            )

        if has_rois and has_interaction:
            return ActivityDecision(
                "Workspace interaction",
                "only heuristic ROIs detected, with hand interaction",
                0.58,
            )

        if has_rois and high_motion:
            return ActivityDecision(
                "Workspace manipulation",
                "only heuristic ROIs detected, with strong motion",
                0.58,
            )

        if has_rois and medium_motion:
            return ActivityDecision(
                "Workspace adjustment",
                "only heuristic ROIs detected, with moderate motion",
                0.54,
            )

        if has_rois and low_motion:
            if segment_index == num_segments - 1:
                return ActivityDecision(
                    "Final inspection or pause",
                    "last segment with low motion and only heuristic ROIs",
                    0.52,
                )

            return ActivityDecision(
                "Workspace inspection",
                "only heuristic ROIs detected with low motion",
                0.50,
            )

        # ------------------------------------------------------------
        # 9. Complete generic fallback
        # ------------------------------------------------------------
        if high_motion:
            return ActivityDecision(
                "Unspecified hand activity",
                "motion detected but no reliable object detected",
                0.48,
            )

        if medium_motion:
            return ActivityDecision(
                "Unspecified adjustment",
                "moderate motion but no reliable object detected",
                0.46,
            )

        return ActivityDecision(
            "Inspection or pause",
            "low motion and no reliable object detected",
            0.44,
        )