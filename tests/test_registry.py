"""
Tests for the clip registry, in particular the separation between
human-annotated clips (clips.json) and AI-drafted ones (clips_ai_draft.json).

The guarantee under test: an annotation that no person has reviewed can never
reach tuning, the leave-one-clip-out study or the headline held-out number,
because dev_clips / heldout_clips / clean_names read only clips.json.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import clip_registry as cr  # noqa: E402


def _entry(name, gt, **extra):
    return {"name": name, "video": f"{name}.mp4", "results_dir": f"results_{name}",
            "ground_truth": gt, "fps": 10.0, "footage": "edited", "split": "heldout", **extra}


def _gt(path, annotation=None):
    data = {"video": "x.mp4", "annotator": "A", "steps": [
        {"id": 0, "start": 0.0, "end": 4.3, "label": "a"},
        {"id": 1, "start": 4.3, "end": 9.0, "label": "b"}]}
    if annotation:
        data["annotation"] = annotation
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


class TestAiDraftSeparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name
        _gt(os.path.join(self.src, "gt_dev.json"))
        _gt(os.path.join(self.src, "gt_held.json"))
        _gt(os.path.join(self.src, "gt_ai.json"), annotation="ai_draft")
        human = [dict(_entry("dev1", "gt_dev.json"), split="dev"),
                 _entry("held1", "gt_held.json")]
        self._write(cr.REGISTRY_FILE, human)
        self._write(cr.AI_REGISTRY_FILE, [_entry("ai1", "gt_ai.json", annotation="ai_draft")])

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, fname, clips):
        with open(os.path.join(self.src, fname), "w", encoding="utf-8") as fh:
            json.dump({"clips": clips}, fh)

    def test_ai_clips_never_in_human_sets(self):
        self.assertEqual(set(cr.dev_clips(self.src)), {"dev1"})
        self.assertEqual(set(cr.heldout_clips(self.src)), {"held1"})
        self.assertNotIn("ai1", cr.clean_names(self.src, split=None))
        self.assertEqual(set(cr.ai_draft_clips(self.src)), {"ai1"})

    def test_missing_ai_registry_means_none(self):
        os.remove(os.path.join(self.src, cr.AI_REGISTRY_FILE))
        self.assertEqual(cr.load_ai_registry(self.src), [])
        self.assertEqual(cr.ai_draft_clips(self.src), {})

    def test_ai_entry_must_be_marked(self):
        self._write(cr.AI_REGISTRY_FILE, [_entry("ai1", "gt_ai.json")])
        with self.assertRaises(SystemExit):
            cr.load_ai_registry(self.src)

    def test_clip_cannot_be_in_both_registries(self):
        self._write(cr.AI_REGISTRY_FILE, [_entry("held1", "gt_held.json", annotation="ai_draft")])
        with self.assertRaises(SystemExit):
            cr.load_ai_registry(self.src)

    def test_add_clip_routes_by_annotation(self):
        _gt(os.path.join(self.src, "gt_ai2.json"), annotation="ai_draft")
        cr.add_clip(_entry("ai2", "gt_ai2.json", annotation="ai_draft"), self.src)
        self.assertIn("ai2", {c["name"] for c in cr.load_ai_registry(self.src)})
        self.assertNotIn("ai2", {c["name"] for c in cr.load_registry(self.src)})
        with self.assertRaises(SystemExit):        # name already taken (other registry)
            cr.add_clip(_entry("ai2", "gt_ai2.json"), self.src)


class TestPromote(unittest.TestCase):
    """promote_clip.py moves a clip into clips.json only with a human annotation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name
        with open(os.path.join(self.src, cr.REGISTRY_FILE), "w", encoding="utf-8") as fh:
            json.dump({"clips": [dict(_entry("dev1", "gt_dev.json"), split="dev")]}, fh)
        with open(os.path.join(self.src, cr.AI_REGISTRY_FILE), "w", encoding="utf-8") as fh:
            json.dump({"clips": [_entry("ai1", "gt_ai.json", annotation="ai_draft")]}, fh)
        _gt(os.path.join(self.src, "gt_dev.json"))
        _gt(os.path.join(self.src, "gt_ai.json"), annotation="ai_draft")
        human = {"video": "x.mp4", "annotator": "Test Person", "annotation_method": "frames",
                 "steps": [{"id": 0, "start": 0.0, "end": 4.6, "label": "a"},
                           {"id": 1, "start": 4.6, "end": 9.0, "label": "b"}]}
        with open(os.path.join(self.src, "gt_ai_human.json"), "w", encoding="utf-8") as fh:
            json.dump(human, fh)

    def tearDown(self):
        self.tmp.cleanup()

    def test_promote_moves_clip_to_heldout(self):
        from promote_clip import promote
        promote(self.src, "ai1", "gt_ai_human.json")
        human = {c["name"]: c for c in cr.load_registry(self.src)}
        self.assertEqual(human["ai1"]["split"], "heldout")
        self.assertEqual(human["ai1"]["ground_truth"], "gt_ai_human.json")
        self.assertNotIn("annotation", human["ai1"])
        self.assertEqual(cr.load_ai_registry(self.src), [])
        self.assertIn("ai1", cr.heldout_clips(self.src))

    def test_reviewed_draft_is_promoted_with_its_provenance(self):
        from promote_clip import promote
        path = os.path.join(self.src, "gt_ai.json")
        with open(path, encoding="utf-8") as fh:
            gt = json.load(fh)
        gt.update(annotator="Test Person",
                  review_status="human_reviewed",
                  annotation_method="Drafted with an AI tool; checked against the video by Test Person")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(gt, fh)
        promote(self.src, "ai1", "gt_ai.json")
        entry = {c["name"]: c for c in cr.load_registry(self.src)}["ai1"]
        self.assertEqual(entry["annotation_source"], "ai_draft_reviewed")
        self.assertIn("ai1", cr.heldout_clips(self.src))
        self.assertNotIn("ai1", cr.hand_heldout_clips(self.src))

    def test_reviewed_draft_must_name_the_review(self):
        from promote_clip import promote
        path = os.path.join(self.src, "gt_ai.json")
        with open(path, encoding="utf-8") as fh:
            gt = json.load(fh)
        gt.update(annotator="Test Person", review_status="human_reviewed")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(gt, fh)
        with self.assertRaises(SystemExit):
            promote(self.src, "ai1", "gt_ai.json")

    def test_refuses_the_ai_draft_itself(self):
        from promote_clip import promote
        with self.assertRaises(SystemExit):
            promote(self.src, "ai1", "gt_ai.json")
        self.assertEqual([c["name"] for c in cr.load_ai_registry(self.src)], ["ai1"])


class TestRepositoryRegistry(unittest.TestCase):
    """Checks on the real clips.json / clips_ai_draft.json in this repository."""

    def test_ai_registry_files_are_marked_as_drafts(self):
        for c in cr.load_ai_registry(ROOT):
            path = cr.resolve_gt(c["ground_truth"], ROOT)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                gt = json.load(fh)
            self.assertEqual(gt.get("annotation"), "ai_draft", c["name"])
            self.assertIn("review_status", gt, c["name"])

    def test_ai_provenance_is_recorded_in_both_places(self):
        """A clip with AI-drafted ground truth says so in clips.json AND in its
        annotation file, so results can always be split by provenance."""
        for c in cr.load_registry(ROOT):
            path = cr.resolve_gt(c["ground_truth"], ROOT)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                gt = json.load(fh)
            drafted = gt.get("annotation") == "ai_draft"
            self.assertEqual(drafted, not cr.is_hand_annotated(c), c["name"])

    def test_no_ai_annotation_in_human_registry(self):
        for c in cr.load_registry(ROOT):
            path = cr.resolve_gt(c["ground_truth"], ROOT)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                gt = json.load(fh)
            if gt.get("annotation") == "ai_draft":
                self.assertEqual(gt.get("review_status"), "human_reviewed",
                                 f"{c['name']}: an unreviewed AI draft is in clips.json")


if __name__ == "__main__":
    unittest.main()
