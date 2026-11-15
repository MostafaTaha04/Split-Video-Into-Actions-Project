"""
check_everything.py
===================
Run every verification step of the project in one go and save the complete
output to one log file, ready to send for review.

    python check_everything.py              # all fast checks + the 20-minute smoke test
    python check_everything.py --no-smoke   # fast checks only (~3 minutes)

Steps, in order (each is described with its expected output in
docs/TEST_COMMANDS.md):

  1. environment versions
  2. lint (ruff) and the unit tests
  3. clip registry (30 clips: 5 hand-annotated, 25 AI-drafted + human-reviewed)
  4. every ground-truth file through the validator
  5. saved run reports have not drifted
  6. extended evaluation (both scorers, standard metrics, edit-cut baseline,
     held-out set)
  7. every over-segmentation fix that was tried
  8. learned scorers vs the hand-designed ones
  9. assembly vs disassembly per step (evaluate_direction.py)
 10. turning vs moving step names (evaluate_activity_names.py)
 11. module ablation table (from the runs already made)
 12. smoke test of the full pipeline on all 30 videos (verify_all.py --quick)

The long end-to-end check (verify_all.py without --quick, ~5-6 h) is separate:
run it overnight and send its table too.

Writes check_everything_log.txt and prints a pass/fail line per step.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

LOG = "check_everything_log.txt"


def steps(src, smoke):
    py = sys.executable
    out = [
        ("environment", [py, "-c",
                         "import sys, cv2, numpy, sklearn, mediapipe, scipy, ultralytics; "
                         "print(sys.version.split()[0], 'cv2', cv2.__version__, 'numpy', numpy.__version__, "
                         "'sklearn', sklearn.__version__, 'mediapipe', mediapipe.__version__, "
                         "'ultralytics', ultralytics.__version__)"]),
        ("lint", [py, "-m", "ruff", "check", "."]),
        ("unit tests", [py, "-m", "unittest", "discover", "-s", "tests"]),
        ("clip registry", [py, "clip_registry.py", src]),
        ("ground-truth validation", [py, os.path.abspath(__file__), "--validate-only"]),
        ("saved reports (no drift)", [py, "refresh_run_reports.py", "--dry-run"]),
        ("extended evaluation", [py, "evaluate_extended.py", "--src", src]),
        ("over-segmentation fixes", [py, "compare_scorer_variants.py", "--src", src]),
        ("learned scorers", [py, "learned_baseline.py", "--src", src]),
        ("assembly vs disassembly", [py, "evaluate_direction.py", "--src", src]),
        ("turning vs moving names", [py, "evaluate_activity_names.py", "--src", src]),
        ("module ablations", [py, "run_ablations.py", "--summarize"]),
    ]
    if smoke:
        out.append(("pipeline smoke test, 30 videos", [py, "verify_all.py", "--quick"]))
    return out


def validate_all(src):
    """Every annotation file referenced by either registry, one line each."""
    from clip_registry import load_ai_registry, load_registry, resolve_gt
    bad = 0
    for c in load_registry(src) + load_ai_registry(src):
        gt = resolve_gt(c["ground_truth"], src)
        r = subprocess.run([sys.executable, "annotate.py", "validate", "--ground-truth", gt],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        bad += r.returncode != 0
        print(f"{c['name']:20s} {last}", flush=True)
    print(f"{bad} file(s) with errors")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="Run every verification step; save one log.")
    ap.add_argument("--src", default=".")
    ap.add_argument("--no-smoke", action="store_true", help="skip the 20-minute pipeline smoke test")
    ap.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.validate_only:
        sys.exit(validate_all(a.src))

    env = dict(os.environ, PYTHONIOENCODING="utf-8", REQUIRE_SAVED_RESULTS="1")
    results = []
    with open(LOG, "w", encoding="utf-8") as log:
        for i, (name, cmd) in enumerate(steps(a.src, not a.no_smoke), 1):
            shown = " ".join(cmd) if len(" ".join(cmd)) < 200 else " ".join(cmd[:2]) + " ..."
            header = f"\n{'#' * 78}\n# STEP {i}: {name}\n# $ {shown}\n{'#' * 78}\n"
            print(header, end="", flush=True)
            log.write(header)
            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", env=env)
            for line in proc.stdout:
                # tqdm progress bars repeat a line hundreds of times; keep the log readable
                if "it/s]" in line or "s/it]" in line:
                    if "100%" not in line:
                        continue
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
            dt = time.time() - t0
            status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
            results.append((i, name, status, dt))
            tail = f"\n# -> {status} in {dt:.0f} s\n"
            print(tail, end="")
            log.write(tail)

        summary = ["", "=" * 78, "SUMMARY"]
        summary += [f"  {i:2d}. {name:34s} {status:16s} {dt:6.0f} s" for i, name, status, dt in results]
        n_ok = sum(s == "PASS" for _, _, s, _ in results)
        summary += [f"  {n_ok}/{len(results)} steps passed", "=" * 78, ""]
        text = "\n".join(summary)
        print(text)
        log.write(text)
    print(f"Full output saved to {LOG} - send that file.")


if __name__ == "__main__":
    main()
