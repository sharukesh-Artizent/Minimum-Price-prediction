"""SDD Helper script for extracting briefs and creating review packages."""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = REPO_ROOT / "docs" / "2026-10-09-reserve-price-clm.md"
WORKSPACE = REPO_ROOT / ".superpowers" / "sdd" / "2026-10-09-reserve-price-clm"


def extract_brief(task_num: int) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    out_file = WORKSPACE / f"task-{task_num}-brief.md"

    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    task_pattern = re.compile(rf"^#+\s+Task\s+{task_num}(?:\D|$)")
    next_task_pattern = re.compile(r"^#+\s+Task\s+\d+")
    
    in_fence = False
    in_task = False
    task_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            if task_pattern.match(line):
                in_task = True
            elif next_task_pattern.match(line) and in_task:
                in_task = False
                break
        if in_task:
            task_lines.append(line)

    if not task_lines:
        raise ValueError(f"Task {task_num} not found in {PLAN_FILE}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(task_lines)

    print(f"Wrote {out_file}: {len(task_lines)} lines")
    return out_file


def create_review_package(base_sha: str, head_sha: str, outfile: str = None) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    short_base = subprocess.check_output(["git", "rev-parse", "--short", base_sha], cwd=REPO_ROOT, text=True).strip()
    short_head = subprocess.check_output(["git", "rev-parse", "--short", head_sha], cwd=REPO_ROOT, text=True).strip()
    
    out_path = Path(outfile) if outfile else WORKSPACE / f"review-{short_base}..{short_head}.diff"

    log_out = subprocess.check_output(["git", "log", "--oneline", f"{base_sha}..{head_sha}"], cwd=REPO_ROOT, text=True)
    stat_out = subprocess.check_output(["git", "diff", "--stat", f"{base_sha}..{head_sha}"], cwd=REPO_ROOT, text=True)
    diff_out = subprocess.check_output(["git", "diff", "-U10", f"{base_sha}..{head_sha}"], cwd=REPO_ROOT, text=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Review package: {base_sha}..{head_sha}\n\n")
        f.write("## Commits\n")
        f.write(log_out + "\n")
        f.write("## Files changed\n")
        f.write(stat_out + "\n")
        f.write("## Diff\n")
        f.write(diff_out + "\n")

    print(f"Wrote {out_path}: {len(diff_out)} bytes")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    brief_p = subparsers.add_parser("brief")
    brief_p.add_argument("task", type=int)

    pkg_p = subparsers.add_parser("review-package")
    pkg_p.add_argument("base")
    pkg_p.add_argument("head")
    pkg_p.add_argument("--out", default=None)

    args = parser.parse_args()
    if args.cmd == "brief":
        extract_brief(args.task)
    elif args.cmd == "review-package":
        create_review_package(args.base, args.head, args.out)
    else:
        parser.print_help()
