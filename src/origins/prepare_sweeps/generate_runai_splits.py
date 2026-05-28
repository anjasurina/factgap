"""
Extended version of the original *split_launch_commands.py* that **incrementally**
updates a directory of ``job_XX.sh`` files:

*  Reads *all* launch commands from ``--input-file``.
*  Detects existing job scripts in ``--output-dir`` and gathers the commands
   already scheduled there.
*  For **new** commands that are not yet present in *any* job file:
    - Keep *all* existing job_XX.sh files **immutable** – they are never
      modified, even if they contain zero commands.
    - Create **new** job scripts (continuing the numerical index) that contain
      the missing commands, grouped so that each new job roughly matches the
      number of commands of the first non-empty existing job file (or one
      command per job if all existing files are empty).

This makes it convenient to re-run the script after expanding the sweep without
duplicating work or manually renumbering job files.

Example
-------
    # Create four job files from experiments/lora/all_jobs.sh
    python split_launch_commands.py \
        --input-file experiments/lora/all_jobs.sh \
        --num-jobs 4 \
        --output-dir experiments/lora/jobs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import re
from typing import List, Dict, Tuple


def read_commands(src: Path) -> List[str]:
    """Return the list of non-empty, non-comment, non-shebang lines.

    Whitespace is stripped from each line to simplify duplicate checks.
    """
    with src.open(encoding="utf-8") as f:
        return [
            line.rstrip()
            for line in f
            if line.strip()                                   # skip blank
            and not line.lstrip().startswith("#")             # skip comments
            and not line.startswith("#!")                     # skip she-bang
        ]


def even_chunks(items: List[str], k: int) -> List[List[str]]:
    """Split *items* into *k* chunks whose sizes differ by at most 1."""
    if k <= 0:
        return [items]
    n = len(items)
    base, extra = divmod(n, k)
    chunks = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def write_job_file(path: Path, commands: List[str]) -> None:
    """(Over)write *path* with ``#!/bin/bash`` header and *commands*."""
    with path.open("w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")
    os.chmod(path, 0o755)


# -----------------------------------------------------------------------------
# Helpers for incremental writing
# -----------------------------------------------------------------------------


JOB_RE = re.compile(r"job_(\d+)\.sh$")


def list_existing_jobs(out_dir: Path) -> List[Tuple[int, Path]]:
    """Return ``[(idx, path), …]`` sorted by *idx* ascending for existing jobs."""
    jobs = []
    for p in out_dir.glob("job_*.sh"):
        m = JOB_RE.match(p.name)
        if m:
            jobs.append((int(m.group(1)), p))
    jobs.sort(key=lambda t: t[0])
    return jobs


def gather_existing_commands(jobs: List[Tuple[int, Path]]) -> Tuple[Dict[int, List[str]], List[str]]:
    """Return mapping ``idx -> cmds`` and *flat* list of all cmds."""
    mapping: Dict[int, List[str]] = {}
    all_cmds: List[str] = []
    for idx, path in jobs:
        cmds = read_commands(path)
        mapping[idx] = cmds
        all_cmds.extend(cmds)
    return mapping, all_cmds


def main() -> None:
    p = argparse.ArgumentParser(description="Split all_jobs.sh into N job files.")
    p.add_argument(
        "--input-file",
        default="experiments/lora/all_jobs.sh",
        help="Shell script that contains the full list of launch commands.",
    )
    p.add_argument(
        "--num-jobs",
        type=int,
        default=10,
        help="*Initial* number of job files to create if none exist. Ignored when job files are already present except when new files need to be created.",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/lora/runai_splits",
        help="Directory where the job_XX.sh files will be written.",
    )
    args = p.parse_args()

    all_commands = read_commands(Path(args.input_file))
    if not all_commands:
        sys.exit(f"No commands found in {args.input_file!s}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Gather existing job scripts (if any) and commands they hold
    # ------------------------------------------------------------------
    jobs = list_existing_jobs(out_dir)
    job_cmds_map, scheduled_cmds_flat = gather_existing_commands(jobs)

    scheduled_set = set(scheduled_cmds_flat)
    new_commands = [cmd for cmd in all_commands if cmd not in scheduled_set]

    if not jobs:
        # --------------------------------------------
        # No existing job scripts – fall back to old behaviour
        # --------------------------------------------
        if args.num_jobs < 1:
            sys.exit("--num-jobs must be ≥ 1")

        chunks = even_chunks(all_commands, args.num_jobs)
        for idx, chunk in enumerate(chunks, 1):
            write_job_file(out_dir / f"job_{idx:02d}.sh", chunk)
        print(
            f"Wrote {len(chunks)} job file(s) containing {len(all_commands)} total command(s) to {out_dir}"
        )
        return

    # ------------------------------------------------------------------
    # 2) Incremental update
    # ------------------------------------------------------------------

    if not new_commands:
        print("All commands are already present – nothing to do.")
        return

    # Determine target capacity per job from the FIRST *non-empty* job file.
    # If all existing jobs are empty, we default to a capacity of 1 so that at
    # least one command is scheduled per newly created job script.
    cap_per_job = next(
        (len(cmds) for idx, cmds in job_cmds_map.items() if cmds),
        1,  # fallback when all jobs are empty
    )

    print(
        f"Found {len(jobs)} existing job file(s) with ~{cap_per_job} command(s) each."
    )

    # -----------------------------------------------------
    # 2.1  Create NEW job files for *all* missing commands
    # -----------------------------------------------------
    # We NEVER modify existing job_XX.sh files.  Instead, pack the new
    # commands into freshly created job scripts that continue the numerical
    # sequence.

    # Decide how many new jobs we need given the desired capacity per job.
    num_new_jobs = (len(new_commands) + cap_per_job - 1) // cap_per_job

    # Determine the starting index for the new job files.  We simply continue
    # counting after the highest existing index to avoid touching any files
    # already present on disk (even if some indices are missing because the
    # corresponding files were deleted manually).
    last_idx = max((idx for idx, _ in jobs), default=0)

    for i in range(1, num_new_jobs + 1):
        idx = last_idx + i
        chunk = new_commands[:cap_per_job]
        new_commands = new_commands[cap_per_job:]

        write_job_file(out_dir / f"job_{idx:02d}.sh", chunk)
        print(f"Created job_{idx:02d}.sh with {len(chunk)} command(s).")

    print("✓ Job scripts updated.")


if __name__ == "__main__":
    main()