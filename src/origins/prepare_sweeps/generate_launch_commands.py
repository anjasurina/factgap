"""
Generate shell commands for launching an sweep of experiments from a YAML configuration
file and save them to a separate shell script (default: ``all_jobs.sh``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_launch_command(
    config_file: Path,
    num_gpus: int,
    accelerate_config_path: Path,
    mixed_precision: str | None = None,
    gradient_accumulation_steps: int | None = None,
    run_command: str | None = None,
) -> str:
    """Construct the full shell command for launching the run.

    Parameters
    ----------
    config_file: Path
        Absolute or relative path to the YAML configuration file.
    num_gpus: int
        Number of GPUs to use. ``1`` runs without ``accelerate``; ``>1`` uses it.
    mixed_precision: str | None
        Optional override for ``accelerate --mixed_precision`` (e.g. "bf16", "fp16", "no").
    gradient_accumulation_steps: int | None
        Optional override for ``accelerate --gradient_accumulation_steps``.
    run_command: str | None
        If provided, use this command verbatim instead of building an
        ``accelerate launch …`` invocation.  Hydra ``--config-path`` and
        ``--config-name`` arguments are still appended.

    Returns
    -------
    str
        A fully-formed shell command ready to be executed.
    """

    config_path = config_file.parent.as_posix()
    config_name = config_file.stem  # filename without extension

    if run_command is not None:
        base_cmd = run_command
    else:
        if not accelerate_config_path.is_file():
            sys.exit(f"Error: Accelerate config file '{accelerate_config_path}' not found.")

        base_cmd_parts = [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config_path.as_posix(),
            "--num_processes",
            str(num_gpus),
        ]

        if mixed_precision is not None:
            base_cmd_parts.extend(["--mixed_precision", mixed_precision])

        if gradient_accumulation_steps is not None:
            base_cmd_parts.extend(["--gradient_accumulation_steps", str(gradient_accumulation_steps)])

        base_cmd_parts.append("src/origins/main.py")
        base_cmd = " ".join(base_cmd_parts)

    config_path_to_use_final_command = f"../../{config_path}"
    full_cmd = (
        base_cmd
        + f" --config-path {config_path_to_use_final_command} --config-name {config_name}"
    )

    return full_cmd


def append_to_output_file(command: str, output_file: Path) -> None:
    """Append *command* to *output_file* **iff** it is not already present.

    A command is considered to be present if a line that matches it *verbatim*
    (ignoring leading/trailing whitespace) already exists in the file.  If the
    file does not yet exist, it will be created together with a small shebang
    header.
    """

    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Read existing commands (if the file exists)
    # ------------------------------------------------------------------
    existing_commands: set[str] = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            existing_commands = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }

    # Skip if the command is already present
    if command.strip() in existing_commands:
        return

    # ------------------------------------------------------------------
    # 2) Append the new command
    # ------------------------------------------------------------------
    output_exists = output_file.exists()
    with output_file.open("a", encoding="utf-8") as f:
        if not output_exists:
            # Write shebang & header for new files
            f.write("#!/bin/bash\n\n")
        f.write(command + "\n")

    # Make the output file executable for convenience
    if not os.access(output_file, os.X_OK):
        output_file.chmod(output_file.stat().st_mode | 0o111)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_configs_path", default="experiments/lora", help="Path to pregenerated sweep configs.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--output-file",
        default="all_jobs.sh",
        help=f"File to which the launch command will be appended.",
    )
    parser.add_argument("--accelerate_config_path", default="accelerate_configs/deepspeed.yaml", help="Path to the accelerate config file.")

    # ------------------------------------------------------------------
    # Optional overrides for Accelerate
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        help="Override mixed precision setting passed to Accelerate (default: use value from config file).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Override gradient accumulation steps passed to Accelerate (default: use value from config file).",
    )
    parser.add_argument(
        "--run_command",
        default=None,
        help="Custom run command to use instead of the default accelerate launch invocation.",
    )

    args = parser.parse_args()

    experiments_dir = Path(args.sweep_configs_path)
    accelerate_config_path = Path(args.accelerate_config_path)

    # Validate GPU count
    if args.num_gpus < 1:
        sys.exit("Error: <num_gpus> must be at least 1.")

    # Resolve the path to the output script once so we can query it up front
    output_path = experiments_dir / args.output_file

    # Inform the user when we are reusing an existing launch script
    if output_path.exists():
        print(f"[generate_launch_commands] Detected existing launch script at '{output_path}'. "
              "New commands will be appended only if they are not already present.")

    # Read all the yaml files in the configs folder
    config_files = [f for f in experiments_dir.glob("*.yaml") if f.is_file()]
    for config_file in config_files:
        command = build_launch_command(
            config_file=config_file,
            num_gpus=args.num_gpus,
            accelerate_config_path=accelerate_config_path,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            run_command=args.run_command,
        )

        if output_path.exists():
            # Check inside append_to_output_file, but we can early-skip printing to avoid clutter
            with output_path.open("r", encoding="utf-8") as f:
                existing = {line.strip() for line in f}
            if command.strip() in existing:
                print(f"[generate_launch_commands] Skipping duplicate command for '{config_file.name}'.")
                continue

        print(f"Generated command:\n{command}\n")
        append_to_output_file(command, output_path)


if __name__ == "__main__":
    main() 
