from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STN_AFFERENTS = SCRIPT_DIR / "STN.py"
OUTPUT_ROOT = Path("/home/dtorbin/Downloads/articles") / (
    "succession_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

# Edit these defaults once, then override per run only where needed.
DEFAULT_PLOTS = {
    "trace": True,
    "spta": True,
    "mpta": True,
    "spectrogram": True,
    "hilbert_amp": True,
    "post_rate_distribution": True,
    "plv_histogram": False,
    "recruitment_dynamics": True,
    "geometry": False,
    "paper_plot": True,
}

DEFAULT_ANALYSIS = {
    "dbs_rate_edge_window_ms": 100.0,
    "paper_trace_pre_ms": 40.0,
    "paper_trace_post_ms": 160.0,
}

# Overnight plan. Add/remove/edit runs here.
# Required per run: name, pw_ms, amp_uA, freq_hz, tstop_ms, dbs_start_ms,
# dbs_stop_ms, n_cells. Optional: seed, plots, analysis, extra_args.
RUN_PLAN = [
    {
        "name": "run01_135Hz_25uA_100us",
        "pw_ms": 0.100,
        "amp_uA": 25.0,
        "freq_hz": 135.0,
        "tstop_ms": 10000.0,
        "dbs_start_ms": 2000.0,
        "dbs_stop_ms": 8000.0,
        "n_cells": 10,
        "plots": DEFAULT_PLOTS,
    },
    {
        "name": "run02_135Hz_TEED_200us",
        "pw_ms": 0.200,
        "amp_uA": 17,
        "freq_hz": 135.0,
        "tstop_ms": 10000.0,
        "dbs_start_ms": 2000.0,
        "dbs_stop_ms": 8000.0,
        "n_cells": 10,
        "plots": DEFAULT_PLOTS,
    },
    {
        "name": "run03_135Hz_TEED_60us",
        "pw_ms": 0.060,
        "amp_uA": 32.0,
        "freq_hz": 135.0,
        "tstop_ms": 10000.0,
        "dbs_start_ms": 2000.0,
        "dbs_stop_ms": 8000.0,
        "n_cells": 10,
        "plots": DEFAULT_PLOTS,
    },
     {
        "name": "run04_40Hz_TEED_100us",
        "pw_ms": 0.100,
        "amp_uA": 45.0,
        "freq_hz": 40.0,
        "tstop_ms": 10000.0,
        "dbs_start_ms": 2000.0,
        "dbs_stop_ms": 8000.0,
        "n_cells": 10,
        "plots": DEFAULT_PLOTS,
    },
    {
        "name": "run05_100_135Hz_TEED_100us",
        "pw_ms": 0.100,
        "amp_uA": 25.0,
        "freq_hz": 135.0,
        "tstop_ms": 1000.0,
        "dbs_start_ms": 200.0,
        "dbs_stop_ms": 800.0,
        "n_cells": 100,
        "plots": DEFAULT_PLOTS,
    },
]


PLOT_FLAGS = {
    "trace": "--plot",
    "spta": "--plot-spta",
    "mpta": "--plot-mpta",
    "spectrogram": "--plot-spectrogram",
    "hilbert_amp": "--plot-hilbert-amp",
    "post_rate_distribution": "--plot-post-rate-distribution",
    "plv_histogram": "--plot-plv-histogram",
    "recruitment_dynamics": "--plot-recruitment-dynamics",
    "geometry": "--plot-geometry",
    "paper_plot": "--paper-plot",
}


def bool_flag(flag: str, enabled: bool) -> str:
    if enabled:
        return flag
    return "--no-" + flag.removeprefix("--")


def run_dir_name(index: int, name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return f"{index:02d}_{safe.strip('._-') or 'run'}"


def build_command(run: dict, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(STN_AFFERENTS),
        "--n-cells",
        str(int(run["n_cells"])),
        "--tstop",
        str(float(run["tstop_ms"])),
        "--dbs-start",
        str(float(run["dbs_start_ms"])),
        "--dbs-stop-ms",
        str(float(run["dbs_stop_ms"])),
        "--dbs-freq",
        str(float(run["freq_hz"])),
        "--dbs-pw",
        str(float(run["pw_ms"])),
        "--dbs-amp-uA",
        str(float(run["amp_uA"])),
        "--save-figure",
        "--save-figure-dir",
        str(run_dir),
        "--print-summary",
    ]

    analysis = dict(DEFAULT_ANALYSIS)
    analysis.update(run.get("analysis", {}))
    cmd.extend(
        [
            "--dbs-rate-edge-window-ms",
            str(float(analysis["dbs_rate_edge_window_ms"])),
            "--paper-trace-pre-ms",
            str(float(analysis["paper_trace_pre_ms"])),
            "--paper-trace-post-ms",
            str(float(analysis["paper_trace_post_ms"])),
        ]
    )

    if "seed" in run:
        cmd.extend(["--seed", str(int(run["seed"]))])

    plots = dict(DEFAULT_PLOTS)
    plots.update(run.get("plots", {}))
    for key, flag in PLOT_FLAGS.items():
        cmd.append(bool_flag(flag, bool(plots.get(key, False))))

    cmd.extend(str(item) for item in run.get("extra_args", []))
    return cmd


def stream_process(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return int(process.wait())


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Keeps overnight runs from blocking on GUI windows while still saving figures.
    env.setdefault("MPLBACKEND", "Agg")

    plan_path = OUTPUT_ROOT / "run_plan.json"
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(RUN_PLAN, handle, indent=2)
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Saved plan: {plan_path}")

    failures = []
    for index, run in enumerate(RUN_PLAN, start=1):
        name = str(run["name"])
        run_dir = OUTPUT_ROOT / run_dir_name(index, name)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(run, run_dir)

        with open(run_dir / "command.txt", "w", encoding="utf-8") as handle:
            handle.write(" ".join(cmd) + "\n")
        with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(run, handle, indent=2)

        print("\n" + "=" * 88)
        print(f"Starting {index}/{len(RUN_PLAN)}: {name}")
        print(f"Figures/logs: {run_dir}")
        print("=" * 88)

        code = stream_process(cmd, run_dir / "run.log", env)
        if code != 0:
            failures.append({"name": name, "exit_code": code, "run_dir": str(run_dir)})
            print(f"[FAILED] {name} exited with code {code}")
        else:
            print(f"[DONE] {name}")

    summary = {
        "output_root": str(OUTPUT_ROOT),
        "n_runs": len(RUN_PLAN),
        "n_failed": len(failures),
        "failures": failures,
    }
    with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 88)
    if failures:
        print(f"Finished with {len(failures)} failed run(s). See {OUTPUT_ROOT / 'summary.json'}")
        raise SystemExit(1)
    print("All succession runs finished cleanly.")


if __name__ == "__main__":
    main()
