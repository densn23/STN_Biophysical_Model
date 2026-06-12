from __future__ import annotations

import argparse
import csv
import ctypes
import os
import pickle
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

SYSTEM_LIBSTDCPP = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
if SYSTEM_LIBSTDCPP.exists():
    ctypes.CDLL(str(SYSTEM_LIBSTDCPP), mode=ctypes.RTLD_GLOBAL)

import numpy as np


LAUNCH_CWD = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/dtorbin/Downloads/STN-Neuron-main")
MECH_DIR = DATASET_ROOT / "sth"
DATA_DIR = MECH_DIR / "sth-data"
MORPH_DIR = DATASET_ROOT / "Detailed Morphology"
POOL_PATH = DATASET_ROOT / "MatingPool.pickle"
DEPRESSION_MECH_DIR = SCRIPT_DIR / "internal_shape_mechanisms"
BEST_POOL_INDEX = 44

# Running this script from a directory that already has an `x86_64/libnrnmech.so`
os.chdir(DATASET_ROOT)

from neuron import h, load_mechanisms
from neuron.units import ms

PARAMETER_NAMES = [
    "soma_gcaL",
    "soma_gcaN",
    "soma_gcaT",
    "soma_gIh",
    "soma_gKDR",
    "soma_gKv31",
    "soma_gsKCa",
    "soma_gNaL",
    "soma_gNa",
    "dend_gcaL_scale_prox",
    "dend_gcaN_all",
    "dend_gcaT_prox",
    "dend_gKDR_scale_prox",
    "dend_gKv31_scale_prox",
    "dend_gsKCa_scale_all",
    "dend_gNaL_scale_prox",
    "dend_gNa_scale_prox",
    "gpas",
    "Ra",
    "ais_gNa_scale",
]

# run toggles
DEFAULT_RUN_CONFIG = {
    "model": "gw",
    "morphology": "20160119_sham3.CNG.swc",
    "n_cells": 10,
    "param_index": None,
    "param_jitter_frac": 0.0,
    "seed": 2,
    "tstop_ms": 1000.0,
    "dbs_start_ms": 195,
    "dbs_stop_ms": 795,
    "amp_nA": 0.0,
    "delay_ms": 0.0,
    "dur_ms": 500.0,
    "temperature_c": 37.0,
    "suite": False,
    # plot/summary toggles
    "plot": True,
    "plot_geometry": True,
    "geometry_show_dendrites": True,
    "geometry_max_hdp_axons": 1,
    "geometry_max_gpe_axons": 1,
    "geometry_show_legend": True,
    "plot_spectrogram": False,
    "plot_spta": False,
    "plot_mpta": False,
    "plot_hilbert_phase": False,
    "plot_hilbert_amp": False,
    "plot_post_rate_distribution": False,
    "plot_plv_histogram": False,
    "plot_recruitment_dynamics": False,
    "paper_plot": False,
    "save_figure": False,
    "save_figure_dir": "/home/dtorbin/Downloads/articles",
    "print_summary": True,
    "print_activation_origin": False,
    "print_synapse_details": False,
    "analysis_fs_hz": None,
    "spectrogram_nfft": None,
    "spectrogram_window_ms": 100.0,
    "spectrogram_overlap_frac": 0.95,
    "spectrogram_fmax_hz": 500.0,
    "spectrogram_mode": "absolute",
    "spectrogram_rel_baseline": "pre_stim",
    "spectrogram_baseline_pre_ms": None,
    "spectrogram_cmap": "magma",
    "spta_baseline_pre_ms": 500.0,
    "mpta_period_fraction": 0.95,
    "mpta_window_scale": 1.0,
    "mpta_display_post_periods": 3.0,
    "hilbert_half_band_hz": 1.0,
    "dbs_rate_edge_window_ms": 100.0,
    "paper_trace_pre_ms": 75.0,
    "paper_trace_post_ms": 100.0,
}

DEFAULT_DBS_CONFIG = {
    "enabled": True,
    "a_positive_first": True,
    "start_ms": 500.0,
    "stop_ms": None,
    "freq_hz": 135.0,
    "pw_ms": 0.10,
    "amp_uA": 25.00,
    "ipg_ms": 0.05,
    "omit_pulse": None,
    "sigma_S_per_m": 0.33,
    "r_floor_mm": 0.05,
    # Anode & Cathode polarity + placement
    "electrode_a_pos_mm": (0.110, 0.025, 0.0),
    "electrode_b_pos_mm": (-0.110, -0.025, 0.0),
    "fiber_center_mm": (0.0, -0.145, 0.0),
    "fiber_radius_mm": 0.10,
    "use_manual_placement": False,
    "manual_soma_pos_mm": [(0.0, -0.145, 0.0)],
    "manual_axon_dir": [(-0.228, 0.973, 0.0)],
    "manual_dend_dir": [],
    "min_r_mm": 0.0,
    "max_r_mm": 0.10,
    "min_clearance_mm": 0.01,
}

DEFAULT_HDP_CONFIG = {
    "enabled": True,
    "n_axons": 12,
    # HDP axons Kita&Kita style
    "parent_diameter_um": 0.6,
    "diameter_jitter_frac": 0.0,
    "collateral_diameter_frac": 0.5,
    "parent_length_mm": 2.5,
    "parent_nodes": 13,
    "collateral_nodes": 7,
    "parent_pass_radius_mm": 0.18,
    "terminal_radius_mm": 0.10,
    "direction_jitter_frac": 0.25,
    "parent_dir": (-0.18, 0.97, 0.10),
    "seed_offset": 10000,
    "synapses_enabled": True,
    "inputs_per_cell": 12,
    "syn_target": "distal",
    "syn_distal_frac": 0.5,
    "syn_min_dist_um": None,
    "syn_weight_uS": 0.0005,
    "syn_tau1_ms": 1,
    "syn_tau2_ms": 4.0,
    "syn_delay_ms": 0.2,
    "syn_depression_enabled": True,
    "syn_depression_u": 0.03,
    "syn_depression_tau_rec_ms": 600.0,
    "syn_depression_tau_facil_ms": 1.0,
}

DEFAULT_GPE_CONFIG = {
    "enabled": True,
    "n_axons": 1,
    # GPE_axons Smith/Baufreton/Atherton style
    "parent_diameter_um": 0.8,
    "diameter_jitter_frac": 0.0,
    "collateral_diameter_frac": 0.5,
    "parent_length_mm": 2.0,
    "parent_nodes": 11,
    "collateral_nodes": 7,
    "parent_pass_radius_mm": 0.14,
    "terminal_radius_mm": 0.08,
    "direction_jitter_frac": 0.25,
    "parent_dir": (0.94, 0.28, 0.12),
    "seed_offset": 20000,
    "synapses_enabled": True,
    "inputs_per_cell": 1,
    "contacts_per_input": 15,
    "target_soma_frac": 0.30,
    "target_proximal_frac": 0.40,
    "target_distal_frac": 0.30,
    "syn_proximal_frac": 0.33,
    "syn_distal_frac": 0.5,
    "syn_proximal_max_dist_um": None,
    "syn_distal_min_dist_um": None,
    "syn_weight_uS": 0.0007,
    "syn_tau1_ms": 2,
    "syn_tau2_ms": 7.7,
    "syn_e_mV": -84.0,
    "syn_delay_ms": 0.2,
    "syn_depression_enabled": True,
    "syn_depression_u": 0.16,
    "syn_depression_tau_rec_ms": 120.0,
    "syn_depression_tau_facil_ms": 1.0,
}

_MECHANISMS_READY = False
_DEPRESSION_MECHANISMS_READY = False
_TREE_CACHE: dict[str, list[list[float]]] = {}
_LENGTH_CACHE: dict[str, list[float]] = {}
_SANITIZED_SWC_CACHE: dict[Path, Path] = {}
_SAVE_FIGURES = bool(DEFAULT_RUN_CONFIG["save_figure"])
_FIGURE_OUTPUT_DIR = Path(DEFAULT_RUN_CONFIG["save_figure_dir"])
_FIGURE_RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
_FIGURE_SAVE_COUNTER = 0


def parse_geometry_axon_limit(value) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "-1"}:
            return -1
        if text in {"none", "off", "no"}:
            return 0
        return int(text)
    value = int(value)
    return -1 if value < 0 else value


def configure_figure_saving(enabled: bool, output_dir: str | Path | None = None) -> None:
    global _SAVE_FIGURES, _FIGURE_OUTPUT_DIR, _FIGURE_SAVE_COUNTER
    _SAVE_FIGURES = bool(enabled)
    if output_dir is not None:
        _FIGURE_OUTPUT_DIR = Path(output_dir)
    _FIGURE_SAVE_COUNTER = 0


def _safe_figure_stem(title: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title).strip()).strip("._-")
    return stem[:140] if stem else "figure"


def finish_figure(fig, title: str) -> None:
    global _FIGURE_SAVE_COUNTER
    if _SAVE_FIGURES:
        _FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _FIGURE_SAVE_COUNTER += 1
        path = _FIGURE_OUTPUT_DIR / f"{_FIGURE_RUN_TAG}_{_FIGURE_SAVE_COUNTER:02d}_{_safe_figure_stem(title)}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {path}")

    import matplotlib.pyplot as plt

    plt.show()


def ensure_2024_dataset() -> None:
    required = [DATASET_ROOT, MECH_DIR, DATA_DIR, MORPH_DIR, POOL_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "2024 STN dataset is incomplete. Missing: " + ", ".join(missing)
        )


def ensure_2024_mechanisms() -> None:
    global _MECHANISMS_READY
    if _MECHANISMS_READY:
        return

    ensure_2024_dataset()
    lib_path = MECH_DIR / "x86_64" / "libnrnmech.so"
    if not lib_path.exists():
        subprocess.run(["nrnivmodl"], cwd=MECH_DIR, check=True)

    try:
        load_mechanisms(str(MECH_DIR))
    except RuntimeError as exc:
        # If another module already loaded mechanisms with the same suffix names
        # in this Python process, NEURON may reject the second library load.
        if "already exists" not in str(exc):
            raise
        if not hasattr(h, "axnode75") or not hasattr(h, "parak75"):
            raise RuntimeError(
                "Failed to load 2024 mechanisms. Run 2024.py in a fresh Python process."
            ) from exc

    h.load_file("stdrun.hoc")
    _MECHANISMS_READY = True


def ensure_depression_mechanisms() -> None:
    global _DEPRESSION_MECHANISMS_READY
    if _DEPRESSION_MECHANISMS_READY or hasattr(h, "DepExp2Syn"):
        _DEPRESSION_MECHANISMS_READY = True
        return

    lib_path = DEPRESSION_MECH_DIR / "x86_64" / "libnrnmech.so"
    if not lib_path.exists():
        subprocess.run(["nrnivmodl"], cwd=DEPRESSION_MECH_DIR, check=True)

    try:
        load_mechanisms(str(DEPRESSION_MECH_DIR))
    except RuntimeError as exc:
        if "already exists" not in str(exc):
            raise
        if not hasattr(h, "DepExp2Syn"):
            raise

    _DEPRESSION_MECHANISMS_READY = True


def make_plastic_exp2syn(
    section,
    x: float,
    *,
    e_mV: float,
    tau1_ms: float,
    tau2_ms: float,
    depression_enabled: bool,
    depression_u: float,
    depression_tau_rec_ms: float,
):
    if depression_enabled:
        ensure_depression_mechanisms()
        syn = h.DepExp2Syn(section(float(x)))
        syn.U = max(0.0, float(depression_u))
        syn.tau_rec = max(0.001, float(depression_tau_rec_ms))
    else:
        syn = h.Exp2Syn(section(float(x)))
    syn.e = float(e_mV)
    syn.tau1 = float(tau1_ms)
    syn.tau2 = float(tau2_ms)
    return syn


def read_tree_dat(file_name: str) -> list[list[float]]:
    if file_name in _TREE_CACHE:
        return _TREE_CACHE[file_name]

    rows: list[list[float]] = []
    with open(DATA_DIR / file_name, newline="", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 6:
                continue
            rows.append([float(value) for value in parts])

    rows.sort(key=lambda row: row[0])
    _TREE_CACHE[file_name] = rows
    return rows


def read_length_csv(file_name: str, column_name: str) -> list[float]:
    cache_key = f"{file_name}:{column_name}"
    if cache_key in _LENGTH_CACHE:
        return _LENGTH_CACHE[cache_key]

    values: list[float] = []
    with open(DATA_DIR / file_name, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row[column_name]))

    _LENGTH_CACHE[cache_key] = values
    return values


def sanitize_swc_path(path: str | Path) -> Path:
    source = Path(path).resolve()
    cached = _SANITIZED_SWC_CACHE.get(source)
    if cached is not None and cached.exists():
        return cached

    target = Path(tempfile.gettempdir()) / f"{source.stem}_sanitized.swc"
    with open(source, "r", encoding="utf-8", errors="ignore") as src, open(
        target, "w", encoding="ascii"
    ) as dst:
        for raw_line in src:
            line = raw_line.encode("ascii", "ignore").decode("ascii").strip()
            if not line:
                continue
            if line.startswith("#"):
                dst.write(line + "\n")
                continue

            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                [float(value) for value in parts[:7]]
            except ValueError:
                continue
            dst.write(" ".join(parts[:7]) + "\n")

    _SANITIZED_SWC_CACHE[source] = target
    return target


def tree_entry(index: int, tree_rows: list[list[float]]) -> tuple[list[int], float, float, int]:
    row = tree_rows[index]
    children = [int(row[1]) - 1, int(row[2]) - 1]
    diam = float(row[3])
    length = float(row[4])
    nseg = int(row[5])
    return children, diam, length, nseg


def insert_somatodendritic_channels(sec) -> None:
    for mech in ("STh", "Na", "NaL", "KDR", "Kv31", "Ih", "Cacum", "sKCa", "CaT", "HVA", "extracellular"):
        sec.insert(mech)
    sec.xraxial[0] = 1e9
    sec.xg[0] = 1e9
    sec.xc[0] = 0


def set_acsf(condition: int = 4) -> None:
    if condition == 3:
        h.nai0_na_ion = 15
        h.nao0_na_ion = 150
        h.ki0_k_ion = 140
        h.ko0_k_ion = 3.6
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.4
    elif condition == 4:
        h.nai0_na_ion = 15
        h.nao0_na_ion = 128.5
        h.ki0_k_ion = 140
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.0
    else:
        h.nai0_na_ion = 10
        h.nao0_na_ion = 140
        h.ki0_k_ion = 54
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 5e-05
        h.cao0_ca_ion = 2.0


def load_mating_pool():
    ensure_2024_dataset()
    with open(POOL_PATH, "rb") as handle:
        pool, scores = pickle.load(handle)
    return pool, scores


def best_parameter_index() -> int:
    pool, scores = load_mating_pool()
    return int(np.argmin(np.array(scores, dtype=float)))


def load_parameter_vector(index: int | None = None) -> list[float]:
    pool, scores = load_mating_pool()
    if index is None:
        index = int(np.argmin(np.array(scores, dtype=float)))
    return [float(value) for value in pool[index]]


def parameter_dict(params: Iterable[float]) -> dict[str, float]:
    values = list(params)
    return dict(zip(PARAMETER_NAMES, values))


def list_available_morphologies() -> list[Path]:
    ensure_2024_dataset()
    return sorted(MORPH_DIR.glob("*.swc"))


def parse_parameter_overrides(items: Iterable[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if items is None:
        return overrides

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Use name=value.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in PARAMETER_NAMES:
            raise ValueError(
                f"Unknown parameter '{name}'. Valid names: {', '.join(PARAMETER_NAMES)}"
            )
        overrides[name] = float(raw_value)
    return overrides


def apply_parameter_overrides(params: Iterable[float], overrides: dict[str, float] | None = None) -> list[float]:
    values = list(params)
    if not overrides:
        return values

    for i, name in enumerate(PARAMETER_NAMES):
        if name in overrides:
            values[i] = float(overrides[name])
    return values


def build_parameter_vector(
    param_index: int | None = None,
    overrides: dict[str, float] | None = None,
) -> list[float]:
    return apply_parameter_overrides(load_parameter_vector(param_index), overrides)


def jitter_parameter_vector(params: Iterable[float], frac: float, rng: np.random.Generator) -> list[float]:
    values = np.array(list(params), dtype=float)
    if frac <= 0.0:
        return values.tolist()

    scale = rng.uniform(1.0 - frac, 1.0 + frac, size=len(values))
    values = np.maximum(values * scale, 0.0)
    return values.tolist()


def resolve_morphology(morphology: str | Path | None, cell_index: int = 0):
    if morphology is None:
        return None

    if isinstance(morphology, Path):
        return morphology

    if morphology == "auto":
        options = list_available_morphologies()
        if not options:
            raise FileNotFoundError(f"No SWC morphologies found in {MORPH_DIR}")
        return options[cell_index % len(options)]

    candidate = Path(morphology)
    if candidate.is_absolute():
        return candidate
    if (LAUNCH_CWD / candidate).exists():
        return LAUNCH_CWD / candidate
    if (MORPH_DIR / candidate).exists():
        return MORPH_DIR / candidate
    return candidate


def normalize(vector: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = [float(value) for value in vector]
    length = float(np.sqrt(x * x + y * y + z * z))
    if length <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (x / length, y / length, z / length)


def dist_mm(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, az = [float(value) for value in a]
    bx, by, bz = [float(value) for value in b]
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))


def random_unit_vector(rng: np.random.Generator) -> tuple[float, float, float]:
    while True:
        v = rng.uniform(-1.0, 1.0, size=3)
        length = float(np.linalg.norm(v))
        if length > 1e-12:
            return tuple((v / length).tolist())


def random_pos_in_shell(
    min_r_mm: float,
    max_r_mm: float,
    rng: np.random.Generator,
    center_mm: Iterable[float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    direction = np.array(random_unit_vector(rng))
    radius = float(rng.uniform(min_r_mm, max_r_mm))
    center = np.array([float(v) for v in center_mm], dtype=float)
    return tuple((center + direction * radius).tolist())


def dist_point_to_segment_mm(point, start, end) -> float:
    point = np.array(point, dtype=float)
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    segment = end - start
    seg_len2 = float(np.dot(segment, segment))
    if seg_len2 <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def pick_safe_axon_dir(
    soma_pos_mm,
    *,
    electrode_a_pos_mm,
    electrode_b_pos_mm,
    min_clearance_mm: float,
    axon_path_len_mm: float,
    rng: np.random.Generator,
):
    while True:
        axon_dir = random_unit_vector(rng)
        start = np.array(soma_pos_mm, dtype=float)
        end = start + axon_path_len_mm * np.array(axon_dir, dtype=float)
        d_a = dist_point_to_segment_mm(electrode_a_pos_mm, start, end)
        d_b = dist_point_to_segment_mm(electrode_b_pos_mm, start, end)
        if d_a >= min_clearance_mm and d_b >= min_clearance_mm:
            return axon_dir


def pick_perpendicular_dend_dir(axon_dir, rng: np.random.Generator):
    axon = np.array(normalize(axon_dir), dtype=float)
    while True:
        rv = np.array(random_unit_vector(rng), dtype=float)
        rv = rv - np.dot(rv, axon) * axon
        length = float(np.linalg.norm(rv))
        if length > 1e-12:
            return tuple((rv / length).tolist())


def dbs_pulse_uA(t_ms: float, dbs_config: dict) -> float:
    if t_ms < dbs_config["start_ms"]:
        return 0.0
    stop_ms = dbs_config.get("stop_ms", None)
    if stop_ms is not None and t_ms >= float(stop_ms):
        return 0.0

    period_ms = 1000.0 / dbs_config["freq_hz"]
    elapsed_ms = t_ms - dbs_config["start_ms"]
    pulse_number = int(np.floor(elapsed_ms / period_ms)) + 1
    omit_pulse = dbs_config.get("omit_pulse", None)
    if omit_pulse is not None and int(omit_pulse) == pulse_number:
        return 0.0

    phase = elapsed_ms % period_ms
    pw_ms = dbs_config["pw_ms"]
    ipg_ms = dbs_config["ipg_ms"]
    amp_uA = dbs_config["amp_uA"]
    first_phase_sign = first_phase_sign_uA(dbs_config)

    if phase < pw_ms:
        return first_phase_sign * amp_uA
    if phase < pw_ms + ipg_ms:
        return 0.0
    if phase < 2.0 * pw_ms + ipg_ms:
        return -first_phase_sign * amp_uA
    return 0.0


def first_phase_sign_uA(dbs_config: dict) -> float:
    return -1.0 if dbs_config.get("a_positive_first", False) else 1.0


def point_source_phi_uA(I_uA: float, r_mm: float, sigma_S_per_m: float, r_floor_mm: float) -> float:
    r_mm_eff = max(float(r_mm), float(r_floor_mm))
    current_a = float(I_uA) * 1e-6
    radius_m = r_mm_eff * 1e-3
    phi_v = current_a / (4.0 * np.pi * float(sigma_S_per_m) * radius_m)
    return phi_v * 1e3


def bipolar_phi_uA(I_uA: float, seg_pos_mm, dbs_config: dict) -> float:
    r_c = dist_mm(seg_pos_mm, dbs_config["electrode_a_pos_mm"])
    r_a = dist_mm(seg_pos_mm, dbs_config["electrode_b_pos_mm"])
    sigma = dbs_config["sigma_S_per_m"]
    r_floor = dbs_config["r_floor_mm"]

    if I_uA >= 0.0:
        phi_cath = point_source_phi_uA(-I_uA, r_c, sigma, r_floor)
        phi_anod = point_source_phi_uA(+I_uA, r_a, sigma, r_floor)
    else:
        magnitude = -I_uA
        phi_cath = point_source_phi_uA(+magnitude, r_c, sigma, r_floor)
        phi_anod = point_source_phi_uA(-magnitude, r_a, sigma, r_floor)
    return phi_cath + phi_anod


def parse_triplet(raw_values):
    if raw_values is None:
        return None
    if len(raw_values) != 3:
        raise ValueError("Expected exactly 3 values for an xyz triplet.")
    return tuple(float(value) for value in raw_values)


def build_dbs_config_from_args(args) -> dict:
    config = dict(DEFAULT_DBS_CONFIG)
    config["enabled"] = bool(args.dbs)
    config["a_positive_first"] = bool(args.a_positive_first)
    config["start_ms"] = float(args.dbs_start)
    config["stop_ms"] = None if args.dbs_stop_ms is None or float(args.dbs_stop_ms) <= 0.0 else float(args.dbs_stop_ms)
    config["freq_hz"] = float(args.dbs_freq)
    config["pw_ms"] = float(args.dbs_pw)
    config["amp_uA"] = float(args.dbs_amp_uA)
    config["ipg_ms"] = float(args.dbs_ipg)
    config["omit_pulse"] = None if args.omit_pulse is None or int(args.omit_pulse) <= 0 else int(args.omit_pulse)
    config["sigma_S_per_m"] = float(args.sigma)
    config["electrode_a_pos_mm"] = tuple(float(value) for value in args.electrode_a)
    config["electrode_b_pos_mm"] = tuple(float(value) for value in args.electrode_b)
    config["fiber_center_mm"] = tuple(float(value) for value in args.fiber_center)
    config["fiber_radius_mm"] = float(args.fiber_radius_mm)
    config["min_r_mm"] = float(args.shell_min_r_mm)
    config["max_r_mm"] = float(args.shell_max_r_mm)
    if args.soma_pos is not None:
        config["manual_soma_pos_mm"] = [parse_triplet(args.soma_pos)]
        config["use_manual_placement"] = True
    if args.axon_dir is not None:
        config["manual_axon_dir"] = [normalize(args.axon_dir)]
        config["use_manual_placement"] = True
    if args.dend_dir is not None:
        config["manual_dend_dir"] = [normalize(args.dend_dir)]
        config["use_manual_placement"] = True
    return config


def build_hdp_config_from_args(args) -> dict:
    config = dict(DEFAULT_HDP_CONFIG)
    config["enabled"] = bool(args.hdp)
    config["n_axons"] = int(args.hdp_axons)
    config["parent_diameter_um"] = float(args.hdp_parent_diameter_um)
    config["diameter_jitter_frac"] = float(args.hdp_diameter_jitter_frac)
    config["collateral_diameter_frac"] = float(args.hdp_collateral_diameter_frac)
    config["parent_length_mm"] = float(args.hdp_parent_length_mm)
    config["parent_nodes"] = int(args.hdp_parent_nodes)
    config["collateral_nodes"] = int(args.hdp_collateral_nodes)
    config["parent_pass_radius_mm"] = float(args.hdp_parent_pass_radius_mm)
    config["terminal_radius_mm"] = float(args.hdp_terminal_radius_mm)
    config["direction_jitter_frac"] = float(args.hdp_direction_jitter_frac)
    config["synapses_enabled"] = bool(args.hdp_synapses)
    config["inputs_per_cell"] = int(args.hdp_inputs_per_cell)
    config["syn_target"] = str(args.hdp_syn_target)
    config["syn_distal_frac"] = float(args.hdp_syn_distal_frac)
    config["syn_min_dist_um"] = None if args.hdp_syn_min_dist_um is None else float(args.hdp_syn_min_dist_um)
    config["syn_weight_uS"] = float(args.hdp_syn_weight_uS)
    config["syn_tau1_ms"] = float(args.hdp_syn_tau1_ms)
    config["syn_tau2_ms"] = float(args.hdp_syn_tau2_ms)
    config["syn_delay_ms"] = float(args.hdp_syn_delay_ms)
    config["syn_depression_enabled"] = bool(args.hdp_syn_depression)
    config["syn_depression_u"] = float(args.hdp_syn_depression_u)
    config["syn_depression_tau_rec_ms"] = float(args.hdp_syn_depression_tau_rec_ms)
    config["syn_depression_tau_facil_ms"] = float(args.hdp_syn_depression_tau_facil_ms)
    if not config["enabled"]:
        config["synapses_enabled"] = False
    return config


def build_gpe_config_from_args(args) -> dict:
    config = dict(DEFAULT_GPE_CONFIG)
    config["enabled"] = bool(args.gpe)
    config["n_axons"] = int(args.gpe_axons)
    config["parent_diameter_um"] = float(args.gpe_parent_diameter_um)
    config["diameter_jitter_frac"] = float(args.gpe_diameter_jitter_frac)
    config["collateral_diameter_frac"] = float(args.gpe_collateral_diameter_frac)
    config["parent_length_mm"] = float(args.gpe_parent_length_mm)
    config["parent_nodes"] = int(args.gpe_parent_nodes)
    config["collateral_nodes"] = int(args.gpe_collateral_nodes)
    config["parent_pass_radius_mm"] = float(args.gpe_parent_pass_radius_mm)
    config["terminal_radius_mm"] = float(args.gpe_terminal_radius_mm)
    config["direction_jitter_frac"] = float(args.gpe_direction_jitter_frac)
    config["synapses_enabled"] = bool(args.gpe_synapses)
    config["inputs_per_cell"] = int(args.gpe_inputs_per_cell)
    config["contacts_per_input"] = int(args.gpe_contacts_per_input)
    config["target_soma_frac"] = float(args.gpe_target_soma_frac)
    config["target_proximal_frac"] = float(args.gpe_target_proximal_frac)
    config["target_distal_frac"] = float(args.gpe_target_distal_frac)
    config["syn_proximal_frac"] = float(args.gpe_syn_proximal_frac)
    config["syn_distal_frac"] = float(args.gpe_syn_distal_frac)
    config["syn_proximal_max_dist_um"] = (
        None if args.gpe_syn_proximal_max_dist_um is None else float(args.gpe_syn_proximal_max_dist_um)
    )
    config["syn_distal_min_dist_um"] = (
        None if args.gpe_syn_distal_min_dist_um is None else float(args.gpe_syn_distal_min_dist_um)
    )
    config["syn_weight_uS"] = float(args.gpe_syn_weight_uS)
    config["syn_tau1_ms"] = float(args.gpe_syn_tau1_ms)
    config["syn_tau2_ms"] = float(args.gpe_syn_tau2_ms)
    config["syn_e_mV"] = float(args.gpe_syn_e_mV)
    config["syn_delay_ms"] = float(args.gpe_syn_delay_ms)
    config["syn_depression_enabled"] = bool(args.gpe_syn_depression)
    config["syn_depression_u"] = float(args.gpe_syn_depression_u)
    config["syn_depression_tau_rec_ms"] = float(args.gpe_syn_depression_tau_rec_ms)
    config["syn_depression_tau_facil_ms"] = float(args.gpe_syn_depression_tau_facil_ms)
    if not config["enabled"]:
        config["synapses_enabled"] = False
    return config


class _AxonMixin:
    def _initialize_axon_constants(self) -> None:
        self.rhoa = 7e5
        self.mycm = 0.1
        self.mygm = 0.001
        self.nl = 30
        pi = np.pi
        node_d = 1.4
        axon_d = 1.6
        para_d1 = 1.4
        para_d2 = 1.6
        space_p1 = 0.002
        space_p2 = 0.004
        space_i = 0.004
        self.Rpn0 = (self.rhoa * 0.01) / (pi * ((((node_d / 2) + space_p1) ** 2) - ((node_d / 2) ** 2)))
        self.Rpn1 = (self.rhoa * 0.01) / (pi * ((((para_d1 / 2) + space_p1) ** 2) - ((para_d1 / 2) ** 2)))
        self.Rpn2 = (self.rhoa * 0.01) / (pi * ((((para_d2 / 2) + space_p2) ** 2) - ((para_d2 / 2) ** 2)))
        self.Rpx = (self.rhoa * 0.01) / (pi * ((((axon_d / 2) + space_i) ** 2) - ((axon_d / 2) ** 2)))

    def _build_axon(self, parent_sec) -> None:
        self.initseg = h.Section(name="initseg", cell=self)
        self.initseg.diam = 1.8904976874853334
        self.initseg.L = 21.7413353424173

        self.node = []
        self.MYSA = []
        self.FLUT = []
        self.STIN = []

        axon_nodes = 10
        paranodes1 = (axon_nodes - 1) * 2
        paranodes2 = (axon_nodes - 1) * 2
        axon_inter = (axon_nodes - 1) * 3

        for i in range(axon_nodes):
            sec = h.Section(name=f"node{i}", cell=self)
            sec.diam = 1.4
            sec.L = 1.0
            sec.nseg = 1
            self.node.append(sec)

        for i in range(paranodes1):
            sec = h.Section(name=f"MYSA{i}", cell=self)
            sec.diam = 1.4
            sec.L = 1.5 if i < 6 else 3.0
            sec.nseg = 1
            self.MYSA.append(sec)

        for i in range(paranodes2):
            sec = h.Section(name=f"FLUT{i}", cell=self)
            sec.diam = 1.6
            sec.L = 5.0 if i < 6 else 10.0
            sec.nseg = 1
            self.FLUT.append(sec)

        for i in range(axon_inter):
            sec = h.Section(name=f"STIN{i}", cell=self)
            sec.diam = 1.6
            sec.L = 29.0 if i < 9 else 58.0
            sec.nseg = 1
            self.STIN.append(sec)

        self.initseg.connect(parent_sec, 1.0)
        self.node[0].connect(self.initseg, 1.0)
        for i in range(axon_nodes - 1):
            self.MYSA[2 * i].connect(self.node[i], 1.0)
            self.FLUT[2 * i].connect(self.MYSA[2 * i], 1.0)
            self.STIN[3 * i].connect(self.FLUT[2 * i], 1.0)
            self.STIN[3 * i + 1].connect(self.STIN[3 * i], 1.0)
            self.STIN[3 * i + 2].connect(self.STIN[3 * i + 1], 1.0)
            self.FLUT[2 * i + 1].connect(self.STIN[3 * i + 2], 1.0)
            self.MYSA[2 * i + 1].connect(self.FLUT[2 * i + 1], 1.0)
            self.node[i + 1].connect(self.MYSA[2 * i + 1], 1.0)

        order = [self.initseg, self.node[0]]
        for i in range(axon_nodes - 1):
            order.extend(
                [
                    self.MYSA[2 * i],
                    self.FLUT[2 * i],
                    self.STIN[3 * i],
                    self.STIN[3 * i + 1],
                    self.STIN[3 * i + 2],
                    self.FLUT[2 * i + 1],
                    self.MYSA[2 * i + 1],
                    self.node[i + 1],
                ]
            )
        self._axon_path_start_um = {}
        offset_um = 0.0
        for sec in order:
            self._axon_path_start_um[sec.name()] = offset_um
            offset_um += float(sec.L)
        self._axon_total_len_um = offset_um

    def _setup_axon_biophysics(self) -> None:
        for sec in self.axon_sections:
            sec.Ra = self.rhoa / 10000.0
            sec.cm = 2.0

        for count, sec in enumerate(self.node):
            sec.insert("extracellular")
            sec.xraxial[0] = self.Rpn0
            sec.xg[0] = 1e10
            sec.xc[0] = 0
            if count == len(self.node) - 1:
                sec.insert("pas")
                for seg in sec:
                    seg.pas.g = 0.0001
                    seg.pas.e = -65
            else:
                sec.insert("axnode75")
                for seg in sec:
                    seg.axnode75.gnabar = 2.0
                    seg.axnode75.gnapbar = 0.05
                    seg.axnode75.gkbar = 0.07
                    seg.axnode75.gl = 0.005
                    seg.axnode75.ek = -85
                    seg.axnode75.ena = 55
                    seg.axnode75.el = -60

        for sec in self.MYSA:
            sec.insert("pas")
            sec.insert("extracellular")
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65
            sec.xraxial[0] = self.Rpn1
            sec.xg[0] = self.mygm / (self.nl * 2)
            sec.xc[0] = self.mycm / (self.nl * 2)

        for sec in self.FLUT:
            sec.insert("parak75")
            sec.insert("pas")
            sec.insert("extracellular")
            for seg in sec:
                seg.parak75.gkbar = 0.02
                seg.parak75.ek = -85
                seg.pas.g = 0.0001
                seg.pas.e = -60
            sec.xraxial[0] = self.Rpn2
            sec.xg[0] = self.mygm / (self.nl * 2)
            sec.xc[0] = self.mycm / (self.nl * 2)

        for sec in self.STIN:
            sec.insert("pas")
            sec.insert("extracellular")
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65
            sec.xraxial[0] = self.Rpx
            sec.xg[0] = self.mygm / (self.nl * 2)
            sec.xc[0] = self.mycm / (self.nl * 2)


def random_point_in_ball(radius_mm: float, rng: np.random.Generator) -> np.ndarray:
    direction = np.array(random_unit_vector(rng), dtype=float)
    radius = float(radius_mm) * float(rng.random()) ** (1.0 / 3.0)
    return direction * radius


def orthonormal_vector(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    candidate = np.array(random_unit_vector(rng), dtype=float)
    candidate = candidate - float(np.dot(candidate, direction)) * direction
    length = float(np.linalg.norm(candidate))
    if length <= 1e-12:
        candidate = np.array([direction[1], -direction[0], 0.0], dtype=float)
        length = float(np.linalg.norm(candidate))
    return candidate / max(length, 1e-12)


def interpolate_polyline(points: np.ndarray, distance_mm: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(3, dtype=float)
    if len(points) == 1:
        return points[0].copy()

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return points[0].copy()

    distance_mm = min(max(float(distance_mm), 0.0), total)
    idx = int(np.searchsorted(cumulative, distance_mm, side="right") - 1)
    idx = max(0, min(idx, len(seg_lengths) - 1))
    seg_len = float(seg_lengths[idx])
    if seg_len <= 1e-12:
        return points[idx].copy()
    alpha = (distance_mm - float(cumulative[idx])) / seg_len
    return points[idx] + alpha * (points[idx + 1] - points[idx])


def extracellular_resistance_for_diameter(diam_um: float, space_um: float, rhoa_ohm_cm: float = 7e5) -> float:
    diam_um = max(float(diam_um), 0.1)
    space_um = max(float(space_um), 1e-4)
    outer_r = diam_um / 2.0 + space_um
    inner_r = diam_um / 2.0
    area = np.pi * (outer_r * outer_r - inner_r * inner_r)
    return (float(rhoa_ohm_cm) * 0.01) / max(float(area), 1e-12)


class ParametricHDPAxon:
    pathway_name = "hdp"

    def __init__(self, axon_index: int, hdp_config: dict, dbs_config: dict, rng: np.random.Generator):
        ensure_2024_mechanisms()
        self.axon_index = int(axon_index)
        self.hdp_config = dict(hdp_config)
        self.pathway_name = str(getattr(self, "pathway_name", "hdp"))
        self.parent_sections = []
        self.collateral_sections = []
        self.parent_nodes = []
        self.collateral_nodes = []
        self.all_sections = []
        self._section_geometry: dict[str, dict] = {}
        self._section_kind: dict[str, str] = {}

        self.geometry = self._build_geometry(dbs_config, rng)
        parent_diam = float(hdp_config["parent_diameter_um"])
        jitter = max(0.0, float(hdp_config.get("diameter_jitter_frac", 0.0)))
        if jitter > 0.0:
            parent_diam *= float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        parent_diam = max(parent_diam, 0.25)
        collateral_diam = max(0.2, parent_diam * float(hdp_config["collateral_diameter_frac"]))
        self.parent_diameter_um = float(parent_diam)
        self.collateral_diameter_um = float(collateral_diam)

        self.parent_nodes = self._build_myelinated_cable(
            prefix=f"{self.pathway_name}{self.axon_index}_parent",
            path_points=self.geometry["parent_points_mm"],
            n_nodes=int(hdp_config["parent_nodes"]),
            diameter_um=parent_diam,
            connect_to=None,
            passive_endpoint_nodes=True,
            kind=f"{self.pathway_name}_parent",
        )
        branch_node_index = len(self.parent_nodes) // 2
        self.branch_node_index = int(branch_node_index)
        self.collateral_nodes = self._build_myelinated_cable(
            prefix=f"{self.pathway_name}{self.axon_index}_collateral",
            path_points=self.geometry["collateral_points_mm"],
            n_nodes=int(hdp_config["collateral_nodes"]),
            diameter_um=collateral_diam,
            connect_to=self.parent_nodes[branch_node_index],
            passive_endpoint_nodes=False,
            kind=f"{self.pathway_name}_collateral",
        )
        self.all_sections = self.parent_sections + self.collateral_sections

    def _build_geometry(self, dbs_config: dict, rng: np.random.Generator) -> dict:
        fiber_center = np.array(dbs_config["fiber_center_mm"], dtype=float)
        parent_dir = np.array(self.hdp_config["parent_dir"], dtype=float)
        parent_dir = parent_dir + float(self.hdp_config["direction_jitter_frac"]) * np.array(random_unit_vector(rng))
        parent_dir = parent_dir / max(float(np.linalg.norm(parent_dir)), 1e-12)
        radial_dir = orthonormal_vector(parent_dir, rng)

        terminal = fiber_center + random_point_in_ball(float(self.hdp_config["terminal_radius_mm"]), rng)
        branch_offset = radial_dir * float(self.hdp_config["parent_pass_radius_mm"]) * float(rng.uniform(0.4, 1.0))
        branch_point = fiber_center + branch_offset + random_point_in_ball(0.03, rng)
        parent_half = 0.5 * float(self.hdp_config["parent_length_mm"])
        parent_start = branch_point - parent_dir * parent_half
        parent_end = branch_point + parent_dir * parent_half
        bend = branch_point + 0.55 * (terminal - branch_point) + orthonormal_vector(terminal - branch_point, rng) * 0.025

        return {
            "parent_points_mm": np.vstack([parent_start, branch_point, parent_end]).astype(float),
            "collateral_points_mm": np.vstack([branch_point, bend, terminal]).astype(float),
            "branch_point_mm": branch_point.astype(float),
            "terminal_point_mm": terminal.astype(float),
            "parent_dir": parent_dir.astype(float),
        }

    def _compartment_lengths_um(self, interval_um: float) -> tuple[float, float, float]:
        interval_um = max(float(interval_um), 12.0)
        mysa = min(3.0, 0.04 * interval_um)
        flut = min(10.0, 0.08 * interval_um)
        stin = max(1.0, (interval_um - 2.0 * mysa - 2.0 * flut) / 3.0)
        return float(mysa), float(flut), float(stin)

    def _configure_node(self, sec, diameter_um: float, passive: bool) -> None:
        sec.Ra = 70.0
        sec.cm = 2.0
        sec.insert("extracellular")
        sec.xraxial[0] = extracellular_resistance_for_diameter(diameter_um, 0.002)
        sec.xg[0] = 1e10
        sec.xc[0] = 0.0
        if passive:
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65.0
        else:
            sec.insert("axnode75")
            for seg in sec:
                seg.axnode75.gnabar = 2.0
                seg.axnode75.gnapbar = 0.05
                seg.axnode75.gkbar = 0.07
                seg.axnode75.gl = 0.005
                seg.axnode75.ek = -85.0
                seg.axnode75.ena = 55.0
                seg.axnode75.el = -60.0

    def _configure_passive_myelin_section(self, sec, diameter_um: float, xraxial: float, leak_e: float = -65.0) -> None:
        sec.Ra = 70.0
        sec.cm = 2.0
        sec.insert("pas")
        sec.insert("extracellular")
        for seg in sec:
            seg.pas.g = 0.0001
            seg.pas.e = leak_e
        sec.xraxial[0] = xraxial
        sec.xg[0] = 0.001 / (30.0 * 2.0)
        sec.xc[0] = 0.1 / (30.0 * 2.0)

    def _configure_flut(self, sec, diameter_um: float) -> None:
        sec.Ra = 70.0
        sec.cm = 2.0
        sec.insert("parak75")
        sec.insert("pas")
        sec.insert("extracellular")
        for seg in sec:
            seg.parak75.gkbar = 0.02
            seg.parak75.ek = -85.0
            seg.pas.g = 0.0001
            seg.pas.e = -60.0
        sec.xraxial[0] = extracellular_resistance_for_diameter(diameter_um, 0.004)
        sec.xg[0] = 0.001 / (30.0 * 2.0)
        sec.xc[0] = 0.1 / (30.0 * 2.0)

    def _register_section_geometry(
        self,
        sec,
        path_points: np.ndarray,
        start_um: float,
        end_um: float,
        kind: str,
    ) -> None:
        self._section_geometry[sec.name()] = {
            "path_points_mm": np.asarray(path_points, dtype=float),
            "start_mm": float(start_um) * 1e-3,
            "end_mm": float(end_um) * 1e-3,
        }
        self._section_kind[sec.name()] = kind

    def _build_myelinated_cable(
        self,
        *,
        prefix: str,
        path_points: np.ndarray,
        n_nodes: int,
        diameter_um: float,
        connect_to,
        passive_endpoint_nodes: bool,
        kind: str,
    ) -> list:
        n_nodes = max(2, int(n_nodes))
        total_len_um = max(20.0, path_length_mm(path_points) * 1000.0)
        interval_um = total_len_um / float(n_nodes - 1)
        mysa_len, flut_len, stin_len = self._compartment_lengths_um(interval_um)

        nodes = []
        cable_sections = []
        order_entries = []
        for i in range(n_nodes):
            node = h.Section(name=f"{prefix}_node{i}")
            node.diam = max(0.2, 0.5 * float(diameter_um))
            node.L = 1.0
            node.nseg = 1
            self._configure_node(
                node,
                node.diam,
                passive=bool(passive_endpoint_nodes and (i == 0 or i == n_nodes - 1)),
            )
            nodes.append(node)
            cable_sections.append(node)
            order_entries.append((node, i * interval_um, i * interval_um + node.L))

        for i in range(n_nodes - 1):
            pieces = []
            specs = [
                ("MYSAa", mysa_len, max(0.2, 0.5 * diameter_um), "mysa"),
                ("FLUTa", flut_len, max(0.2, diameter_um), "flut"),
                ("STINa", stin_len, max(0.2, diameter_um), "stin"),
                ("STINb", stin_len, max(0.2, diameter_um), "stin"),
                ("STINc", stin_len, max(0.2, diameter_um), "stin"),
                ("FLUTb", flut_len, max(0.2, diameter_um), "flut"),
                ("MYSAb", mysa_len, max(0.2, 0.5 * diameter_um), "mysa"),
            ]
            for label, length_um, diam_um, sec_type in specs:
                sec = h.Section(name=f"{prefix}_{label}{i}")
                sec.L = float(length_um)
                sec.diam = float(diam_um)
                sec.nseg = 1
                if sec_type == "flut":
                    self._configure_flut(sec, diam_um)
                elif sec_type == "mysa":
                    self._configure_passive_myelin_section(
                        sec,
                        diam_um,
                        extracellular_resistance_for_diameter(diam_um, 0.002),
                        leak_e=-65.0,
                    )
                else:
                    self._configure_passive_myelin_section(
                        sec,
                        diam_um,
                        extracellular_resistance_for_diameter(diam_um, 0.004),
                        leak_e=-65.0,
                    )
                pieces.append(sec)
                cable_sections.append(sec)

            pieces[0].connect(nodes[i], 1.0)
            for left, right in zip(pieces[:-1], pieces[1:]):
                right.connect(left, 1.0)
            nodes[i + 1].connect(pieces[-1], 1.0)

            start_um = i * interval_um
            running_um = start_um
            for sec in pieces:
                end_um = running_um + float(sec.L)
                order_entries.append((sec, running_um, end_um))
                running_um = end_um

        if connect_to is not None:
            nodes[0].connect(connect_to, 1.0)

        for sec, start_um, end_um in order_entries:
            self._register_section_geometry(sec, path_points, start_um, end_um, kind)

        if "collateral" in prefix:
            self.collateral_sections.extend(cable_sections)
        else:
            self.parent_sections.extend(cable_sections)
        return nodes

    def segment_position_mm(self, sec, seg_x: float) -> tuple[float, float, float]:
        geom = self._section_geometry.get(sec.name())
        if geom is None:
            return tuple(np.asarray(self.geometry["branch_point_mm"], dtype=float).tolist())
        distance_mm = (1.0 - float(seg_x)) * float(geom["start_mm"]) + float(seg_x) * float(geom["end_mm"])
        return tuple(interpolate_polyline(geom["path_points_mm"], distance_mm).tolist())

    def recording_sites(self) -> list[dict]:
        sites = []
        for i, sec in enumerate(self.parent_nodes):
            sites.append(
                {
                    "name": f"parent_node{i}",
                    "kind": f"{self.pathway_name}_parent",
                    "node_index": i,
                    "segment": sec(0.5),
                }
            )
        for i, sec in enumerate(self.collateral_nodes):
            sites.append(
                {
                    "name": f"collateral_node{i}",
                    "kind": f"{self.pathway_name}_collateral",
                    "node_index": i,
                    "segment": sec(0.5),
                }
            )
        return sites


class ParametricGPeAxon(ParametricHDPAxon):
    pathway_name = "gpe"

    def __init__(self, axon_index: int, gpe_config: dict, dbs_config: dict, rng: np.random.Generator):
        super().__init__(
            axon_index=axon_index,
            hdp_config=gpe_config,
            dbs_config=dbs_config,
            rng=rng,
        )

    def _build_geometry(self, dbs_config: dict, rng: np.random.Generator) -> dict:
        fiber_center = np.array(dbs_config["fiber_center_mm"], dtype=float)
        parent_dir = np.array(self.hdp_config["parent_dir"], dtype=float)
        parent_dir = parent_dir + float(self.hdp_config["direction_jitter_frac"]) * np.array(random_unit_vector(rng))
        parent_dir = parent_dir / max(float(np.linalg.norm(parent_dir)), 1e-12)
        radial_dir = orthonormal_vector(parent_dir, rng)

        terminal = fiber_center + random_point_in_ball(float(self.hdp_config["terminal_radius_mm"]), rng)
        branch_offset = radial_dir * float(self.hdp_config["parent_pass_radius_mm"]) * float(rng.uniform(0.25, 0.85))
        branch_point = fiber_center + branch_offset + random_point_in_ball(0.02, rng)
        parent_half = 0.5 * float(self.hdp_config["parent_length_mm"])
        parent_start = branch_point - parent_dir * parent_half
        parent_end = branch_point + parent_dir * parent_half
        bend = branch_point + 0.65 * (terminal - branch_point) + orthonormal_vector(terminal - branch_point, rng) * 0.015

        return {
            "parent_points_mm": np.vstack([parent_start, branch_point, parent_end]).astype(float),
            "collateral_points_mm": np.vstack([branch_point, bend, terminal]).astype(float),
            "branch_point_mm": branch_point.astype(float),
            "terminal_point_mm": terminal.astype(float),
            "parent_dir": parent_dir.astype(float),
        }


def path_length_mm(path_points: np.ndarray) -> float:
    path_points = np.asarray(path_points, dtype=float)
    if len(path_points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))


def cache_hdp_segment_positions(axon: ParametricHDPAxon):
    entries = []
    for sec in axon.all_sections:
        for seg in sec:
            entries.append((seg, axon.segment_position_mm(sec, float(seg.x))))
    axon._dbs_segment_positions = entries
    return entries


def snapshot_hdp_geometry(axon: ParametricHDPAxon, dbs_config: dict) -> dict:
    entries = getattr(axon, "_dbs_segment_positions", None)
    if entries is None:
        entries = cache_hdp_segment_positions(axon)
    first_phase_current = first_phase_sign_uA(dbs_config) * float(dbs_config["amp_uA"])
    positions = np.array([pos for _, pos in entries], dtype=float)
    phi_mV = np.array([bipolar_phi_uA(first_phase_current, pos, dbs_config) for _, pos in entries], dtype=float)
    pathway_name = str(getattr(axon, "pathway_name", "hdp"))
    kinds = np.array(
        [axon._section_kind.get(seg.sec.name(), pathway_name) for seg, _ in entries],
        dtype=object,
    )
    return {
        "segment_positions_mm": positions,
        "segment_phi_mV": phi_mV,
        "segment_kinds": kinds,
        "parent_points_mm": np.asarray(axon.geometry["parent_points_mm"], dtype=float),
        "collateral_points_mm": np.asarray(axon.geometry["collateral_points_mm"], dtype=float),
        "branch_point_mm": np.asarray(axon.geometry["branch_point_mm"], dtype=float),
        "terminal_point_mm": np.asarray(axon.geometry["terminal_point_mm"], dtype=float),
        "parent_diameter_um": float(axon.parent_diameter_um),
        "collateral_diameter_um": float(axon.collateral_diameter_um),
    }


class GWSTN2024(_AxonMixin):
    def __init__(self, params: Iterable[float] | None = None):
        ensure_2024_mechanisms()
        self.params = list(load_parameter_vector() if params is None else params)
        self._initialize_axon_constants()
        set_acsf(4)

        self.tree0_rows = read_tree_dat("tree0-nom.dat")
        self.tree1_rows = read_tree_dat("tree1-nom.dat")
        self.tree0_lengths = read_length_csv("Tree_0_length.csv", "Tree 0")
        self.tree1_lengths = read_length_csv("Tree_1_length.csv", "Tree 1")

        self.soma = h.Section(name="soma", cell=self)
        self.soma.diam = 18.3112
        self.soma.L = 18.8

        self.dend0 = []
        self.dend1 = []
        self._build_gw_dendrites()
        self._build_axon(self.soma)

        self.somatodendritic_sections = [self.soma] + self.dend0 + self.dend1 + [self.initseg]
        self.axon_sections = self.node + self.MYSA + self.FLUT + self.STIN
        self.all_sections = self.somatodendritic_sections + self.axon_sections
        self._setup_biophysics()

    def _build_gw_dendrites(self) -> None:
        self._gw_dend_path_start_um = {}

        for i in range(11):
            children, diam, length, nseg = tree_entry(i, self.tree1_rows)
            sec = h.Section(name=f"dend1{i}", cell=self)
            sec.diam = diam
            sec.L = length
            sec.nseg = nseg
            self.dend1.append(sec)

        for i in range(11):
            children, _, _, _ = tree_entry(i, self.tree1_rows)
            if i == 0:
                self.dend1[i].connect(self.soma, 0.0)
                self._gw_dend_path_start_um[self.dend1[i].name()] = 0.0
            if children != [-1, -1]:
                parent_start = self._gw_dend_path_start_um[self.dend1[i].name()]
                self.dend1[children[0]].connect(self.dend1[i], 1.0)
                self.dend1[children[1]].connect(self.dend1[i], 1.0)
                self._gw_dend_path_start_um[self.dend1[children[0]].name()] = parent_start + float(self.dend1[i].L)
                self._gw_dend_path_start_um[self.dend1[children[1]].name()] = parent_start + float(self.dend1[i].L)

        for i in range(23):
            children, diam, length, nseg = tree_entry(i, self.tree0_rows)
            sec = h.Section(name=f"dend0{i}", cell=self)
            sec.diam = diam
            sec.L = length
            sec.nseg = nseg
            self.dend0.append(sec)

        for i in range(23):
            children, _, _, _ = tree_entry(i, self.tree0_rows)
            if i == 0:
                self.dend0[i].connect(self.soma, 1.0)
                self._gw_dend_path_start_um[self.dend0[i].name()] = 0.0
            if children != [-1, -1]:
                parent_start = self._gw_dend_path_start_um[self.dend0[i].name()]
                self.dend0[children[0]].connect(self.dend0[i], 1.0)
                self.dend0[children[1]].connect(self.dend0[i], 1.0)
                self._gw_dend_path_start_um[self.dend0[children[0]].name()] = parent_start + float(self.dend0[i].L)
                self._gw_dend_path_start_um[self.dend0[children[1]].name()] = parent_start + float(self.dend0[i].L)

    def _setup_biophysics(self) -> None:
        for sec in self.somatodendritic_sections:
            sec.Ra = self.params[18]
            sec.cm = 1.0
            insert_somatodendritic_channels(sec)

        self._setup_axon_biophysics()

        for seg in self.soma:
            seg.NaL.gna = self.params[7]
            seg.Na.gna = self.params[8]
            seg.HVA.gcaL = self.params[0]
            seg.HVA.gcaN = self.params[1]
            seg.CaT.gcaT = self.params[2]
            seg.Ih.gk = self.params[3]
            seg.KDR.gk = self.params[4]
            seg.Kv31.gk = self.params[5]
            seg.sKCa.gk = self.params[6]
            seg.STh.gpas = self.params[17]

        for seg in self.initseg:
            seg.NaL.gna = self.params[7]
            seg.Na.gna = self.params[8] * self.params[19]
            seg.HVA.gcaL = self.params[0]
            seg.HVA.gcaN = self.params[1]
            seg.CaT.gcaT = self.params[2]
            seg.Ih.gk = self.params[3]
            seg.KDR.gk = self.params[4]
            seg.Kv31.gk = self.params[5]
            seg.sKCa.gk = self.params[6]
            seg.STh.gpas = self.params[17]

        proximal = 359.0 / 2.0
        dend_index = 0
        for sec in self.dend0:
            for seg in sec:
                if self.tree0_lengths[dend_index] < proximal:
                    seg.NaL.gna = self.params[15] * self.params[7]
                    seg.Na.gna = self.params[16] * self.params[8]
                    seg.HVA.gcaL = self.params[9] * self.params[0]
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = self.params[11]
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = self.params[12] * self.params[4]
                    seg.Kv31.gk = self.params[13] * self.params[5]
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                else:
                    seg.NaL.gna = 0.0
                    seg.Na.gna = 0.0
                    seg.HVA.gcaL = 0.0
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = 0.0
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = 0.0
                    seg.Kv31.gk = 0.0
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                dend_index += 1

        dend_index = 0
        for sec in self.dend1:
            for seg in sec:
                if self.tree1_lengths[dend_index] < proximal:
                    seg.NaL.gna = self.params[15] * self.params[7]
                    seg.Na.gna = self.params[16] * self.params[8]
                    seg.HVA.gcaL = self.params[9] * self.params[0]
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = self.params[11]
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = self.params[12] * self.params[4]
                    seg.Kv31.gk = self.params[13] * self.params[5]
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                else:
                    seg.NaL.gna = 0.0
                    seg.Na.gna = 0.0
                    seg.HVA.gcaL = 0.0
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = 0.0
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = 0.0
                    seg.Kv31.gk = 0.0
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                dend_index += 1

        for sec in self.somatodendritic_sections:
            h.ion_style("na_ion", 1, 2, 1, 0, 1, sec=sec)
            h.ion_style("k_ion", 1, 2, 1, 0, 1, sec=sec)
            h.ion_style("ca_ion", 3, 2, 1, 1, 1, sec=sec)

    def __repr__(self) -> str:
        return "GWSTN2024"


class DetailedSTN2024(_AxonMixin):
    def __init__(self, params: Iterable[float] | None = None, morphology: str | Path | None = None):
        ensure_2024_mechanisms()
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")

        if morphology is None:
            morphology = MORPH_DIR / "20160119_sham3.CNG.swc"
        morphology = Path(morphology)
        if not morphology.is_absolute():
            launch_path = LAUNCH_CWD / morphology
            if launch_path.exists():
                morphology = launch_path
            else:
                morphology = DATASET_ROOT / morphology
        if not morphology.exists():
            raise FileNotFoundError(f"Detailed morphology not found: {morphology}")

        self.params = list(load_parameter_vector() if params is None else params)
        self.morphology_path = morphology

        reader = h.Import3d_SWC_read()
        reader.input(str(sanitize_swc_path(morphology)))
        importer = h.Import3d_GUI(reader, 0)
        importer.instantiate(self)

        set_acsf(4)
        for sec in self.dend:
            sec.L = sec.L * 2
            sec.nseg = 13

        try:
            for sec in self.axon:
                h.delete_section(sec=sec)
        except Exception:
            pass

        path_lengths = [h.distance(self.soma[0](0.5), sec(1.0)) for sec in self.dend]
        self.max_path = max(path_lengths)

        self._initialize_axon_constants()
        self._build_axon(self.soma[0])
        self.somatodendritic_sections = list(self.soma) + list(self.dend) + [self.initseg]
        self.axon_sections = self.node + self.MYSA + self.FLUT + self.STIN
        self.all_sections = self.somatodendritic_sections + self.axon_sections
        self._setup_biophysics()

    def _setup_biophysics(self) -> None:
        for sec in self.somatodendritic_sections:
            sec.Ra = self.params[18]
            sec.cm = 1.0
            insert_somatodendritic_channels(sec)

        self._setup_axon_biophysics()

        for sec in self.soma:
            for seg in sec:
                seg.NaL.gna = self.params[7]
                seg.Na.gna = self.params[8]
                seg.HVA.gcaL = self.params[0]
                seg.HVA.gcaN = self.params[1]
                seg.CaT.gcaT = self.params[2]
                seg.Ih.gk = self.params[3]
                seg.KDR.gk = self.params[4]
                seg.Kv31.gk = self.params[5]
                seg.sKCa.gk = self.params[6]
                seg.STh.gpas = self.params[17]

        for seg in self.initseg:
            seg.NaL.gna = self.params[7]
            seg.Na.gna = self.params[8] * self.params[19]
            seg.HVA.gcaL = self.params[0]
            seg.HVA.gcaN = self.params[1]
            seg.CaT.gcaT = self.params[2]
            seg.Ih.gk = self.params[3]
            seg.KDR.gk = self.params[4]
            seg.Kv31.gk = self.params[5]
            seg.sKCa.gk = self.params[6]
            seg.STh.gpas = self.params[17]

        proximal = self.max_path / 2.0
        for sec in self.dend:
            for seg in sec:
                if h.distance(self.soma[0](0.5), seg) < proximal:
                    seg.NaL.gna = self.params[15] * self.params[7]
                    seg.Na.gna = self.params[16] * self.params[8]
                    seg.HVA.gcaL = self.params[9] * self.params[0]
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = self.params[11]
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = self.params[12] * self.params[4]
                    seg.Kv31.gk = self.params[13] * self.params[5]
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                else:
                    seg.NaL.gna = 0.0
                    seg.Na.gna = 0.0
                    seg.HVA.gcaL = 0.0
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = 0.0
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = 0.0
                    seg.Kv31.gk = 0.0
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]

        for sec in self.somatodendritic_sections:
            h.ion_style("na_ion", 1, 2, 1, 0, 1, sec=sec)
            h.ion_style("k_ion", 1, 2, 1, 0, 1, sec=sec)
            h.ion_style("ca_ion", 3, 2, 1, 1, 1, sec=sec)

    def __repr__(self) -> str:
        return "DetailedSTN2024"


def create_2024_cell(
    model: str = "gw",
    params: Iterable[float] | None = None,
    morphology: str | Path | None = None,
    cell_index: int = 0,
):
    if model == "gw":
        return GWSTN2024(params=params)
    if model == "detail":
        return DetailedSTN2024(params=params, morphology=resolve_morphology(morphology, cell_index=cell_index))
    raise ValueError("model must be 'gw' or 'detail'")


def soma_section(cell):
    soma = cell.soma
    if hasattr(soma, "L") and hasattr(soma, "diam"):
        return soma
    return soma[0]


def section_has_pt3d(sec) -> bool:
    return int(sec.n3d()) > 0


def section_xyz_um(sec, x: float) -> tuple[float, float, float] | None:
    n3d = int(sec.n3d())
    if n3d <= 0:
        return None

    arcs = np.array([h.arc3d(i, sec=sec) for i in range(n3d)], dtype=float)
    xs = np.array([h.x3d(i, sec=sec) for i in range(n3d)], dtype=float)
    ys = np.array([h.y3d(i, sec=sec) for i in range(n3d)], dtype=float)
    zs = np.array([h.z3d(i, sec=sec) for i in range(n3d)], dtype=float)
    target = float(x) * float(arcs[-1])
    return (
        float(np.interp(target, arcs, xs)),
        float(np.interp(target, arcs, ys)),
        float(np.interp(target, arcs, zs)),
    )


def raw_soma_center_mm(cell) -> tuple[float, float, float]:
    soma_sec = soma_section(cell)
    xyz_um = section_xyz_um(soma_sec, 0.5)
    if xyz_um is None:
        return (0.0, 0.0, 0.0)
    return tuple(value * 1e-3 for value in xyz_um)


def build_cell_placement(cell, cell_index: int, dbs_config: dict, rng: np.random.Generator) -> dict:
    use_manual = bool(dbs_config["use_manual_placement"])

    manual_soma = dbs_config["manual_soma_pos_mm"]
    if use_manual and cell_index < len(manual_soma):
        soma_pos_mm = tuple(float(v) for v in manual_soma[cell_index])
    else:
        soma_pos_mm = random_pos_in_shell(
            dbs_config["min_r_mm"],
            dbs_config["max_r_mm"],
            rng,
            center_mm=dbs_config["fiber_center_mm"],
        )

    manual_axon = dbs_config["manual_axon_dir"]
    if use_manual and cell_index < len(manual_axon):
        axon_dir = normalize(manual_axon[cell_index])
    else:
        axon_path_len_mm = float(getattr(cell, "_axon_total_len_um", 0.0)) * 1e-3
        axon_dir = pick_safe_axon_dir(
            soma_pos_mm,
            electrode_a_pos_mm=dbs_config["electrode_a_pos_mm"],
            electrode_b_pos_mm=dbs_config["electrode_b_pos_mm"],
            min_clearance_mm=dbs_config["min_clearance_mm"],
            axon_path_len_mm=max(axon_path_len_mm, 0.54),
            rng=rng,
        )

    manual_dend = dbs_config["manual_dend_dir"]
    if use_manual and cell_index < len(manual_dend):
        dend_dir = normalize(manual_dend[cell_index])
    else:
        dend_dir = pick_perpendicular_dend_dir(axon_dir, rng)

    placement = {
        "soma_pos_mm": soma_pos_mm,
        "axon_dir": axon_dir,
        "dend_dir": dend_dir,
    }

    if hasattr(cell, "dend") and len(list(cell.dend)) > 0:
        raw_center = raw_soma_center_mm(cell)
        placement["translation_mm"] = (
            soma_pos_mm[0] - raw_center[0],
            soma_pos_mm[1] - raw_center[1],
            soma_pos_mm[2] - raw_center[2],
        )
    else:
        placement["translation_mm"] = soma_pos_mm

    return placement


def synthetic_segment_position_mm(cell, sec, seg_x: float, placement: dict) -> tuple[float, float, float]:
    sec_name = sec.name()
    soma_pos = np.array(placement["soma_pos_mm"], dtype=float)

    if sec_name in getattr(cell, "_axon_path_start_um", {}):
        start_mm = cell._axon_path_start_um[sec_name] * 1e-3
        pos = soma_pos + np.array(placement["axon_dir"], dtype=float) * (start_mm + float(sec.L) * 1e-3 * seg_x)
        return tuple(pos.tolist())

    if sec_name in getattr(cell, "_gw_dend_path_start_um", {}):
        start_mm = cell._gw_dend_path_start_um[sec_name] * 1e-3
        pos = soma_pos + np.array(placement["dend_dir"], dtype=float) * (start_mm + float(sec.L) * 1e-3 * seg_x)
        return tuple(pos.tolist())

    return tuple(soma_pos.tolist())


def segment_position_mm(cell, sec, seg_x: float, placement: dict) -> tuple[float, float, float]:
    xyz_um = section_xyz_um(sec, seg_x)
    if xyz_um is not None:
        translation = placement["translation_mm"]
        return (
            xyz_um[0] * 1e-3 + translation[0],
            xyz_um[1] * 1e-3 + translation[1],
            xyz_um[2] * 1e-3 + translation[2],
        )
    return synthetic_segment_position_mm(cell, sec, seg_x, placement)


def cache_segment_positions(cell, placement: dict):
    entries = []
    for sec in cell.all_sections:
        for seg in sec:
            entries.append((seg, segment_position_mm(cell, sec, float(seg.x), placement)))
    cell._dbs_segment_positions = entries
    cell._dbs_placement = placement
    return entries


def apply_bipolar_phi_to_cell(cell, I_uA: float, dbs_config: dict) -> None:
    entries = getattr(cell, "_dbs_segment_positions", None)
    if entries is None:
        raise RuntimeError("DBS positions were not cached for this cell.")
    for seg, pos_mm in entries:
        seg.e_extracellular = bipolar_phi_uA(I_uA, pos_mm, dbs_config)


def section_kind(sec_name: str) -> str:
    lower = sec_name.lower()
    if "soma" in lower:
        return "soma"
    if any(tag in lower for tag in ("dend",)):
        return "dend"
    return "axon"


def snapshot_segment_geometry(cell, dbs_config: dict):
    entries = getattr(cell, "_dbs_segment_positions", None)
    if entries is None:
        raise RuntimeError("DBS positions were not cached for this cell.")
    first_phase_current = first_phase_sign_uA(dbs_config) * float(dbs_config["amp_uA"])

    positions = np.array([pos for _, pos in entries], dtype=float)
    kinds = [section_kind(seg.sec.name()) for seg, _ in entries]
    phi_mV = np.array(
        [bipolar_phi_uA(first_phase_current, pos, dbs_config) for _, pos in entries],
        dtype=float,
    )
    return positions, phi_mV, kinds


def activation_recording_sites(cell) -> list[dict]:
    sites = [
        {
            "name": "soma",
            "kind": "soma",
            "distance_um": 0.0,
            "segment": soma_section(cell)(0.5),
        },
        {
            "name": "AIS",
            "kind": "AIS",
            "distance_um": float(cell.initseg.L) * 0.5,
            "segment": cell.initseg(0.5),
        },
    ]
    for index, sec in enumerate(getattr(cell, "node", [])):
        start_um = float(getattr(cell, "_axon_path_start_um", {}).get(sec.name(), np.nan))
        distance_um = start_um + float(sec.L) * 0.5 if np.isfinite(start_um) else np.nan
        sites.append(
            {
                "name": f"node{index}",
                "kind": "axon",
                "distance_um": distance_um,
                "segment": sec(0.5),
            }
        )
    return sites


def dendrite_sections(cell) -> list:
    if hasattr(cell, "dend"):
        dend = list(cell.dend)
        if dend:
            return dend
    sections = []
    sections.extend(list(getattr(cell, "dend0", [])))
    sections.extend(list(getattr(cell, "dend1", [])))
    return sections


def dendritic_path_distance_um(cell, sec, x: float) -> float:
    sec_name = sec.name()
    path_starts = getattr(cell, "_gw_dend_path_start_um", {})
    if sec_name in path_starts:
        return float(path_starts[sec_name]) + float(sec.L) * float(x)
    try:
        return float(h.distance(soma_section(cell)(0.5), sec(float(x))))
    except Exception:
        return np.nan


def max_dendritic_path_um(cell) -> float:
    max_path = getattr(cell, "max_path", None)
    if max_path is not None and np.isfinite(float(max_path)):
        return float(max_path)
    distances = [
        dendritic_path_distance_um(cell, sec, 1.0)
        for sec in dendrite_sections(cell)
    ]
    distances = [dist for dist in distances if np.isfinite(dist)]
    return float(max(distances)) if distances else np.nan


def hdp_synapse_distal_cutoff_um(cell, hdp_config: dict) -> float:
    manual_cutoff = hdp_config.get("syn_min_dist_um", None)
    if manual_cutoff is not None and np.isfinite(float(manual_cutoff)):
        return max(0.0, float(manual_cutoff))
    max_path = max_dendritic_path_um(cell)
    if not np.isfinite(max_path):
        return np.nan
    return max(0.0, float(hdp_config.get("syn_distal_frac", 0.5)) * float(max_path))


def collect_hdp_synapse_target_candidates(cell, hdp_config: dict) -> tuple[list[dict], float, bool]:
    target = str(hdp_config.get("syn_target", "distal"))
    cutoff_um = hdp_synapse_distal_cutoff_um(cell, hdp_config)
    fallback_to_all_dendrites = False

    if target == "soma":
        return (
            [
                {
                    "section": soma_section(cell),
                    "x": 0.5,
                    "distance_um": 0.0,
                    "kind": "soma",
                }
            ],
            cutoff_um,
            fallback_to_all_dendrites,
        )

    def dendrite_candidates(require_distal: bool) -> list[dict]:
        candidates = []
        for sec in dendrite_sections(cell):
            for seg in sec:
                distance_um = dendritic_path_distance_um(cell, sec, float(seg.x))
                if not np.isfinite(distance_um):
                    continue
                if require_distal and np.isfinite(cutoff_um) and distance_um < cutoff_um:
                    continue
                candidates.append(
                    {
                        "section": sec,
                        "x": float(seg.x),
                        "distance_um": float(distance_um),
                        "kind": "dendrite",
                    }
                )
        return candidates

    candidates = dendrite_candidates(require_distal=(target == "distal"))
    if target == "distal" and not candidates:
        candidates = dendrite_candidates(require_distal=False)
        fallback_to_all_dendrites = True

    if not candidates:
        raise RuntimeError("No eligible STN dendritic target segments were found for HDP synapses.")

    return candidates, cutoff_um, fallback_to_all_dendrites


def gpe_synapse_cutoffs_um(cell, gpe_config: dict) -> tuple[float, float]:
    max_path = max_dendritic_path_um(cell)
    if not np.isfinite(max_path):
        return np.nan, np.nan

    proximal_cutoff = gpe_config.get("syn_proximal_max_dist_um", None)
    if proximal_cutoff is None or not np.isfinite(float(proximal_cutoff)):
        proximal_cutoff = float(gpe_config.get("syn_proximal_frac", 0.33)) * float(max_path)

    distal_cutoff = gpe_config.get("syn_distal_min_dist_um", None)
    if distal_cutoff is None or not np.isfinite(float(distal_cutoff)):
        distal_cutoff = float(gpe_config.get("syn_distal_frac", 0.5)) * float(max_path)

    return max(0.0, float(proximal_cutoff)), max(0.0, float(distal_cutoff))


def collect_gpe_synapse_target_pools(cell, gpe_config: dict) -> tuple[dict[str, list[dict]], float, float, dict[str, bool]]:
    proximal_cutoff_um, distal_cutoff_um = gpe_synapse_cutoffs_um(cell, gpe_config)
    pools: dict[str, list[dict]] = {
        "soma": [
            {
                "section": soma_section(cell),
                "x": 0.5,
                "distance_um": 0.0,
                "kind": "soma",
            }
        ],
        "proximal": [],
        "distal": [],
    }

    all_dendrites = []
    for sec in dendrite_sections(cell):
        for seg in sec:
            distance_um = dendritic_path_distance_um(cell, sec, float(seg.x))
            if not np.isfinite(distance_um):
                continue
            candidate = {
                "section": sec,
                "x": float(seg.x),
                "distance_um": float(distance_um),
                "kind": "dendrite",
            }
            all_dendrites.append(candidate)
            if np.isfinite(proximal_cutoff_um) and distance_um <= proximal_cutoff_um:
                pools["proximal"].append(candidate)
            if np.isfinite(distal_cutoff_um) and distance_um >= distal_cutoff_um:
                pools["distal"].append(candidate)

    fallback = {"proximal": False, "distal": False}
    if not pools["proximal"] and all_dendrites:
        pools["proximal"] = list(all_dendrites)
        fallback["proximal"] = True
    if not pools["distal"] and all_dendrites:
        pools["distal"] = list(all_dendrites)
        fallback["distal"] = True
    if not all_dendrites:
        raise RuntimeError("No eligible STN dendritic target segments were found for GPe synapses.")

    return pools, proximal_cutoff_um, distal_cutoff_um, fallback


def allocate_counts_by_fraction(total: int, fractions: dict[str, float]) -> dict[str, int]:
    total = max(0, int(total))
    cleaned = {key: max(0.0, float(value)) for key, value in fractions.items()}
    fraction_sum = sum(cleaned.values())
    if total <= 0 or fraction_sum <= 0.0:
        return {key: 0 for key in cleaned}

    raw = {key: total * value / fraction_sum for key, value in cleaned.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(cleaned, key=lambda key: (raw[key] - counts[key], cleaned[key]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def setup_coupled_hdp_synapses(
    *,
    cell,
    cell_index: int,
    hdp_config: dict | None,
    dbs_config: dict | None,
    rng: np.random.Generator,
) -> dict:
    if not hdp_config or not hdp_config.get("enabled", False):
        return {"enabled": False, "reason": "hdp_disabled"}
    if not hdp_config.get("synapses_enabled", False):
        return {"enabled": False, "reason": "synapses_disabled"}
    if not dbs_config or not dbs_config.get("enabled", False):
        return {"enabled": False, "reason": "dbs_disabled"}

    inputs_per_cell = max(0, int(hdp_config.get("inputs_per_cell", 0)))
    if inputs_per_cell <= 0:
        return {"enabled": False, "reason": "no_inputs"}

    n_axons = max(1, int(hdp_config.get("n_axons", 0)))
    axon_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    target_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    axon_index_offset = int(cell_index) * 100000
    axons = [
        ParametricHDPAxon(
            axon_index=axon_index_offset + i,
            hdp_config=hdp_config,
            dbs_config=dbs_config,
            rng=axon_rng,
        )
        for i in range(n_axons)
    ]
    for axon in axons:
        cache_hdp_segment_positions(axon)

    candidates, cutoff_um, fallback_to_all_dendrites = collect_hdp_synapse_target_candidates(cell, hdp_config)
    candidate_indices = target_rng.choice(
        len(candidates),
        size=inputs_per_cell,
        replace=inputs_per_cell > len(candidates),
    )

    tau1_ms = max(0.001, float(hdp_config.get("syn_tau1_ms", 0.5)))
    tau2_ms = max(tau1_ms + 0.001, float(hdp_config.get("syn_tau2_ms", 3.0)))
    weight_uS = max(0.0, float(hdp_config.get("syn_weight_uS", 0.0005)))
    delay_ms = max(0.0, float(hdp_config.get("syn_delay_ms", 0.2)))
    depression_enabled = bool(hdp_config.get("syn_depression_enabled", False))
    depression_u = max(0.0, float(hdp_config.get("syn_depression_u", 0.0)))
    depression_tau_rec_ms = max(0.001, float(hdp_config.get("syn_depression_tau_rec_ms", 600.0)))
    depression_tau_facil_ms = max(0.0, float(hdp_config.get("syn_depression_tau_facil_ms", 1.0)))

    synapses = []
    netcons = []
    target_infos = []
    for input_index, candidate_index in enumerate(candidate_indices):
        candidate = candidates[int(candidate_index)]
        axon = axons[input_index % len(axons)]
        syn = make_plastic_exp2syn(
            candidate["section"],
            candidate["x"],
            e_mV=0.0,
            tau1_ms=tau1_ms,
            tau2_ms=tau2_ms,
            depression_enabled=depression_enabled,
            depression_u=depression_u,
            depression_tau_rec_ms=depression_tau_rec_ms,
        )

        terminal_sec = axon.collateral_nodes[-1]
        terminal = terminal_sec(0.5)
        netcon = h.NetCon(terminal._ref_v, syn, sec=terminal_sec)
        netcon.threshold = 0.0
        netcon.delay = delay_ms
        netcon.weight[0] = weight_uS

        synapses.append(syn)
        netcons.append(netcon)
        target_infos.append(
            {
                "cell_index": int(cell_index),
                "input_index": int(input_index),
                "axon_index": int(axon.axon_index),
                "section": candidate["section"].name(),
                "x": float(candidate["x"]),
                "distance_um": float(candidate["distance_um"]),
                "kind": candidate["kind"],
            }
        )

    recorders = []
    for axon in axons:
        for site in axon.recording_sites():
            detector = h.APCount(site["segment"])
            detector.thresh = 0.0
            times = h.Vector()
            detector.record(times)
            recorders.append(
                {
                    "axon": axon,
                    "site_name": site["name"],
                    "kind": site["kind"],
                    "node_index": int(site["node_index"]),
                    "detector": detector,
                    "times": times,
                }
            )

    return {
        "enabled": True,
        "cell_index": int(cell_index),
        "axons": axons,
        "synapses": synapses,
        "netcons": netcons,
        "recorders": recorders,
        "target_infos": target_infos,
        "target": str(hdp_config.get("syn_target", "distal")),
        "cutoff_um": float(cutoff_um) if np.isfinite(cutoff_um) else np.nan,
        "fallback_to_all_dendrites": bool(fallback_to_all_dendrites),
        "n_candidates": int(len(candidates)),
        "weight_uS": weight_uS,
        "tau1_ms": tau1_ms,
        "tau2_ms": tau2_ms,
        "delay_ms": delay_ms,
        "depression_enabled": depression_enabled,
        "depression_u": depression_u,
        "depression_tau_rec_ms": depression_tau_rec_ms,
        "depression_tau_facil_ms": depression_tau_facil_ms,
    }


def setup_coupled_gpe_synapses(
    *,
    cell,
    cell_index: int,
    gpe_config: dict | None,
    dbs_config: dict | None,
    rng: np.random.Generator,
) -> dict:
    if not gpe_config or not gpe_config.get("enabled", False):
        return {"enabled": False, "reason": "gpe_disabled"}
    if not gpe_config.get("synapses_enabled", False):
        return {"enabled": False, "reason": "synapses_disabled"}
    if not dbs_config or not dbs_config.get("enabled", False):
        return {"enabled": False, "reason": "dbs_disabled"}

    inputs_per_cell = max(0, int(gpe_config.get("inputs_per_cell", 0)))
    contacts_per_input = max(0, int(gpe_config.get("contacts_per_input", 0)))
    if inputs_per_cell <= 0 or contacts_per_input <= 0:
        return {"enabled": False, "reason": "no_inputs"}

    n_axons = max(1, int(gpe_config.get("n_axons", 0)))
    axon_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    target_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    axon_index_offset = int(cell_index) * 100000
    axons = [
        ParametricGPeAxon(
            axon_index=axon_index_offset + i,
            gpe_config=gpe_config,
            dbs_config=dbs_config,
            rng=axon_rng,
        )
        for i in range(n_axons)
    ]
    for axon in axons:
        cache_hdp_segment_positions(axon)

    pools, proximal_cutoff_um, distal_cutoff_um, fallback = collect_gpe_synapse_target_pools(cell, gpe_config)
    target_counts = allocate_counts_by_fraction(
        contacts_per_input,
        {
            "soma": float(gpe_config.get("target_soma_frac", 0.30)),
            "proximal": float(gpe_config.get("target_proximal_frac", 0.40)),
            "distal": float(gpe_config.get("target_distal_frac", 0.30)),
        },
    )

    tau1_ms = max(0.001, float(gpe_config.get("syn_tau1_ms", 0.4)))
    tau2_ms = max(tau1_ms + 0.001, float(gpe_config.get("syn_tau2_ms", 7.7)))
    weight_uS = max(0.0, float(gpe_config.get("syn_weight_uS", 0.0007)))
    reversal_mV = float(gpe_config.get("syn_e_mV", -84.0))
    delay_ms = max(0.0, float(gpe_config.get("syn_delay_ms", 0.2)))
    depression_enabled = bool(gpe_config.get("syn_depression_enabled", False))
    depression_u = max(0.0, float(gpe_config.get("syn_depression_u", 0.0)))
    depression_tau_rec_ms = max(0.001, float(gpe_config.get("syn_depression_tau_rec_ms", 30.0)))
    depression_tau_facil_ms = max(0.0, float(gpe_config.get("syn_depression_tau_facil_ms", 1.0)))

    synapses = []
    netcons = []
    target_infos = []
    for input_index in range(inputs_per_cell):
        axon = axons[input_index % len(axons)]
        terminal_sec = axon.collateral_nodes[-1]
        terminal = terminal_sec(0.5)
        contact_index = 0
        for target_kind, count in target_counts.items():
            candidates = pools.get(target_kind, [])
            if not candidates:
                continue
            candidate_indices = target_rng.choice(
                len(candidates),
                size=int(count),
                replace=int(count) > len(candidates),
            )
            for candidate_index in candidate_indices:
                candidate = candidates[int(candidate_index)]
                syn = make_plastic_exp2syn(
                    candidate["section"],
                    candidate["x"],
                    e_mV=reversal_mV,
                    tau1_ms=tau1_ms,
                    tau2_ms=tau2_ms,
                    depression_enabled=depression_enabled,
                    depression_u=depression_u,
                    depression_tau_rec_ms=depression_tau_rec_ms,
                )

                netcon = h.NetCon(terminal._ref_v, syn, sec=terminal_sec)
                netcon.threshold = 0.0
                netcon.delay = delay_ms
                netcon.weight[0] = weight_uS

                synapses.append(syn)
                netcons.append(netcon)
                target_infos.append(
                    {
                        "cell_index": int(cell_index),
                        "input_index": int(input_index),
                        "contact_index": int(contact_index),
                        "axon_index": int(axon.axon_index),
                        "section": candidate["section"].name(),
                        "x": float(candidate["x"]),
                        "distance_um": float(candidate["distance_um"]),
                        "kind": candidate["kind"],
                        "target_kind": target_kind,
                    }
                )
                contact_index += 1

    recorders = []
    for axon in axons:
        for site in axon.recording_sites():
            detector = h.APCount(site["segment"])
            detector.thresh = 0.0
            times = h.Vector()
            detector.record(times)
            recorders.append(
                {
                    "axon": axon,
                    "site_name": site["name"],
                    "kind": site["kind"],
                    "node_index": int(site["node_index"]),
                    "detector": detector,
                    "times": times,
                }
            )

    return {
        "enabled": True,
        "cell_index": int(cell_index),
        "axons": axons,
        "synapses": synapses,
        "netcons": netcons,
        "recorders": recorders,
        "target_infos": target_infos,
        "target_counts_per_input": target_counts,
        "inputs_per_cell": int(inputs_per_cell),
        "contacts_per_input": int(contacts_per_input),
        "proximal_cutoff_um": float(proximal_cutoff_um) if np.isfinite(proximal_cutoff_um) else np.nan,
        "distal_cutoff_um": float(distal_cutoff_um) if np.isfinite(distal_cutoff_um) else np.nan,
        "fallback_to_all_dendrites": fallback,
        "n_candidates": {key: int(len(value)) for key, value in pools.items()},
        "weight_uS": weight_uS,
        "tau1_ms": tau1_ms,
        "tau2_ms": tau2_ms,
        "reversal_mV": reversal_mV,
        "delay_ms": delay_ms,
        "depression_enabled": depression_enabled,
        "depression_u": depression_u,
        "depression_tau_rec_ms": depression_tau_rec_ms,
        "depression_tau_facil_ms": depression_tau_facil_ms,
    }


def materialize_coupled_hdp_results(coupling: dict, t_vec, dbs_config: dict | None, tstop_ms: float) -> list[dict]:
    if not coupling or not coupling.get("enabled", False) or not dbs_config:
        return []

    pulse_times = dbs_first_phase_pulse_times_ms(0.0, float(tstop_ms), dbs_config)
    first_pulse = float(pulse_times[0]) if len(pulse_times) else float(dbs_config["start_ms"])
    t_ms = np.asarray(t_vec, dtype=float)
    results = []

    for axon in coupling["axons"]:
        site_spikes = {}
        site_firsts = []
        for rec in coupling["recorders"]:
            if rec["axon"] is not axon:
                continue
            spikes = np.asarray(rec["times"], dtype=float)
            site_spikes[rec["site_name"]] = spikes
            first = first_spike_after(spikes, first_pulse)
            if np.isfinite(first):
                site_firsts.append((first, rec["site_name"], rec["kind"], rec["node_index"]))

        if site_firsts:
            first_time, first_site, first_kind, first_node = min(site_firsts, key=lambda item: item[0])
            activated = True
            first_latency = float(first_time - first_pulse)
        else:
            first_time = np.nan
            first_site = "none"
            first_kind = "none"
            first_node = -1
            activated = False
            first_latency = np.nan

        terminal_name = f"collateral_node{len(axon.collateral_nodes) - 1}"
        cortical_name = "parent_node0"
        distal_parent_name = f"parent_node{len(axon.parent_nodes) - 1}"
        terminal_first = first_spike_after(site_spikes.get(terminal_name, np.array([])), first_pulse)
        cortical_first = first_spike_after(site_spikes.get(cortical_name, np.array([])), first_pulse)
        distal_parent_first = first_spike_after(site_spikes.get(distal_parent_name, np.array([])), first_pulse)

        result = {
            "cell_index": int(coupling["cell_index"]),
            "axon_index": int(axon.axon_index),
            "activated": bool(activated),
            "first_spike_time_ms": float(first_time) if np.isfinite(first_time) else np.nan,
            "first_pulse_latency_ms": first_latency,
            "first_spike_site": first_site,
            "first_spike_kind": first_kind,
            "first_spike_node_index": int(first_node),
            "terminal_arrival_ms": float(terminal_first - first_pulse) if np.isfinite(terminal_first) else np.nan,
            "cortical_end_arrival_ms": float(cortical_first - first_pulse) if np.isfinite(cortical_first) else np.nan,
            "distal_parent_arrival_ms": float(distal_parent_first - first_pulse) if np.isfinite(distal_parent_first) else np.nan,
            "site_spike_times_ms": site_spikes,
            "t_ms": t_ms,
            "dbs_uA": np.array([dbs_pulse_uA(float(value), dbs_config) for value in t_ms]),
            "mode": "coupled",
        }
        result.update(snapshot_hdp_geometry(axon, dbs_config))
        results.append(result)

    return results


def materialize_coupled_gpe_results(coupling: dict, t_vec, dbs_config: dict | None, tstop_ms: float) -> list[dict]:
    results = materialize_coupled_hdp_results(coupling, t_vec, dbs_config, tstop_ms)
    for result in results:
        result["mode"] = "gpe_coupled"
    return results


def summarize_coupled_hdp_synapses(coupling: dict) -> dict:
    if not coupling or not coupling.get("enabled", False):
        return {
            "enabled": False,
            "n_synapses": 0,
            "n_axons": 0,
            "target_distances_um": np.array([], dtype=float),
        }
    distances = np.array([info["distance_um"] for info in coupling["target_infos"]], dtype=float)
    return {
        "enabled": True,
        "cell_index": int(coupling["cell_index"]),
        "n_synapses": int(len(coupling["synapses"])),
        "n_axons": int(len(coupling["axons"])),
        "target": coupling["target"],
        "cutoff_um": float(coupling["cutoff_um"]),
        "fallback_to_all_dendrites": bool(coupling["fallback_to_all_dendrites"]),
        "n_candidates": int(coupling["n_candidates"]),
        "target_infos": list(coupling["target_infos"]),
        "target_distances_um": distances,
        "weight_uS": float(coupling["weight_uS"]),
        "tau1_ms": float(coupling["tau1_ms"]),
        "tau2_ms": float(coupling["tau2_ms"]),
        "delay_ms": float(coupling["delay_ms"]),
        "depression_enabled": bool(coupling.get("depression_enabled", False)),
        "depression_u": float(coupling.get("depression_u", np.nan)),
        "depression_tau_rec_ms": float(coupling.get("depression_tau_rec_ms", np.nan)),
        "depression_tau_facil_ms": float(coupling.get("depression_tau_facil_ms", np.nan)),
    }


def summarize_coupled_gpe_synapses(coupling: dict) -> dict:
    if not coupling or not coupling.get("enabled", False):
        return {
            "enabled": False,
            "n_synapses": 0,
            "n_axons": 0,
            "target_distances_um": np.array([], dtype=float),
        }
    distances = np.array([info["distance_um"] for info in coupling["target_infos"]], dtype=float)
    target_counts = {}
    for info in coupling["target_infos"]:
        target_kind = str(info.get("target_kind", "unknown"))
        target_counts[target_kind] = target_counts.get(target_kind, 0) + 1
    return {
        "enabled": True,
        "cell_index": int(coupling["cell_index"]),
        "n_synapses": int(len(coupling["synapses"])),
        "n_axons": int(len(coupling["axons"])),
        "inputs_per_cell": int(coupling["inputs_per_cell"]),
        "contacts_per_input": int(coupling["contacts_per_input"]),
        "target_counts": target_counts,
        "target_counts_per_input": dict(coupling["target_counts_per_input"]),
        "proximal_cutoff_um": float(coupling["proximal_cutoff_um"]),
        "distal_cutoff_um": float(coupling["distal_cutoff_um"]),
        "fallback_to_all_dendrites": dict(coupling["fallback_to_all_dendrites"]),
        "n_candidates": dict(coupling["n_candidates"]),
        "target_infos": list(coupling["target_infos"]),
        "target_distances_um": distances,
        "weight_uS": float(coupling["weight_uS"]),
        "tau1_ms": float(coupling["tau1_ms"]),
        "tau2_ms": float(coupling["tau2_ms"]),
        "reversal_mV": float(coupling["reversal_mV"]),
        "delay_ms": float(coupling["delay_ms"]),
        "depression_enabled": bool(coupling.get("depression_enabled", False)),
        "depression_u": float(coupling.get("depression_u", np.nan)),
        "depression_tau_rec_ms": float(coupling.get("depression_tau_rec_ms", np.nan)),
        "depression_tau_facil_ms": float(coupling.get("depression_tau_facil_ms", np.nan)),
    }


def selected_node_voltage_indices(cell) -> list[int]:
    n_nodes = len(getattr(cell, "node", []))
    if n_nodes <= 1:
        return []
    candidates = [1, n_nodes // 2, max(0, n_nodes - 2)]
    return sorted({idx for idx in candidates if 0 <= idx < n_nodes})


def run_soma_simulation(
    cell,
    cell_index: int = 0,
    tstop_ms: float = 1500.0,
    amp_nA: float = 0.0,
    delay_ms: float = 500.0,
    dur_ms: float = 500.0,
    temperature_c: float = 37.0,
    dbs_config: dict | None = None,
    hdp_config: dict | None = None,
    gpe_config: dict | None = None,
    rng: np.random.Generator | None = None,
    gpe_rng: np.random.Generator | None = None,
):
    h.dt = 0.025
    h.celsius = temperature_c
    if rng is None:
        rng = np.random.default_rng()
    if gpe_rng is None:
        gpe_rng = np.random.default_rng()

    soma_sec = soma_section(cell)
    stim = h.IClamp(soma_sec(0.5))
    stim.delay = delay_ms
    stim.dur = dur_ms
    stim.amp = amp_nA

    soma_ref = soma_sec(0.5)
    t_vec = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(soma_ref._ref_v)
    v_initseg = h.Vector().record(cell.initseg(0.5)._ref_v)
    v_node0 = h.Vector().record(cell.node[0](0.5)._ref_v)
    selected_node_v = {
        f"node{idx}_v_mV": h.Vector().record(cell.node[idx](0.5)._ref_v)
        for idx in selected_node_voltage_indices(cell)
    }

    apc = h.APCount(soma_ref)
    apc.thresh = 0.0
    spike_times = h.Vector()
    apc.record(spike_times)
    activation_apcounts = []
    activation_vectors = []
    activation_sites = []
    for site in activation_recording_sites(cell):
        detector = h.APCount(site["segment"])
        detector.thresh = 0.0
        times = h.Vector()
        detector.record(times)
        activation_apcounts.append(detector)
        activation_vectors.append(times)
        activation_sites.append({key: site[key] for key in ("name", "kind", "distance_um")})

    hdp_coupling = setup_coupled_hdp_synapses(
        cell=cell,
        cell_index=cell_index,
        hdp_config=hdp_config,
        dbs_config=dbs_config,
        rng=rng,
    )
    gpe_coupling = setup_coupled_gpe_synapses(
        cell=cell,
        cell_index=cell_index,
        gpe_config=gpe_config,
        dbs_config=dbs_config,
        rng=gpe_rng,
    )

    h.finitialize(-60.0)
    if dbs_config and dbs_config.get("enabled", False):
        while h.t < tstop_ms:
            I_uA = dbs_pulse_uA(float(h.t), dbs_config)
            apply_bipolar_phi_to_cell(cell, I_uA, dbs_config)
            if hdp_coupling.get("enabled", False):
                for axon in hdp_coupling["axons"]:
                    apply_bipolar_phi_to_cell(axon, I_uA, dbs_config)
            if gpe_coupling.get("enabled", False):
                for axon in gpe_coupling["axons"]:
                    apply_bipolar_phi_to_cell(axon, I_uA, dbs_config)
            h.fadvance()
    else:
        h.continuerun(tstop_ms * ms)

    coupled_hdp_results = materialize_coupled_hdp_results(hdp_coupling, t_vec, dbs_config, tstop_ms)
    coupled_gpe_results = materialize_coupled_gpe_results(gpe_coupling, t_vec, dbs_config, tstop_ms)
    result = {
        "t_ms": np.array(t_vec),
        "soma_v_mV": np.array(v_soma),
        "initseg_v_mV": np.array(v_initseg),
        "node0_v_mV": np.array(v_node0),
        "dbs_uA": None
        if not (dbs_config and dbs_config.get("enabled", False))
        else np.array([dbs_pulse_uA(float(t_ms), dbs_config) for t_ms in np.array(t_vec)]),
        "dbs_plot_uA": None
        if not (dbs_config and dbs_config.get("enabled", False))
        else -np.array([dbs_pulse_uA(float(t_ms), dbs_config) for t_ms in np.array(t_vec)]),
        "spike_count": int(apc.n),
        "spike_times_ms": np.array(spike_times),
    }
    for key, vec in selected_node_v.items():
        result[key] = np.array(vec)
    result["activation_spike_times_ms"] = {
        site["name"]: np.array(times)
        for site, times in zip(activation_sites, activation_vectors)
    }
    result["activation_sites"] = activation_sites
    result["hdp_coupled_results"] = coupled_hdp_results
    result["hdp_synapse_summary"] = summarize_coupled_hdp_synapses(hdp_coupling)
    result["gpe_coupled_results"] = coupled_gpe_results
    result["gpe_synapse_summary"] = summarize_coupled_gpe_synapses(gpe_coupling)
    # Keep APCount objects alive until after vectors are materialized above.
    _ = activation_apcounts
    _ = hdp_coupling
    _ = gpe_coupling
    return result


def run_population_simulation(
    *,
    n_cells: int,
    model: str,
    base_params: Iterable[float],
    morphology: str | Path | None,
    tstop_ms: float,
    amp_nA: float,
    delay_ms: float,
    dur_ms: float,
    temperature_c: float,
    param_jitter_frac: float = 0.0,
    seed: int | None = None,
    dbs_config: dict | None = None,
    hdp_config: dict | None = None,
    gpe_config: dict | None = None,
):
    rng = np.random.default_rng(seed)
    results = []
    base_params = list(base_params)

    for cell_index in range(n_cells):
        params = jitter_parameter_vector(base_params, param_jitter_frac, rng)
        cell = create_2024_cell(
            model=model,
            params=params,
            morphology=morphology,
            cell_index=cell_index,
        )
        placement = build_cell_placement(cell, cell_index, dbs_config or DEFAULT_DBS_CONFIG, rng)
        cache_segment_positions(cell, placement)
        if seed is None:
            hdp_cell_rng = np.random.default_rng()
            gpe_cell_rng = np.random.default_rng()
        else:
            hdp_cell_rng = np.random.default_rng(
                int(seed)
                + int((hdp_config or DEFAULT_HDP_CONFIG).get("seed_offset", 0))
                + 1000003 * int(cell_index)
            )
            gpe_cell_rng = np.random.default_rng(
                int(seed)
                + int((gpe_config or DEFAULT_GPE_CONFIG).get("seed_offset", 0))
                + 1000003 * int(cell_index)
            )
        result = run_soma_simulation(
            cell,
            cell_index=cell_index,
            tstop_ms=tstop_ms,
            amp_nA=amp_nA,
            delay_ms=delay_ms,
            dur_ms=dur_ms,
            temperature_c=temperature_c,
            dbs_config=dbs_config,
            hdp_config=hdp_config,
            gpe_config=gpe_config,
            rng=hdp_cell_rng,
            gpe_rng=gpe_cell_rng,
        )
        result["cell_index"] = cell_index
        result["params"] = params
        result["morphology"] = str(resolve_morphology(morphology, cell_index=cell_index)) if model == "detail" else None
        result["placement"] = placement
        soma_pos = placement["soma_pos_mm"]
        result["dist_to_electrode_a_mm"] = dist_mm(soma_pos, (dbs_config or DEFAULT_DBS_CONFIG)["electrode_a_pos_mm"])
        result["dist_to_electrode_b_mm"] = dist_mm(soma_pos, (dbs_config or DEFAULT_DBS_CONFIG)["electrode_b_pos_mm"])
        result["dist_to_fiber_center_mm"] = dist_mm(soma_pos, (dbs_config or DEFAULT_DBS_CONFIG)["fiber_center_mm"])
        positions, phi_mV, kinds = snapshot_segment_geometry(cell, dbs_config or DEFAULT_DBS_CONFIG)
        result["segment_positions_mm"] = positions
        result["segment_phi_mV"] = phi_mV
        result["segment_kinds"] = kinds
        result.update(compute_result_dbs_metrics(result, dbs_config))
        result["activation_origin"] = classify_activation_origin(result, dbs_config)
        results.append(result)

    return results


def first_spike_after(times_ms: np.ndarray, threshold_ms: float) -> float:
    times_ms = np.asarray(times_ms, dtype=float)
    times_ms = times_ms[np.isfinite(times_ms) & (times_ms >= float(threshold_ms))]
    if len(times_ms) == 0:
        return np.nan
    return float(times_ms[0])


def run_hdp_population_simulation(
    *,
    hdp_config: dict,
    dbs_config: dict | None,
    tstop_ms: float,
    temperature_c: float,
    seed: int | None = None,
) -> list[dict]:
    if not hdp_config.get("enabled", False) or int(hdp_config.get("n_axons", 0)) <= 0:
        return []
    if not dbs_config or not dbs_config.get("enabled", False):
        return []

    rng_seed = None if seed is None else int(seed) + int(hdp_config.get("seed_offset", 0))
    rng = np.random.default_rng(rng_seed)
    axons = [
        ParametricHDPAxon(axon_index=i, hdp_config=hdp_config, dbs_config=dbs_config, rng=rng)
        for i in range(int(hdp_config["n_axons"]))
    ]
    for axon in axons:
        cache_hdp_segment_positions(axon)

    h.dt = 0.025
    h.celsius = temperature_c
    t_vec = h.Vector().record(h._ref_t)
    recorders = []
    for axon in axons:
        for site in axon.recording_sites():
            detector = h.APCount(site["segment"])
            detector.thresh = 0.0
            times = h.Vector()
            detector.record(times)
            recorders.append(
                {
                    "axon": axon,
                    "site_name": site["name"],
                    "kind": site["kind"],
                    "node_index": int(site["node_index"]),
                    "detector": detector,
                    "times": times,
                }
            )

    h.finitialize(-60.0)
    while h.t < float(tstop_ms):
        I_uA = dbs_pulse_uA(float(h.t), dbs_config)
        for axon in axons:
            apply_bipolar_phi_to_cell(axon, I_uA, dbs_config)
        h.fadvance()

    pulse_times = dbs_first_phase_pulse_times_ms(0.0, float(tstop_ms), dbs_config)
    first_pulse = float(pulse_times[0]) if len(pulse_times) else float(dbs_config["start_ms"])

    results = []
    for axon in axons:
        site_spikes = {}
        site_firsts = []
        for rec in recorders:
            if rec["axon"] is not axon:
                continue
            spikes = np.asarray(rec["times"], dtype=float)
            site_spikes[rec["site_name"]] = spikes
            first = first_spike_after(spikes, first_pulse)
            if np.isfinite(first):
                site_firsts.append((first, rec["site_name"], rec["kind"], rec["node_index"]))

        if site_firsts:
            first_time, first_site, first_kind, first_node = min(site_firsts, key=lambda item: item[0])
            activated = True
            first_latency = float(first_time - first_pulse)
        else:
            first_time = np.nan
            first_site = "none"
            first_kind = "none"
            first_node = -1
            activated = False
            first_latency = np.nan

        terminal_name = f"collateral_node{len(axon.collateral_nodes) - 1}"
        cortical_name = "parent_node0"
        distal_parent_name = f"parent_node{len(axon.parent_nodes) - 1}"
        terminal_first = first_spike_after(site_spikes.get(terminal_name, np.array([])), first_pulse)
        cortical_first = first_spike_after(site_spikes.get(cortical_name, np.array([])), first_pulse)
        distal_parent_first = first_spike_after(site_spikes.get(distal_parent_name, np.array([])), first_pulse)

        geom = snapshot_hdp_geometry(axon, dbs_config)
        result = {
            "axon_index": int(axon.axon_index),
            "activated": bool(activated),
            "first_spike_time_ms": float(first_time) if np.isfinite(first_time) else np.nan,
            "first_pulse_latency_ms": first_latency,
            "first_spike_site": first_site,
            "first_spike_kind": first_kind,
            "first_spike_node_index": int(first_node),
            "terminal_arrival_ms": float(terminal_first - first_pulse) if np.isfinite(terminal_first) else np.nan,
            "cortical_end_arrival_ms": float(cortical_first - first_pulse) if np.isfinite(cortical_first) else np.nan,
            "distal_parent_arrival_ms": float(distal_parent_first - first_pulse) if np.isfinite(distal_parent_first) else np.nan,
            "site_spike_times_ms": site_spikes,
            "t_ms": np.asarray(t_vec, dtype=float),
            "dbs_uA": np.array([dbs_pulse_uA(float(t_ms), dbs_config) for t_ms in np.asarray(t_vec, dtype=float)]),
        }
        result.update(geom)
        results.append(result)

    # Keep detectors alive until vectors have been materialized above.
    _ = recorders
    return results


def summarize_hdp_results(hdp_results: list[dict]) -> dict:
    if not hdp_results:
        return {
            "n_axons": 0,
            "n_activated": 0,
            "activated_fraction": np.nan,
            "first_latency_mean_ms": np.nan,
            "terminal_arrival_mean_ms": np.nan,
            "synaptic_onset_mean_ms": np.nan,
            "synaptic_peak_mean_ms": np.nan,
        }
    activated = np.array([bool(result.get("activated", False)) for result in hdp_results], dtype=bool)
    first_latencies = np.array(
        [
            float(result.get("first_pulse_latency_ms", np.nan))
            for result in hdp_results
            if np.isfinite(result.get("first_pulse_latency_ms", np.nan))
        ],
        dtype=float,
    )
    terminal_latencies = np.array(
        [
            float(result.get("terminal_arrival_ms", np.nan))
            for result in hdp_results
            if np.isfinite(result.get("terminal_arrival_ms", np.nan))
        ],
        dtype=float,
    )
    synaptic_onsets = np.array(
        [
            float(result.get("synaptic_onset_ms", np.nan))
            for result in hdp_results
            if np.isfinite(result.get("synaptic_onset_ms", np.nan))
        ],
        dtype=float,
    )
    synaptic_peaks = np.array(
        [
            float(result.get("synaptic_peak_ms", np.nan))
            for result in hdp_results
            if np.isfinite(result.get("synaptic_peak_ms", np.nan))
        ],
        dtype=float,
    )
    return {
        "n_axons": int(len(hdp_results)),
        "n_activated": int(np.count_nonzero(activated)),
        "activated_fraction": float(np.mean(activated)) if len(activated) else np.nan,
        "first_latency_mean_ms": float(np.mean(first_latencies)) if len(first_latencies) else np.nan,
        "terminal_arrival_mean_ms": float(np.mean(terminal_latencies)) if len(terminal_latencies) else np.nan,
        "synaptic_onset_mean_ms": float(np.mean(synaptic_onsets)) if len(synaptic_onsets) else np.nan,
        "synaptic_peak_mean_ms": float(np.mean(synaptic_peaks)) if len(synaptic_peaks) else np.nan,
    }


def print_hdp_summary(hdp_results: list[dict]) -> None:
    if not hdp_results:
        return
    summary = summarize_hdp_results(hdp_results)
    print(
        "HDP axons: "
        f"n={summary['n_axons']}, "
        f"activated={summary['n_activated']} ({100.0 * summary['activated_fraction']:.1f}%), "
        f"first_latency_mean={_format_metric(summary['first_latency_mean_ms'], ' ms')}, "
        f"terminal_arrival_mean={_format_metric(summary['terminal_arrival_mean_ms'], ' ms')}, "
        f"synaptic_onset_mean={_format_metric(summary['synaptic_onset_mean_ms'], ' ms')}, "
        f"synaptic_peak_mean={_format_metric(summary['synaptic_peak_mean_ms'], ' ms')}"
    )


def print_gpe_summary(gpe_results: list[dict]) -> None:
    if not gpe_results:
        return
    summary = summarize_hdp_results(gpe_results)
    print(
        "GPe axons: "
        f"n={summary['n_axons']}, "
        f"activated={summary['n_activated']} ({100.0 * summary['activated_fraction']:.1f}%), "
        f"first_latency_mean={_format_metric(summary['first_latency_mean_ms'], ' ms')}, "
        f"terminal_arrival_mean={_format_metric(summary['terminal_arrival_mean_ms'], ' ms')}, "
        f"synaptic_onset_mean={_format_metric(summary['synaptic_onset_mean_ms'], ' ms')}, "
        f"synaptic_peak_mean={_format_metric(summary['synaptic_peak_mean_ms'], ' ms')}"
    )


def collect_coupled_hdp_results(results: list[dict]) -> list[dict]:
    hdp_results = []
    for result in results:
        syn_delay_ms = float(
            result.get("hdp_synapse_summary", {}).get(
                "delay_ms",
                DEFAULT_HDP_CONFIG.get("syn_delay_ms", np.nan),
            )
        )
        syn_peak_offset_ms = exp2syn_peak_offset_ms(
            result.get("hdp_synapse_summary", {}).get(
                "tau1_ms",
                DEFAULT_HDP_CONFIG.get("syn_tau1_ms", np.nan),
            ),
            result.get("hdp_synapse_summary", {}).get(
                "tau2_ms",
                DEFAULT_HDP_CONFIG.get("syn_tau2_ms", np.nan),
            ),
        )
        for hdp_result in result.get("hdp_coupled_results", []):
            hdp_result = dict(hdp_result)
            terminal_arrival_ms = float(hdp_result.get("terminal_arrival_ms", np.nan))
            hdp_result["synaptic_onset_ms"] = (
                terminal_arrival_ms + syn_delay_ms if np.isfinite(terminal_arrival_ms) else np.nan
            )
            hdp_result["synaptic_peak_ms"] = (
                hdp_result["synaptic_onset_ms"] + syn_peak_offset_ms
                if np.isfinite(hdp_result["synaptic_onset_ms"]) and np.isfinite(syn_peak_offset_ms)
                else np.nan
            )
            hdp_results.append(hdp_result)
    return hdp_results


def collect_coupled_gpe_results(results: list[dict]) -> list[dict]:
    gpe_results = []
    for result in results:
        syn_delay_ms = float(
            result.get("gpe_synapse_summary", {}).get(
                "delay_ms",
                DEFAULT_GPE_CONFIG.get("syn_delay_ms", np.nan),
            )
        )
        syn_peak_offset_ms = exp2syn_peak_offset_ms(
            result.get("gpe_synapse_summary", {}).get(
                "tau1_ms",
                DEFAULT_GPE_CONFIG.get("syn_tau1_ms", np.nan),
            ),
            result.get("gpe_synapse_summary", {}).get(
                "tau2_ms",
                DEFAULT_GPE_CONFIG.get("syn_tau2_ms", np.nan),
            ),
        )
        for gpe_result in result.get("gpe_coupled_results", []):
            gpe_result = dict(gpe_result)
            terminal_arrival_ms = float(gpe_result.get("terminal_arrival_ms", np.nan))
            gpe_result["synaptic_onset_ms"] = (
                terminal_arrival_ms + syn_delay_ms if np.isfinite(terminal_arrival_ms) else np.nan
            )
            gpe_result["synaptic_peak_ms"] = (
                gpe_result["synaptic_onset_ms"] + syn_peak_offset_ms
                if np.isfinite(gpe_result["synaptic_onset_ms"]) and np.isfinite(syn_peak_offset_ms)
                else np.nan
            )
            gpe_results.append(gpe_result)
    return gpe_results


def summarize_hdp_synapses_from_results(results: list[dict]) -> dict:
    summaries = [
        result.get("hdp_synapse_summary", {})
        for result in results
        if result.get("hdp_synapse_summary", {}).get("enabled", False)
    ]
    if not summaries:
        return {
            "enabled": False,
            "n_synapses": 0,
            "n_cells": 0,
            "target_distances_um": np.array([], dtype=float),
        }

    distances = np.concatenate(
        [
            np.asarray(summary.get("target_distances_um", np.array([], dtype=float)), dtype=float)
            for summary in summaries
        ]
    )
    cutoffs = np.array(
        [
            float(summary.get("cutoff_um", np.nan))
            for summary in summaries
            if np.isfinite(summary.get("cutoff_um", np.nan))
        ],
        dtype=float,
    )
    n_synapses = int(sum(int(summary.get("n_synapses", 0)) for summary in summaries))
    n_axons = int(sum(int(summary.get("n_axons", 0)) for summary in summaries))
    fallback_count = int(sum(bool(summary.get("fallback_to_all_dendrites", False)) for summary in summaries))
    first = summaries[0]
    return {
        "enabled": True,
        "n_synapses": n_synapses,
        "n_axons": n_axons,
        "n_cells": int(len(summaries)),
        "target": first.get("target", "unknown"),
        "cutoff_mean_um": float(np.mean(cutoffs)) if len(cutoffs) else np.nan,
        "target_distances_um": distances,
        "target_distance_mean_um": float(np.mean(distances)) if len(distances) else np.nan,
        "target_distance_min_um": float(np.min(distances)) if len(distances) else np.nan,
        "target_distance_max_um": float(np.max(distances)) if len(distances) else np.nan,
        "fallback_count": fallback_count,
        "weight_uS": float(first.get("weight_uS", np.nan)),
        "tau1_ms": float(first.get("tau1_ms", np.nan)),
        "tau2_ms": float(first.get("tau2_ms", np.nan)),
        "delay_ms": float(first.get("delay_ms", np.nan)),
        "depression_enabled": bool(first.get("depression_enabled", False)),
        "depression_u": float(first.get("depression_u", np.nan)),
        "depression_tau_rec_ms": float(first.get("depression_tau_rec_ms", np.nan)),
        "depression_tau_facil_ms": float(first.get("depression_tau_facil_ms", np.nan)),
    }


def print_hdp_synapse_summary(results: list[dict]) -> None:
    summary = summarize_hdp_synapses_from_results(results)
    if not summary.get("enabled", False):
        return

    print(
        "HDP synapses: "
        f"n={summary['n_synapses']} over {summary['n_cells']} cells, "
        f"axons_simulated={summary['n_axons']}, "
        f"target={summary['target']}, "
        f"cutoff_mean={_format_metric(summary['cutoff_mean_um'], ' um')}, "
        "target_distance_mean/min/max="
        f"{_format_metric(summary['target_distance_mean_um'], ' um')} / "
        f"{_format_metric(summary['target_distance_min_um'], ' um')} / "
        f"{_format_metric(summary['target_distance_max_um'], ' um')}, "
        f"weight={_format_metric(summary['weight_uS'], ' uS', digits=4)}, "
        f"tau={_format_metric(summary['tau1_ms'], ' ms', digits=2)}/"
        f"{_format_metric(summary['tau2_ms'], ' ms', digits=2)}, "
        f"delay={_format_metric(summary['delay_ms'], ' ms', digits=2)}"
        + (
            ", depression="
            f"U={_format_metric(summary['depression_u'], digits=2)}, "
            f"tau_rec={_format_metric(summary['depression_tau_rec_ms'], ' ms', digits=1)}, "
            f"tau_facil={_format_metric(summary['depression_tau_facil_ms'], ' ms', digits=1)}"
            if summary.get("depression_enabled", False)
            else ", depression=off"
        )
    )
    if summary["fallback_count"] > 0:
        print(
            "HDP synapse note: "
            f"{summary['fallback_count']} cells had no distal candidates and used all dendrites instead."
        )


def summarize_gpe_synapses_from_results(results: list[dict]) -> dict:
    summaries = [
        result.get("gpe_synapse_summary", {})
        for result in results
        if result.get("gpe_synapse_summary", {}).get("enabled", False)
    ]
    if not summaries:
        return {
            "enabled": False,
            "n_synapses": 0,
            "n_cells": 0,
            "target_distances_um": np.array([], dtype=float),
        }

    distances = np.concatenate(
        [
            np.asarray(summary.get("target_distances_um", np.array([], dtype=float)), dtype=float)
            for summary in summaries
        ]
    )
    proximal_cutoffs = np.array(
        [
            float(summary.get("proximal_cutoff_um", np.nan))
            for summary in summaries
            if np.isfinite(summary.get("proximal_cutoff_um", np.nan))
        ],
        dtype=float,
    )
    distal_cutoffs = np.array(
        [
            float(summary.get("distal_cutoff_um", np.nan))
            for summary in summaries
            if np.isfinite(summary.get("distal_cutoff_um", np.nan))
        ],
        dtype=float,
    )
    target_counts = {}
    for summary in summaries:
        for key, value in summary.get("target_counts", {}).items():
            target_counts[str(key)] = target_counts.get(str(key), 0) + int(value)

    n_synapses = int(sum(int(summary.get("n_synapses", 0)) for summary in summaries))
    n_axons = int(sum(int(summary.get("n_axons", 0)) for summary in summaries))
    fallback_count = int(
        sum(
            any(bool(value) for value in summary.get("fallback_to_all_dendrites", {}).values())
            for summary in summaries
        )
    )
    first = summaries[0]
    return {
        "enabled": True,
        "n_synapses": n_synapses,
        "n_axons": n_axons,
        "n_cells": int(len(summaries)),
        "inputs_per_cell": int(first.get("inputs_per_cell", 0)),
        "contacts_per_input": int(first.get("contacts_per_input", 0)),
        "target_counts": target_counts,
        "proximal_cutoff_mean_um": float(np.mean(proximal_cutoffs)) if len(proximal_cutoffs) else np.nan,
        "distal_cutoff_mean_um": float(np.mean(distal_cutoffs)) if len(distal_cutoffs) else np.nan,
        "target_distances_um": distances,
        "target_distance_mean_um": float(np.mean(distances)) if len(distances) else np.nan,
        "target_distance_min_um": float(np.min(distances)) if len(distances) else np.nan,
        "target_distance_max_um": float(np.max(distances)) if len(distances) else np.nan,
        "fallback_count": fallback_count,
        "weight_uS": float(first.get("weight_uS", np.nan)),
        "tau1_ms": float(first.get("tau1_ms", np.nan)),
        "tau2_ms": float(first.get("tau2_ms", np.nan)),
        "reversal_mV": float(first.get("reversal_mV", np.nan)),
        "delay_ms": float(first.get("delay_ms", np.nan)),
        "depression_enabled": bool(first.get("depression_enabled", False)),
        "depression_u": float(first.get("depression_u", np.nan)),
        "depression_tau_rec_ms": float(first.get("depression_tau_rec_ms", np.nan)),
        "depression_tau_facil_ms": float(first.get("depression_tau_facil_ms", np.nan)),
    }


def print_gpe_synapse_summary(results: list[dict]) -> None:
    summary = summarize_gpe_synapses_from_results(results)
    if not summary.get("enabled", False):
        return

    n_synapses = max(1, int(summary["n_synapses"]))
    target_bits = []
    for key in ("proximal", "soma", "distal"):
        count = int(summary.get("target_counts", {}).get(key, 0))
        target_bits.append(f"{key}={count} ({100.0 * count / n_synapses:.1f}%)")

    print(
        "GPe synapses: "
        f"n={summary['n_synapses']} over {summary['n_cells']} cells, "
        f"axons_simulated={summary['n_axons']}, "
        f"inputs/cell={summary['inputs_per_cell']}, "
        f"contacts/input={summary['contacts_per_input']}, "
        "targets="
        + ", ".join(target_bits)
        + ", "
        f"prox/dist cutoffs={_format_metric(summary['proximal_cutoff_mean_um'], ' um')} / "
        f"{_format_metric(summary['distal_cutoff_mean_um'], ' um')}, "
        "target_distance_mean/min/max="
        f"{_format_metric(summary['target_distance_mean_um'], ' um')} / "
        f"{_format_metric(summary['target_distance_min_um'], ' um')} / "
        f"{_format_metric(summary['target_distance_max_um'], ' um')}, "
        f"weight={_format_metric(summary['weight_uS'], ' uS', digits=4)}, "
        f"tau={_format_metric(summary['tau1_ms'], ' ms', digits=2)}/"
        f"{_format_metric(summary['tau2_ms'], ' ms', digits=2)}, "
        f"E={_format_metric(summary['reversal_mV'], ' mV', digits=1)}, "
        f"delay={_format_metric(summary['delay_ms'], ' ms', digits=2)}"
        + (
            ", depression="
            f"U={_format_metric(summary['depression_u'], digits=2)}, "
            f"tau_rec={_format_metric(summary['depression_tau_rec_ms'], ' ms', digits=1)}, "
            f"tau_facil={_format_metric(summary['depression_tau_facil_ms'], ' ms', digits=1)}"
            if summary.get("depression_enabled", False)
            else ", depression=off"
        )
    )
    if summary["fallback_count"] > 0:
        print(
            "GPe synapse note: "
            f"{summary['fallback_count']} cells had empty proximal/distal pools and used all dendrites for that pool."
        )


def summarize_stn_local_axon_activation(results: list[dict], dbs_config: dict | None) -> dict[str, float]:
    if not results or not dbs_config or not dbs_config.get("enabled", False):
        return {
            "n_cells": 0,
            "n_activated": 0,
            "activated_fraction": np.nan,
            "n_sub1_ms": 0,
            "n_1to5_ms": 0,
            "n_gt5_ms": 0,
            "latency_mean_ms": np.nan,
            "latency_min_ms": np.nan,
            "latency_max_ms": np.nan,
        }

    latencies = np.array(
        [first_stn_local_axon_latency_after_first_pulse_ms(result, dbs_config) for result in results],
        dtype=float,
    )
    active = latencies[np.isfinite(latencies)]
    n_cells = int(len(results))
    return {
        "n_cells": n_cells,
        "n_activated": int(len(active)),
        "activated_fraction": float(len(active) / n_cells) if n_cells else np.nan,
        "n_sub1_ms": int(np.count_nonzero(active < 1.0)),
        "n_1to5_ms": int(np.count_nonzero((active >= 1.0) & (active <= 5.0))),
        "n_gt5_ms": int(np.count_nonzero(active > 5.0)),
        "latency_mean_ms": float(np.mean(active)) if len(active) else np.nan,
        "latency_min_ms": float(np.min(active)) if len(active) else np.nan,
        "latency_max_ms": float(np.max(active)) if len(active) else np.nan,
    }


def print_stn_local_axon_activation_summary(results: list[dict], dbs_config: dict | None) -> None:
    summary = summarize_stn_local_axon_activation(results, dbs_config)
    if int(summary["n_cells"]) <= 0:
        return

    print(
        "STN local axons: "
        f"activated={summary['n_activated']}/{summary['n_cells']} "
        f"({100.0 * summary['activated_fraction']:.1f}%), "
        f"<1ms={summary['n_sub1_ms']}/{summary['n_cells']}, "
        f"1-5ms={summary['n_1to5_ms']}/{summary['n_cells']}, "
        f">5ms={summary['n_gt5_ms']}/{summary['n_cells']}, "
        "latency_mean/min/max="
        f"{_format_metric(summary['latency_mean_ms'], ' ms')} / "
        f"{_format_metric(summary['latency_min_ms'], ' ms')} / "
        f"{_format_metric(summary['latency_max_ms'], ' ms')}"
    )


def finite_metric_values(values: Iterable[float]) -> np.ndarray:
    arr = np.array([float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def format_median_iqr(values: Iterable[float], unit: str = "", digits: int = 2) -> str:
    arr = finite_metric_values(values)
    if len(arr) == 0:
        return "n/a"
    q25, median, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return f"{median:.{digits}f} [{q25:.{digits}f}-{q75:.{digits}f}]{unit}"


def format_mean_sd(values: Iterable[float], unit: str = "", digits: int = 2) -> str:
    arr = finite_metric_values(values)
    if len(arr) == 0:
        return "n/a"
    return f"{np.mean(arr):.{digits}f} +/- {np.std(arr):.{digits}f}{unit}"


def print_paper_style_summary(
    results: list[dict],
    dbs_config: dict | None,
    hdp_results: list[dict],
    gpe_results: list[dict],
) -> None:
    if not results:
        return

    n_cells = len(results)
    during_rates = finite_metric_values(result.get("during_dbs_rate_hz", np.nan) for result in results)
    early_rates = finite_metric_values(result.get("early_dbs_rate_hz", np.nan) for result in results)
    late_rates = finite_metric_values(result.get("late_dbs_rate_hz", np.nan) for result in results)
    post_rates = finite_metric_values(result.get("post_dbs_rate_hz", np.nan) for result in results)
    first_soma_latencies = finite_metric_values(result.get("first_pulse_latency_ms", np.nan) for result in results)
    plvs = finite_metric_values(result.get("pulse_plv", np.nan) for result in results)
    local_axon_latencies = finite_metric_values(
        first_stn_local_axon_latency_after_first_pulse_ms(result, dbs_config)
        for result in results
    )

    if dbs_config and dbs_config.get("enabled", False):
        stim_freq_hz = float(dbs_config["freq_hz"])
        entrain_threshold_hz = 0.90 * stim_freq_hz
        entrained = int(np.count_nonzero(during_rates >= entrain_threshold_hz))
        entrain_label = (
            f"{entrained}/{n_cells} ({100.0 * entrained / n_cells:.1f}%; "
            f">=90% DBS freq, {entrain_threshold_hz:.1f} Hz)"
        )
        dbs_label = (
            f"{float(dbs_config['amp_uA']):.2f} uA, "
            f"{stim_freq_hz:.2f} Hz, "
            f"PW {float(dbs_config['pw_ms']):.3f} ms"
        )
        if dbs_config.get("stop_ms", None) is not None:
            dbs_label += f", stop {float(dbs_config['stop_ms']):.1f} ms"
    else:
        entrain_label = "n/a"
        dbs_label = "DBS disabled"

    def pathway_summary(label: str, pathway_results: list[dict]) -> str:
        n_axons = len(pathway_results)
        active_results = [result for result in pathway_results if bool(result.get("activated", False))]
        n_active = len(active_results)
        if n_axons <= 0:
            return f"{label}: not simulated"
        terminal_arrivals = [
            float(result.get("terminal_arrival_ms", np.nan))
            for result in active_results
        ]
        synaptic_peaks = [
            float(result.get("synaptic_peak_ms", np.nan))
            for result in active_results
        ]
        return (
            f"{label}: activated={n_active}/{n_axons} ({100.0 * n_active / n_axons:.1f}%), "
            f"terminal={format_median_iqr(terminal_arrivals, ' ms')}, "
            f"syn_peak={format_median_iqr(synaptic_peaks, ' ms')}"
        )

    n_local_active = int(len(local_axon_latencies))
    n_local_sub1 = int(np.count_nonzero(local_axon_latencies < 1.0))

    print("Paper-style summary:")
    print(f"DBS condition: {dbs_label}, n={n_cells} STN cells")
    print(
        "STN soma response: "
        f"DBS-rate={format_median_iqr(during_rates, ' Hz')} "
        f"(mean +/- SD {format_mean_sd(during_rates, ' Hz')}), "
        f"early={format_median_iqr(early_rates, ' Hz')}, "
        f"late={format_median_iqr(late_rates, ' Hz')}, "
        f"post-rate={format_median_iqr(post_rates, ' Hz')}, "
        f"first-soma-latency={format_median_iqr(first_soma_latencies, ' ms')}, "
        f"entrained={entrain_label}"
    )
    print(
        "DBS locking: "
        f"PLV={format_median_iqr(plvs, '')} "
        f"(mean +/- SD {format_mean_sd(plvs, '')})"
    )
    print(
        "STN local axons: "
        f"activated={n_local_active}/{n_cells} ({100.0 * n_local_active / n_cells:.1f}%), "
        f"<1ms={n_local_sub1}/{n_cells}, "
        f"latency={format_median_iqr(local_axon_latencies, ' ms')}"
    )
    print(pathway_summary("HDP axons", hdp_results))
    print(pathway_summary("GPe axons", gpe_results))


def summarize_population_results(results: list[dict]) -> dict[str, float]:
    spike_counts = np.array([result["spike_count"] for result in results], dtype=float)
    soma_min = min(float(np.min(result["soma_v_mV"])) for result in results)
    soma_max = max(float(np.max(result["soma_v_mV"])) for result in results)
    return {
        "n_cells": len(results),
        "mean_spikes": float(np.mean(spike_counts)),
        "min_spikes": float(np.min(spike_counts)),
        "max_spikes": float(np.max(spike_counts)),
        "soma_min_mV": soma_min,
        "soma_max_mV": soma_max,
    }


def dbs_first_phase_pulse_times_ms(
    t_start_ms: float,
    t_stop_ms: float,
    dbs_config: dict | None,
) -> np.ndarray:
    if not dbs_config or not dbs_config.get("enabled", False):
        return np.array([], dtype=float)

    freq_hz = float(dbs_config["freq_hz"])
    if freq_hz <= 0.0:
        return np.array([], dtype=float)

    stim_start_ms = float(dbs_config["start_ms"])
    if stim_start_ms > t_stop_ms:
        return np.array([], dtype=float)

    period_ms = 1000.0 / freq_hz
    stim_stop_ms = dbs_config.get("stop_ms", None)
    pulse_stop_ms = float(t_stop_ms)
    if stim_stop_ms is not None:
        pulse_stop_ms = min(pulse_stop_ms, float(stim_stop_ms))
    if stim_start_ms >= pulse_stop_ms:
        return np.array([], dtype=float)

    pulses = np.arange(stim_start_ms, pulse_stop_ms + period_ms * 0.5, period_ms, dtype=float)
    keep = (pulses >= t_start_ms) & (pulses <= t_stop_ms)
    if stim_stop_ms is not None:
        keep &= pulses < float(stim_stop_ms)
    omit_pulse = dbs_config.get("omit_pulse", None)
    if omit_pulse is not None and int(omit_pulse) > 0:
        pulse_numbers = np.arange(1, len(pulses) + 1, dtype=int)
        keep &= pulse_numbers != int(omit_pulse)
    return pulses[keep]


def compute_hilbert_plv_at_pulses(
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    pulse_times_ms: np.ndarray,
    freq_hz: float,
    bandwidth_hz: float = 1.0,
) -> dict:
    if len(time_ms) < 8 or len(pulse_times_ms) == 0:
        return {
            "plv": np.nan,
            "sampled_phases_rad": np.array([], dtype=float),
            "sampled_pulse_times_ms": np.array([], dtype=float),
            "band_hz": (freq_hz - bandwidth_hz, freq_hz + bandwidth_hz),
        }

    dt_ms = float(np.median(np.diff(time_ms)))
    fs_hz = 1000.0 / dt_ms
    low_hz = max(0.1, float(freq_hz) - float(bandwidth_hz))
    high_hz = min(float(freq_hz) + float(bandwidth_hz), 0.499 * fs_hz)
    if low_hz >= high_hz:
        return {
            "plv": np.nan,
            "sampled_phases_rad": np.array([], dtype=float),
            "sampled_pulse_times_ms": np.array([], dtype=float),
            "band_hz": (low_hz, high_hz),
        }

    centered_v = np.asarray(voltage_mV, dtype=float) - float(np.mean(voltage_mV))
    try:
        from scipy.signal import butter, hilbert, sosfiltfilt

        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
        filtered_v = sosfiltfilt(sos, centered_v)
        analytic = hilbert(filtered_v)
    except ImportError:
        filtered_v = fft_bandpass_trace(centered_v, fs_hz, (low_hz, high_hz))
        if filtered_v is None:
            return {
                "plv": np.nan,
                "sampled_phases_rad": np.array([], dtype=float),
                "sampled_pulse_times_ms": np.array([], dtype=float),
                "band_hz": (low_hz, high_hz),
            }
        analytic = analytic_signal_fft(filtered_v)

    valid_pulses = pulse_times_ms[
        (pulse_times_ms >= float(time_ms[0])) & (pulse_times_ms <= float(time_ms[-1]))
    ]
    if len(valid_pulses) == 0:
        return {
            "plv": np.nan,
            "sampled_phases_rad": np.array([], dtype=float),
            "sampled_pulse_times_ms": np.array([], dtype=float),
            "band_hz": (low_hz, high_hz),
        }

    sample_real = np.interp(valid_pulses, time_ms, np.real(analytic))
    sample_imag = np.interp(valid_pulses, time_ms, np.imag(analytic))
    sample_complex = sample_real + 1j * sample_imag
    sample_amp = np.abs(sample_complex)
    keep = np.isfinite(sample_amp) & (sample_amp > 1e-12)
    if not np.any(keep):
        return {
            "plv": np.nan,
            "sampled_phases_rad": np.array([], dtype=float),
            "sampled_pulse_times_ms": np.array([], dtype=float),
            "band_hz": (low_hz, high_hz),
        }

    sample_complex = sample_complex[keep]
    sample_phases = np.angle(sample_complex)
    plv = float(np.abs(np.mean(np.exp(1j * sample_phases))))
    return {
        "plv": plv,
        "sampled_phases_rad": sample_phases,
        "sampled_pulse_times_ms": valid_pulses[keep],
        "band_hz": (low_hz, high_hz),
    }


def compute_result_dbs_metrics(result: dict, dbs_config: dict | None) -> dict:
    t_ms = np.asarray(result["t_ms"], dtype=float)
    spike_times_ms = np.asarray(result.get("spike_times_ms", []), dtype=float)
    t_stop_ms = float(t_ms[-1]) if len(t_ms) else 0.0

    if not dbs_config or not dbs_config.get("enabled", False):
        return {
            "first_pulse_latency_ms": np.nan,
            "first_spike_after_dbs_start_ms": np.nan,
            "pre_dbs_rate_hz": np.nan,
            "during_dbs_rate_hz": np.nan,
            "early_dbs_rate_hz": np.nan,
            "late_dbs_rate_hz": np.nan,
            "post_dbs_rate_hz": np.nan,
            "pulse_plv": np.nan,
            "pulse_phase_samples_rad": np.array([], dtype=float),
            "pulse_times_ms": np.array([], dtype=float),
            "plv_band_hz": (np.nan, np.nan),
        }

    stim_start_ms = float(dbs_config["start_ms"])
    raw_stop_ms = dbs_config.get("stop_ms", None)
    stim_stop_ms = t_stop_ms if raw_stop_ms is None else float(raw_stop_ms)
    stim_stop_ms = min(max(stim_stop_ms, stim_start_ms), t_stop_ms)
    pulse_times_ms = dbs_first_phase_pulse_times_ms(float(t_ms[0]), t_stop_ms, dbs_config)

    first_latency_ms = np.nan
    first_spike_after_dbs_start_ms = np.nan
    if len(pulse_times_ms) > 0:
        first_pulse_ms = float(pulse_times_ms[0])
        post_first_spikes = spike_times_ms[spike_times_ms >= first_pulse_ms]
        if len(post_first_spikes) > 0:
            first_spike_after_dbs_start_ms = float(post_first_spikes[0] - first_pulse_ms)
            if len(pulse_times_ms) >= 2:
                second_pulse_ms = float(pulse_times_ms[1])
                first_cycle_spikes = post_first_spikes[post_first_spikes < second_pulse_ms]
            else:
                first_cycle_spikes = post_first_spikes
            if len(first_cycle_spikes) > 0:
                first_latency_ms = float(first_cycle_spikes[0] - first_pulse_ms)

    pre_window_ms = max(stim_start_ms - float(t_ms[0]), 0.0)
    during_window_ms = max(stim_stop_ms - stim_start_ms, 0.0)
    post_window_ms = max(t_stop_ms - stim_stop_ms, 0.0) if dbs_config.get("stop_ms", None) is not None else 0.0
    pre_dbs_rate_hz = np.nan
    during_dbs_rate_hz = np.nan
    early_dbs_rate_hz = np.nan
    late_dbs_rate_hz = np.nan
    post_dbs_rate_hz = np.nan
    if pre_window_ms > 0.0:
        pre_spikes = np.count_nonzero(spike_times_ms < stim_start_ms)
        pre_dbs_rate_hz = 1000.0 * float(pre_spikes) / pre_window_ms
    if during_window_ms > 0.0:
        during_spikes = np.count_nonzero(
            (spike_times_ms >= stim_start_ms) & (spike_times_ms < stim_stop_ms)
        )
        during_dbs_rate_hz = 1000.0 * float(during_spikes) / during_window_ms
        edge_window_ms = min(float(DEFAULT_RUN_CONFIG["dbs_rate_edge_window_ms"]), during_window_ms)
        if edge_window_ms > 0.0:
            early_stop_ms = stim_start_ms + edge_window_ms
            late_start_ms = stim_stop_ms - edge_window_ms
            early_spikes = np.count_nonzero(
                (spike_times_ms >= stim_start_ms) & (spike_times_ms < early_stop_ms)
            )
            late_spikes = np.count_nonzero(
                (spike_times_ms >= late_start_ms) & (spike_times_ms < stim_stop_ms)
            )
            early_dbs_rate_hz = 1000.0 * float(early_spikes) / edge_window_ms
            late_dbs_rate_hz = 1000.0 * float(late_spikes) / edge_window_ms
    if post_window_ms > 0.0:
        post_spikes = np.count_nonzero(spike_times_ms >= stim_stop_ms)
        post_dbs_rate_hz = 1000.0 * float(post_spikes) / post_window_ms

    plv_info = compute_hilbert_plv_at_pulses(
        time_ms=t_ms,
        voltage_mV=np.asarray(result["soma_v_mV"], dtype=float),
        pulse_times_ms=pulse_times_ms,
        freq_hz=float(dbs_config["freq_hz"]),
        bandwidth_hz=1.0,
    )

    return {
        "first_pulse_latency_ms": first_latency_ms,
        "first_spike_after_dbs_start_ms": first_spike_after_dbs_start_ms,
        "pre_dbs_rate_hz": pre_dbs_rate_hz,
        "during_dbs_rate_hz": during_dbs_rate_hz,
        "early_dbs_rate_hz": early_dbs_rate_hz,
        "late_dbs_rate_hz": late_dbs_rate_hz,
        "post_dbs_rate_hz": post_dbs_rate_hz,
        "pulse_plv": float(plv_info["plv"]),
        "pulse_phase_samples_rad": plv_info["sampled_phases_rad"],
        "pulse_times_ms": plv_info["sampled_pulse_times_ms"],
        "plv_band_hz": plv_info["band_hz"],
    }


def summarize_dbs_metrics(results: list[dict], dbs_config: dict | None) -> dict:
    first_pulse_latencies = np.array(
        [
            float(result.get("first_pulse_latency_ms", np.nan))
            for result in results
            if np.isfinite(result.get("first_pulse_latency_ms", np.nan))
        ],
        dtype=float,
    )
    broad_latencies = np.array(
        [
            float(result.get("first_spike_after_dbs_start_ms", np.nan))
            for result in results
            if np.isfinite(result.get("first_spike_after_dbs_start_ms", np.nan))
        ],
        dtype=float,
    )
    pre_rates = np.array(
        [
            float(result.get("pre_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("pre_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    during_rates = np.array(
        [
            float(result.get("during_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("during_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    early_rates = np.array(
        [
            float(result.get("early_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("early_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    late_rates = np.array(
        [
            float(result.get("late_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("late_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    post_rates = np.array(
        [
            float(result.get("post_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("post_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    neuron_plvs = np.array(
        [
            float(result.get("pulse_plv", np.nan))
            for result in results
            if np.isfinite(result.get("pulse_plv", np.nan))
        ],
        dtype=float,
    )

    pooled_phases = [
        np.asarray(result.get("pulse_phase_samples_rad", []), dtype=float)
        for result in results
        if len(np.asarray(result.get("pulse_phase_samples_rad", []), dtype=float)) > 0
    ]
    if pooled_phases:
        pooled_phase_array = np.concatenate(pooled_phases)
        pooled_plv = float(np.abs(np.mean(np.exp(1j * pooled_phase_array))))
        n_phase_samples = int(len(pooled_phase_array))
    else:
        pooled_plv = np.nan
        n_phase_samples = 0

    return {
        "first_pulse_latency_mean_ms": float(np.mean(first_pulse_latencies)) if len(first_pulse_latencies) else np.nan,
        "first_pulse_latency_n_valid": int(len(first_pulse_latencies)),
        "first_spike_after_dbs_start_mean_ms": float(np.mean(broad_latencies)) if len(broad_latencies) else np.nan,
        "first_spike_after_dbs_start_n_valid": int(len(broad_latencies)),
        "pre_rate_mean_hz": float(np.mean(pre_rates)) if len(pre_rates) else np.nan,
        "during_rate_mean_hz": float(np.mean(during_rates)) if len(during_rates) else np.nan,
        "early_rate_mean_hz": float(np.mean(early_rates)) if len(early_rates) else np.nan,
        "late_rate_mean_hz": float(np.mean(late_rates)) if len(late_rates) else np.nan,
        "post_rate_mean_hz": float(np.mean(post_rates)) if len(post_rates) else np.nan,
        "mean_neuron_plv": float(np.mean(neuron_plvs)) if len(neuron_plvs) else np.nan,
        "pooled_population_plv": pooled_plv,
        "n_phase_samples": n_phase_samples,
        "plv_band_hz": (np.nan, np.nan)
        if not dbs_config
        else (
            float(dbs_config["freq_hz"]) - 1.0,
            float(dbs_config["freq_hz"]) + 1.0,
        ),
    }


def _format_metric(value: float, unit: str = "", digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def exp2syn_peak_offset_ms(tau1_ms: float, tau2_ms: float) -> float:
    tau1_ms = max(0.0, float(tau1_ms))
    tau2_ms = max(0.0, float(tau2_ms))
    if tau1_ms <= 0.0 or tau2_ms <= 0.0:
        return np.nan
    if abs(tau2_ms - tau1_ms) <= 1e-12:
        return tau1_ms
    if tau2_ms < tau1_ms:
        tau1_ms, tau2_ms = tau2_ms, tau1_ms
    return float((tau1_ms * tau2_ms / (tau2_ms - tau1_ms)) * np.log(tau2_ms / tau1_ms))


def dbs_cycle_info_for_time_ms(time_ms: float, dbs_config: dict | None) -> tuple[int | None, float, float]:
    if (
        not dbs_config
        or not dbs_config.get("enabled", False)
        or not np.isfinite(time_ms)
        or float(dbs_config.get("freq_hz", 0.0)) <= 0.0
    ):
        return None, np.nan, np.nan

    start_ms = float(dbs_config["start_ms"])
    if time_ms < start_ms:
        return None, np.nan, np.nan
    stop_ms = dbs_config.get("stop_ms", None)
    if stop_ms is not None and time_ms >= float(stop_ms):
        return None, np.nan, np.nan

    period_ms = 1000.0 / float(dbs_config["freq_hz"])
    cycle_number = int(np.floor((float(time_ms) - start_ms) / period_ms)) + 1
    cycle_start_ms = start_ms + float(cycle_number - 1) * period_ms
    return cycle_number, cycle_start_ms, period_ms


def first_post_dbs_soma_spike_ms(result: dict, dbs_config: dict | None) -> float:
    t_ms = np.asarray(result.get("t_ms", []), dtype=float)
    stim_start_ms = float(t_ms[0]) if len(t_ms) else 0.0
    if dbs_config and dbs_config.get("enabled", False):
        stim_start_ms = float(dbs_config["start_ms"])

    spike_map = result.get("activation_spike_times_ms", {})
    soma_spikes = np.asarray(spike_map.get("soma", result.get("spike_times_ms", [])), dtype=float)
    soma_after = soma_spikes[np.isfinite(soma_spikes) & (soma_spikes >= stim_start_ms)]
    return float(soma_after[0]) if len(soma_after) else np.nan


def first_stn_local_axon_latency_after_first_pulse_ms(result: dict, dbs_config: dict | None) -> float:
    if not dbs_config or not dbs_config.get("enabled", False):
        return np.nan
    t_ms = np.asarray(result.get("t_ms", []), dtype=float)
    if len(t_ms) == 0:
        return np.nan
    pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
    if len(pulses) == 0:
        return np.nan

    first_pulse_ms = float(pulses[0])
    if len(pulses) >= 2:
        window_stop_ms = float(pulses[1]) - 1e-9
    else:
        period_ms = pulse_period_ms(dbs_config)
        window_stop_ms = first_pulse_ms + period_ms - 1e-9 if np.isfinite(period_ms) else float(t_ms[-1])
    window_stop_ms = min(window_stop_ms, float(t_ms[-1]))
    return first_sampled_latency_in_window_ms(result, "axon", first_pulse_ms, window_stop_ms)


def first_sampled_latency_in_window_ms(
    result: dict,
    site_kind: str,
    window_start_ms: float,
    window_stop_ms: float,
) -> float:
    if not np.isfinite(window_start_ms) or not np.isfinite(window_stop_ms):
        return np.nan

    spike_map = result.get("activation_spike_times_ms", {})
    sites = result.get("activation_sites", [])
    site_kind_by_name = {str(site["name"]): str(site.get("kind", "unknown")) for site in sites}

    if site_kind == "soma":
        names = ["soma"]
    elif site_kind == "AIS":
        names = ["AIS"]
    elif site_kind == "axon":
        names = [name for name, kind in site_kind_by_name.items() if kind == "axon"]
    else:
        names = []

    first_time_ms = np.nan
    for name in names:
        if name == "soma" and name not in spike_map:
            times = np.asarray(result.get("spike_times_ms", []), dtype=float)
        else:
            times = np.asarray(spike_map.get(name, []), dtype=float)
        local = times[
            np.isfinite(times)
            & (times >= float(window_start_ms) - 1e-9)
            & (times <= float(window_stop_ms) + 1e-9)
        ]
        if len(local) == 0:
            continue
        candidate = float(local[0])
        if not np.isfinite(first_time_ms) or candidate < first_time_ms:
            first_time_ms = candidate

    return first_time_ms - float(window_start_ms) if np.isfinite(first_time_ms) else np.nan


def pathway_cell_timing_summary(result: dict, pathway: str) -> dict:
    if pathway == "hdp":
        coupled_key = "hdp_coupled_results"
        summary_key = "hdp_synapse_summary"
        default_config = DEFAULT_HDP_CONFIG
        tau1_key = "syn_tau1_ms"
        tau2_key = "syn_tau2_ms"
    elif pathway == "gpe":
        coupled_key = "gpe_coupled_results"
        summary_key = "gpe_synapse_summary"
        default_config = DEFAULT_GPE_CONFIG
        tau1_key = "syn_tau1_ms"
        tau2_key = "syn_tau2_ms"
    else:
        raise ValueError("pathway must be 'hdp' or 'gpe'")

    pathway_results = list(result.get(coupled_key, []))
    summary = result.get(summary_key, {})
    target_infos = list(summary.get("target_infos", []))
    connected_axons = {
        int(info["axon_index"])
        for info in target_infos
        if "axon_index" in info
    }
    if not connected_axons and summary.get("enabled", False):
        connected_axons = {
            int(pathway_result.get("axon_index"))
            for pathway_result in pathway_results
            if "axon_index" in pathway_result
        }

    active = []
    for pathway_result in pathway_results:
        axon_index = int(pathway_result.get("axon_index", -1))
        if axon_index not in connected_axons:
            continue
        if bool(pathway_result.get("activated", False)):
            active.append(pathway_result)

    terminal_arrivals = np.array(
        [
            float(pathway_result.get("terminal_arrival_ms", np.nan))
            for pathway_result in active
            if np.isfinite(pathway_result.get("terminal_arrival_ms", np.nan))
        ],
        dtype=float,
    )
    terminal_mean_ms = float(np.mean(terminal_arrivals)) if len(terminal_arrivals) else np.nan
    delay_ms = float(summary.get("delay_ms", default_config.get("syn_delay_ms", np.nan)))
    peak_offset_ms = exp2syn_peak_offset_ms(
        summary.get("tau1_ms", default_config.get(tau1_key, np.nan)),
        summary.get("tau2_ms", default_config.get(tau2_key, np.nan)),
    )
    onset_mean_ms = terminal_mean_ms + delay_ms if np.isfinite(terminal_mean_ms) else np.nan
    peak_mean_ms = (
        onset_mean_ms + peak_offset_ms
        if np.isfinite(onset_mean_ms) and np.isfinite(peak_offset_ms)
        else np.nan
    )
    return {
        "connected": int(len(connected_axons)),
        "active": int(len(active)),
        "terminal_mean_ms": terminal_mean_ms,
        "onset_mean_ms": onset_mean_ms,
        "peak_mean_ms": peak_mean_ms,
    }


def _format_pathway_triplet(info: dict) -> str:
    values = [
        info.get("terminal_mean_ms", np.nan),
        info.get("onset_mean_ms", np.nan),
        info.get("peak_mean_ms", np.nan),
    ]
    if not any(np.isfinite(float(value)) for value in values):
        return "n/a/n/a/n/a"
    return "/".join(_format_metric(float(value), digits=2) for value in values)


def print_compact_cell_table(results: list[dict], dbs_config: dict | None) -> None:
    if not results:
        return

    header = (
        f"{'cell':>4}  {'soma_xyz_mm':>22}  {'spk':>4}  {'cycle':>5}  "
        f"{'pre_Hz':>7}  {'DBS_Hz':>7}  {'early':>7}  {'late':>7}  {'post_Hz':>7}  "
        f"{'axon_ms':>7}  {'AIS_ms':>7}  {'soma_ms':>7}  "
        f"{'HDP act/conn':>12}  {'HDP term/on/peak':>18}  "
        f"{'GPe act/conn':>12}  {'GPe term/on/peak':>18}  "
        f"{'STNax_ms':>8}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        soma_pos = result.get("placement", {}).get("soma_pos_mm", (np.nan, np.nan, np.nan))
        soma_xyz = f"{soma_pos[0]:.3f},{soma_pos[1]:.3f},{soma_pos[2]:.3f}"
        soma_time_ms = first_post_dbs_soma_spike_ms(result, dbs_config)
        cycle_number, cycle_start_ms, period_ms = dbs_cycle_info_for_time_ms(soma_time_ms, dbs_config)

        if cycle_number is None:
            cycle_label = "n/a"
            axon_latency_ms = np.nan
            ais_latency_ms = np.nan
            soma_latency_ms = np.nan
        else:
            cycle_label = str(cycle_number)
            window_stop_ms = soma_time_ms + 0.75
            if np.isfinite(period_ms):
                window_stop_ms = min(window_stop_ms, cycle_start_ms + period_ms - 1e-9)
            axon_latency_ms = first_sampled_latency_in_window_ms(
                result,
                "axon",
                cycle_start_ms,
                window_stop_ms,
            )
            ais_latency_ms = first_sampled_latency_in_window_ms(
                result,
                "AIS",
                cycle_start_ms,
                window_stop_ms,
            )
            soma_latency_ms = first_sampled_latency_in_window_ms(
                result,
                "soma",
                cycle_start_ms,
                window_stop_ms,
            )

        hdp_info = pathway_cell_timing_summary(result, "hdp")
        gpe_info = pathway_cell_timing_summary(result, "gpe")
        hdp_act_label = f"{hdp_info['active']}/{hdp_info['connected']}"
        gpe_act_label = f"{gpe_info['active']}/{gpe_info['connected']}"
        first_pulse_axon_latency_ms = first_stn_local_axon_latency_after_first_pulse_ms(result, dbs_config)

        print(
            f"{int(result['cell_index']):4d}  "
            f"{soma_xyz:>22}  "
            f"{int(result['spike_count']):4d}  "
            f"{cycle_label:>5}  "
            f"{_format_metric(result.get('pre_dbs_rate_hz', np.nan), digits=1):>7}  "
            f"{_format_metric(result.get('during_dbs_rate_hz', np.nan), digits=1):>7}  "
            f"{_format_metric(result.get('early_dbs_rate_hz', np.nan), digits=1):>7}  "
            f"{_format_metric(result.get('late_dbs_rate_hz', np.nan), digits=1):>7}  "
            f"{_format_metric(result.get('post_dbs_rate_hz', np.nan), digits=1):>7}  "
            f"{_format_metric(axon_latency_ms, digits=2):>7}  "
            f"{_format_metric(ais_latency_ms, digits=2):>7}  "
            f"{_format_metric(soma_latency_ms, digits=2):>7}  "
            f"{hdp_act_label:>12}  "
            f"{_format_pathway_triplet(hdp_info):>18}  "
            f"{gpe_act_label:>12}  "
            f"{_format_pathway_triplet(gpe_info):>18}  "
            f"{_format_metric(first_pulse_axon_latency_ms, digits=2):>8}"
        )


def print_dbs_metrics(results: list[dict], dbs_config: dict | None) -> None:
    if not dbs_config or not dbs_config.get("enabled", False):
        return

    band_low_hz, band_high_hz = (
        float(dbs_config["freq_hz"]) - 1.0,
        float(dbs_config["freq_hz"]) + 1.0,
    )
    print(f"DBS metrics (Hilbert PLV band: {band_low_hz:.1f}-{band_high_hz:.1f} Hz)")
    for result in results:
        print(
            f"cell {result['cell_index']}: "
            f"latency_first={_format_metric(result.get('first_pulse_latency_ms', np.nan), ' ms')}, "
            f"PLV={_format_metric(result.get('pulse_plv', np.nan), digits=3)}, "
            f"rate_pre={_format_metric(result.get('pre_dbs_rate_hz', np.nan), ' Hz')}, "
            f"rate_dbs={_format_metric(result.get('during_dbs_rate_hz', np.nan), ' Hz')}, "
            f"rate_early={_format_metric(result.get('early_dbs_rate_hz', np.nan), ' Hz')}, "
            f"rate_late={_format_metric(result.get('late_dbs_rate_hz', np.nan), ' Hz')}, "
            f"rate_post={_format_metric(result.get('post_dbs_rate_hz', np.nan), ' Hz')}, "
            f"dF={_format_metric(result.get('dist_to_fiber_center_mm', np.nan), ' mm', digits=3)}, "
            f"latency_any={_format_metric(result.get('first_spike_after_dbs_start_ms', np.nan), ' ms')}"
        )

    summary = summarize_dbs_metrics(results, dbs_config)
    print(
        "all cells: "
        f"latency_first_mean={_format_metric(summary['first_pulse_latency_mean_ms'], ' ms')} "
        f"(n={summary['first_pulse_latency_n_valid']}), "
        f"PLV_pooled={_format_metric(summary['pooled_population_plv'], digits=3)}, "
        f"PLV_mean={_format_metric(summary['mean_neuron_plv'], digits=3)}, "
        f"rate_pre_mean={_format_metric(summary['pre_rate_mean_hz'], ' Hz')}, "
        f"rate_dbs_mean={_format_metric(summary['during_rate_mean_hz'], ' Hz')}, "
        f"rate_early_mean={_format_metric(summary['early_rate_mean_hz'], ' Hz')}, "
        f"rate_late_mean={_format_metric(summary['late_rate_mean_hz'], ' Hz')}, "
        f"rate_post_mean={_format_metric(summary['post_rate_mean_hz'], ' Hz')}, "
        f"latency_any_mean={_format_metric(summary['first_spike_after_dbs_start_mean_ms'], ' ms')} "
        f"(n={summary['first_spike_after_dbs_start_n_valid']})"
    )


def plot_post_rate_distribution(results: list[dict], dbs_config: dict | None, title: str) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    rates = np.array(
        [
            float(result.get("during_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("during_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    if len(rates) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    bins = min(24, max(8, int(np.sqrt(len(rates))) + 4))
    ax.hist(rates, bins=bins, color="0.35", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("During-DBS firing rate (Hz)")
    ax.set_ylabel("Cells")
    if dbs_config and dbs_config.get("enabled", False):
        stim_freq_hz = float(dbs_config["freq_hz"])
        entrain_threshold_hz = 0.90 * stim_freq_hz
        entrained = int(np.count_nonzero(rates >= entrain_threshold_hz))
        ax.axvline(stim_freq_hz, color="tab:red", ls="--", lw=1.3, label="DBS frequency")
        ax.axvline(entrain_threshold_hz, color="tab:orange", ls=":", lw=1.3, label="90% entrainment")
        ax.text(
            0.98,
            0.95,
            f"entrained: {entrained}/{len(rates)} ({100.0 * entrained / len(rates):.1f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.legend(loc="upper left")
    fig.tight_layout()
    finish_figure(fig, title)


def plot_plv_histogram(results: list[dict], dbs_config: dict | None, title: str) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    plvs = np.array(
        [
            float(result.get("pulse_plv", np.nan))
            for result in results
            if np.isfinite(result.get("pulse_plv", np.nan))
        ],
        dtype=float,
    )
    if len(plvs) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    ax.hist(plvs, bins=np.linspace(0.0, 1.0, 21), color="0.35", edgecolor="white", alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("PLV at DBS frequency")
    ax.set_ylabel("Cells")
    ax.axvline(float(np.mean(plvs)), color="tab:red", ls="--", lw=1.3, label=f"mean {np.mean(plvs):.3f}")
    ax.axvline(float(np.median(plvs)), color="tab:orange", ls=":", lw=1.3, label=f"median {np.median(plvs):.3f}")
    if dbs_config and dbs_config.get("enabled", False):
        band_low_hz = float(dbs_config["freq_hz"]) - 1.0
        band_high_hz = float(dbs_config["freq_hz"]) + 1.0
        ax.text(
            0.02,
            0.95,
            f"band: {band_low_hz:.1f}-{band_high_hz:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    ax.legend(loc="upper center")
    fig.tight_layout()
    finish_figure(fig, title)


def terminal_spike_times_from_pathway_result(pathway_result: dict) -> np.ndarray:
    site_spikes = pathway_result.get("site_spike_times_ms", {})
    if not site_spikes:
        return np.array([], dtype=float)

    terminal_name = None
    terminal_index = -1
    for name in site_spikes:
        match = re.fullmatch(r"collateral_node(\d+)", str(name))
        if match is None:
            continue
        index = int(match.group(1))
        if index > terminal_index:
            terminal_index = index
            terminal_name = str(name)

    if terminal_name is None:
        return np.array([], dtype=float)
    return np.asarray(site_spikes.get(terminal_name, []), dtype=float)


def connected_pathway_axon_keys(results: list[dict], pathway: str) -> set[tuple[int, int]]:
    summary_key = f"{pathway}_synapse_summary"
    keys: set[tuple[int, int]] = set()
    for result in results:
        cell_index = int(result.get("cell_index", -1))
        for info in result.get(summary_key, {}).get("target_infos", []):
            if "axon_index" in info:
                keys.add((cell_index, int(info["axon_index"])))
    return keys


def depression_resource_events(
    spike_times_ms: np.ndarray,
    *,
    depression_enabled: bool,
    depression_u: float,
    depression_tau_rec_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    spikes = np.asarray(spike_times_ms, dtype=float)
    spikes = np.sort(spikes[np.isfinite(spikes)])
    if len(spikes) == 0:
        return spikes, np.array([], dtype=float)

    if not depression_enabled:
        return spikes, np.ones_like(spikes, dtype=float)

    u = min(1.0, max(0.0, float(depression_u)))
    tau_rec = max(0.001, float(depression_tau_rec_ms))
    resources = []
    resource = 1.0
    last_time = None
    for spike_time in spikes:
        if last_time is not None:
            dt_ms = max(0.0, float(spike_time - last_time))
            resource = 1.0 - (1.0 - resource) * np.exp(-dt_ms / tau_rec)
        resource = min(1.0, max(0.0, resource))
        resources.append(resource)
        resource *= 1.0 - u
        last_time = float(spike_time)

    return spikes, np.array(resources, dtype=float)


def pathway_pulse_terminal_metrics(
    pathway_results: list[dict],
    pulses_ms: np.ndarray,
    window_stops_ms: np.ndarray,
    *,
    connected_keys: set[tuple[int, int]] | None = None,
    depression_enabled: bool = False,
    depression_u: float = 0.0,
    depression_tau_rec_ms: float = 1.0,
) -> dict[str, np.ndarray | int]:
    connected_keys = connected_keys or set()
    selected = []
    for pathway_result in pathway_results:
        key = (
            int(pathway_result.get("cell_index", -1)),
            int(pathway_result.get("axon_index", -1)),
        )
        if connected_keys and key not in connected_keys:
            continue
        selected.append(pathway_result)

    n_axons = int(len(selected))
    n_pulses = int(len(pulses_ms))
    active_counts = np.zeros(n_pulses, dtype=float)
    weighted_drive = np.zeros(n_pulses, dtype=float)
    active_resource_sum = np.zeros(n_pulses, dtype=float)
    active_resource_n = np.zeros(n_pulses, dtype=float)

    for pathway_result in selected:
        terminal_spikes, resources = depression_resource_events(
            terminal_spike_times_from_pathway_result(pathway_result),
            depression_enabled=depression_enabled,
            depression_u=depression_u,
            depression_tau_rec_ms=depression_tau_rec_ms,
        )
        if len(terminal_spikes) == 0:
            continue

        for pulse_index, (pulse_ms, stop_ms) in enumerate(zip(pulses_ms, window_stops_ms)):
            in_window = (
                (terminal_spikes >= float(pulse_ms) - 1e-9)
                & (terminal_spikes < float(stop_ms) - 1e-9)
            )
            if not np.any(in_window):
                continue
            resource_mean = float(np.mean(resources[in_window]))
            active_counts[pulse_index] += 1.0
            weighted_drive[pulse_index] += resource_mean
            active_resource_sum[pulse_index] += resource_mean
            active_resource_n[pulse_index] += 1.0

    if n_axons > 0:
        activation_fraction = active_counts / float(n_axons)
        drive_fraction = weighted_drive / float(n_axons)
    else:
        activation_fraction = np.full(n_pulses, np.nan, dtype=float)
        drive_fraction = np.full(n_pulses, np.nan, dtype=float)

    active_resource = np.full(n_pulses, np.nan, dtype=float)
    has_active = active_resource_n > 0
    active_resource[has_active] = active_resource_sum[has_active] / active_resource_n[has_active]

    return {
        "n_axons": n_axons,
        "activation_fraction": activation_fraction,
        "drive_fraction": drive_fraction,
        "active_resource": active_resource,
    }


def compute_pulse_recruitment_dynamics(
    results: list[dict],
    hdp_results: list[dict],
    gpe_results: list[dict],
    dbs_config: dict | None,
) -> dict:
    if not results or not dbs_config or not dbs_config.get("enabled", False):
        return {"enabled": False}

    t_arrays = [np.asarray(result.get("t_ms", []), dtype=float) for result in results]
    t_arrays = [arr for arr in t_arrays if len(arr) > 0]
    if not t_arrays:
        return {"enabled": False}

    t_start_ms = min(float(arr[0]) for arr in t_arrays)
    t_stop_ms = max(float(arr[-1]) for arr in t_arrays)
    pulses_ms = dbs_first_phase_pulse_times_ms(t_start_ms, t_stop_ms, dbs_config)
    if len(pulses_ms) == 0:
        return {"enabled": False}

    period_ms = pulse_period_ms(dbs_config)
    if not np.isfinite(period_ms) or period_ms <= 0.0:
        return {"enabled": False}

    window_stops_ms = np.empty_like(pulses_ms)
    if len(pulses_ms) > 1:
        window_stops_ms[:-1] = pulses_ms[1:]
    window_stops_ms[-1] = min(t_stop_ms + 1e-9, float(pulses_ms[-1]) + period_ms)

    stn_active = np.zeros(len(pulses_ms), dtype=float)
    for pulse_index, (pulse_ms, stop_ms) in enumerate(zip(pulses_ms, window_stops_ms)):
        n_active = 0
        for result in results:
            latency = first_sampled_latency_in_window_ms(result, "axon", float(pulse_ms), float(stop_ms))
            if np.isfinite(latency):
                n_active += 1
        stn_active[pulse_index] = n_active / float(len(results))

    hdp_summary = summarize_hdp_synapses_from_results(results)
    gpe_summary = summarize_gpe_synapses_from_results(results)
    hdp_metrics = pathway_pulse_terminal_metrics(
        hdp_results,
        pulses_ms,
        window_stops_ms,
        connected_keys=connected_pathway_axon_keys(results, "hdp"),
        depression_enabled=bool(hdp_summary.get("depression_enabled", False)),
        depression_u=float(hdp_summary.get("depression_u", 0.0)),
        depression_tau_rec_ms=float(hdp_summary.get("depression_tau_rec_ms", 1.0)),
    )
    gpe_metrics = pathway_pulse_terminal_metrics(
        gpe_results,
        pulses_ms,
        window_stops_ms,
        connected_keys=connected_pathway_axon_keys(results, "gpe"),
        depression_enabled=bool(gpe_summary.get("depression_enabled", False)),
        depression_u=float(gpe_summary.get("depression_u", 0.0)),
        depression_tau_rec_ms=float(gpe_summary.get("depression_tau_rec_ms", 1.0)),
    )

    return {
        "enabled": True,
        "pulse_times_ms": pulses_ms,
        "time_from_first_pulse_s": (pulses_ms - float(pulses_ms[0])) / 1000.0,
        "stn_activation_fraction": stn_active,
        "hdp": hdp_metrics,
        "gpe": gpe_metrics,
    }


def plot_recruitment_dynamics(
    results: list[dict],
    hdp_results: list[dict],
    gpe_results: list[dict],
    dbs_config: dict | None,
    title: str,
) -> None:
    dynamics = compute_pulse_recruitment_dynamics(results, hdp_results, gpe_results, dbs_config)
    if not dynamics.get("enabled", False):
        return

    import matplotlib.pyplot as plt

    x_s = np.asarray(dynamics["time_from_first_pulse_s"], dtype=float)
    hdp = dynamics["hdp"]
    gpe = dynamics["gpe"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    ax0.plot(
        x_s,
        100.0 * np.asarray(dynamics["stn_activation_fraction"], dtype=float),
        label="STN local axons",
        lw=1.5,
    )
    if int(hdp.get("n_axons", 0)) > 0:
        ax0.plot(x_s, 100.0 * np.asarray(hdp["activation_fraction"], dtype=float), label="HDP terminals", lw=1.5)
    if int(gpe.get("n_axons", 0)) > 0:
        ax0.plot(x_s, 100.0 * np.asarray(gpe["activation_fraction"], dtype=float), label="GPe terminals", lw=1.5)
    ax0.set_ylabel("Activated (%)")
    ax0.set_ylim(-2.0, 102.0)
    ax0.set_title("Pulse-by-pulse axon recruitment")
    ax0.legend(loc="upper right")

    if int(hdp.get("n_axons", 0)) > 0:
        ax1.plot(x_s, 100.0 * np.asarray(hdp["drive_fraction"], dtype=float), label="HDP AMPA drive", lw=1.5)
    if int(gpe.get("n_axons", 0)) > 0:
        ax1.plot(x_s, 100.0 * np.asarray(gpe["drive_fraction"], dtype=float), label="GPe GABA-A drive", lw=1.5)
    ax1.set_ylabel("Depression-weighted drive (%)")
    ax1.set_xlabel("Time from first DBS pulse (s)")
    ax1.set_ylim(-2.0, 102.0)
    ax1.set_title("Terminal recruitment weighted by synaptic availability")
    ax1.legend(loc="upper right")

    fig.tight_layout()
    finish_figure(fig, title)


def classify_activation_origin(
    result: dict,
    dbs_config: dict | None,
    *,
    threshold_window_pre_ms: float = 2.0,
    threshold_window_post_ms: float = 0.75,
    tie_tolerance_ms: float = 0.05,
) -> dict:
    t_ms = np.asarray(result.get("t_ms", []), dtype=float)
    if len(t_ms) == 0:
        return {"classification": "unavailable"}

    stim_start_ms = float(t_ms[0])
    if dbs_config and dbs_config.get("enabled", False):
        stim_start_ms = float(dbs_config["start_ms"])

    spike_map = result.get("activation_spike_times_ms", {})
    soma_spikes = np.asarray(spike_map.get("soma", result.get("spike_times_ms", [])), dtype=float)
    soma_after = soma_spikes[soma_spikes >= stim_start_ms]
    if len(soma_after) == 0:
        return {
            "classification": "no_soma_spike",
            "soma_time_ms": np.nan,
            "first_site": "n/a",
            "first_kind": "n/a",
            "first_time_ms": np.nan,
            "delay_to_soma_ms": np.nan,
            "first_distance_um": np.nan,
            "pulse_number": np.nan,
            "latency_from_pulse_ms": np.nan,
        }

    soma_time_ms = float(soma_after[0])
    sites = result.get("activation_sites", [])
    site_by_name = {site["name"]: site for site in sites}
    candidates = []
    win_start = soma_time_ms - float(threshold_window_pre_ms)
    win_stop = soma_time_ms + float(threshold_window_post_ms)
    for name, times in spike_map.items():
        times = np.asarray(times, dtype=float)
        local = times[(times >= win_start) & (times <= win_stop)]
        if len(local) == 0:
            continue
        site = site_by_name.get(name, {"kind": "unknown", "distance_um": np.nan})
        candidates.append(
            {
                "name": name,
                "kind": site.get("kind", "unknown"),
                "distance_um": float(site.get("distance_um", np.nan)),
                "time_ms": float(local[0]),
            }
        )

    if not candidates:
        return {
            "classification": "ambiguous",
            "soma_time_ms": soma_time_ms,
            "first_site": "n/a",
            "first_kind": "n/a",
            "first_time_ms": np.nan,
            "delay_to_soma_ms": np.nan,
            "first_distance_um": np.nan,
            "pulse_number": np.nan,
            "latency_from_pulse_ms": np.nan,
        }

    candidates.sort(key=lambda item: item["time_ms"])
    first = candidates[0]
    first_time_ms = float(first["time_ms"])
    tied = [
        item
        for item in candidates
        if abs(float(item["time_ms"]) - first_time_ms) <= float(tie_tolerance_ms)
    ]
    tied_kinds = {str(item["kind"]) for item in tied}
    delay_to_soma_ms = soma_time_ms - first_time_ms

    if len(tied_kinds) > 1:
        classification = "ambiguous"
    elif first["kind"] == "axon" and delay_to_soma_ms > tie_tolerance_ms:
        classification = "axon-first"
    elif first["kind"] == "AIS" and delay_to_soma_ms > tie_tolerance_ms:
        classification = "AIS-first"
    elif first["kind"] == "soma" and abs(delay_to_soma_ms) <= tie_tolerance_ms:
        classification = "soma-first"
    elif abs(delay_to_soma_ms) <= tie_tolerance_ms:
        classification = "ambiguous"
    else:
        classification = f"{first['kind']}-first"

    pulse_number = np.nan
    latency_from_pulse_ms = np.nan
    if dbs_config and dbs_config.get("enabled", False):
        pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
        previous = pulses[pulses <= first_time_ms + 1e-9]
        if len(previous):
            pulse_number = float(len(previous))
            latency_from_pulse_ms = first_time_ms - float(previous[-1])

    return {
        "classification": classification,
        "soma_time_ms": soma_time_ms,
        "first_site": first["name"],
        "first_kind": first["kind"],
        "first_time_ms": first_time_ms,
        "delay_to_soma_ms": delay_to_soma_ms,
        "first_distance_um": first["distance_um"],
        "pulse_number": pulse_number,
        "latency_from_pulse_ms": latency_from_pulse_ms,
        "tied_sites": [item["name"] for item in tied],
    }


def summarize_activation_origins(results: list[dict]) -> dict:
    labels = [
        "axon-first",
        "AIS-first",
        "soma-first",
        "ambiguous",
        "no_soma_spike",
        "unavailable",
    ]
    counts = {label: 0 for label in labels}
    delays_by_label = {label: [] for label in labels}
    for result in results:
        info = result.get("activation_origin", {})
        label = str(info.get("classification", "unavailable"))
        counts[label] = counts.get(label, 0) + 1
        delay = float(info.get("delay_to_soma_ms", np.nan))
        if np.isfinite(delay):
            delays_by_label.setdefault(label, []).append(delay)

    delay_means = {
        label: float(np.mean(values)) if values else np.nan
        for label, values in delays_by_label.items()
    }
    return {
        "counts": counts,
        "delay_mean_ms": delay_means,
    }


def print_activation_origin_summary(
    results: list[dict],
    dbs_config: dict | None,
    *,
    max_detail_rows: int = 20,
) -> None:
    if not results:
        return
    if not dbs_config or not dbs_config.get("enabled", False):
        return

    print("Activation origin diagnostics (0 mV crossings; sampled soma/AIS/nodes)")
    for result in results[:max_detail_rows]:
        info = result.get("activation_origin", {})
        pulse_number = info.get("pulse_number", np.nan)
        pulse_label = "n/a" if not np.isfinite(pulse_number) else str(int(pulse_number))
        print(
            f"cell {result['cell_index']}: "
            f"{info.get('classification', 'unavailable')}, "
            f"first={info.get('first_site', 'n/a')} "
            f"at {_format_metric(info.get('first_time_ms', np.nan), ' ms')}, "
            f"soma={_format_metric(info.get('soma_time_ms', np.nan), ' ms')}, "
            f"delay_to_soma={_format_metric(info.get('delay_to_soma_ms', np.nan), ' ms', digits=3)}, "
            f"pulse={pulse_label}, "
            f"latency_from_pulse={_format_metric(info.get('latency_from_pulse_ms', np.nan), ' ms', digits=3)}"
        )
    if len(results) > max_detail_rows:
        print(f"... detail rows truncated at {max_detail_rows}/{len(results)} cells")

    summary = summarize_activation_origins(results)
    counts = summary["counts"]
    delay_means = summary["delay_mean_ms"]
    print("Activation origin summary:")
    print("origin          count   mean first-to-soma delay")
    preferred_order = ["axon-first", "AIS-first", "soma-first", "ambiguous", "no_soma_spike", "unavailable"]
    for label in preferred_order:
        if label not in counts:
            continue
        print(
            f"{label:<14} {int(counts[label]):>5}   "
            f"{_format_metric(delay_means.get(label, np.nan), ' ms', digits=3)}"
        )


def _find_peaks(voltage: np.ndarray, height: float = 0.0):
    from scipy.signal import find_peaks

    return find_peaks(voltage, height=height)


def detect_burst(voltage: np.ndarray) -> bool:
    peaks, _ = _find_peaks(np.asarray(voltage), height=0.0)
    diff_p = np.diff(peaks)
    if len(diff_p) <= 1:
        return False

    for i in range(1, len(diff_p)):
        if diff_p[i] / diff_p[i - 1] > 1.1 or diff_p[i - 1] / diff_p[i] > 1.1:
            return True
    return False


def get_frequency(voltage: np.ndarray, dt_ms: float) -> float:
    peaks, _ = _find_peaks(np.asarray(voltage), height=0.0)
    diff_p = np.diff(peaks)
    if detect_burst(voltage) or len(diff_p) < 1:
        return 0.0
    return 1000.0 / dt_ms / float(np.mean(diff_p))


def get_frequency_detect_burst(voltage: np.ndarray, dt_ms: float) -> float:
    if detect_burst(voltage):
        return 0.0
    return get_frequency(voltage, dt_ms)


def check_ahp(voltage: np.ndarray) -> float:
    return float(np.min(voltage))


def check_peak(voltage: np.ndarray) -> float:
    return float(np.max(voltage))


def check_rest(voltage: np.ndarray) -> float:
    peaks, _ = _find_peaks(np.asarray(voltage), height=0.0)
    if len(peaks) == 0:
        return float(np.mean(voltage))

    checkpoints = []
    for i in range(len(peaks) - 1):
        point = int((peaks[i] + peaks[i + 1]) / 2)
        checkpoints.append(voltage[point])
    return float(np.mean(checkpoints)) if checkpoints else float(np.mean(voltage))


def calculate_ap_width(time_ms: np.ndarray, voltage: np.ndarray) -> float | None:
    try:
        import efel
    except ImportError:
        return None

    trace = {
        "T": time_ms,
        "V": voltage,
        "stim_start": [500.0],
        "stim_end": [1000.0],
    }
    results = efel.get_feature_values([trace], ["AP2_width"])
    for feature_map in results:
        values = feature_map.get("AP2_width")
        if values is not None and len(values) > 0:
            return float(np.mean(values))
    return None


def report_input_impedance(cell, temperature_c: float = 37.0) -> float:
    soma_sec = soma_section(cell)
    h.celsius = temperature_c
    h.finitialize()
    z = h.Impedance()
    z.loc(0.5, sec=soma_sec)
    z.compute(0)
    return float(z.input(0.5, sec=soma_sec))


def _run_protocol_step(
    cell,
    *,
    tstop_ms: float,
    amp_nA: float,
    delay_ms: float,
    dur_ms: float,
    temperature_c: float,
    save_state=None,
):
    soma_ref = soma_section(cell)(0.5)
    h.dt = 0.025
    h.celsius = temperature_c

    stim = h.IClamp(soma_ref)
    stim.delay = delay_ms
    stim.dur = dur_ms
    stim.amp = amp_nA

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma_ref._ref_v)

    if save_state is None:
        h.finitialize(-60.0)
    else:
        save_state.restore()
        h.frecord_init()

    h.continuerun(tstop_ms * ms)
    return np.array(t_vec), np.array(v_vec)


def run_protocol_suite(
    model: str = "gw",
    params: Iterable[float] | None = None,
    morphology: str | Path | None = None,
    temperature_c: float = 37.0,
):
    cell = create_2024_cell(model=model, params=params, morphology=morphology)
    input_impedance_mohm = report_input_impedance(cell, temperature_c=temperature_c)

    t_spont, v_spont = _run_protocol_step(
        create_2024_cell(model=model, params=params, morphology=morphology),
        tstop_ms=1500.0,
        amp_nA=0.0,
        delay_ms=500.0,
        dur_ms=500.0,
        temperature_c=temperature_c,
    )

    dt_ms = float(h.dt)
    spont_rel_t = t_spont - t_spont[0]
    spont_mask = spont_rel_t >= 1000.0
    if not np.any(spont_mask):
        spont_mask = np.ones_like(spont_rel_t, dtype=bool)
    t_window = t_spont[spont_mask]
    v_window = v_spont[spont_mask]

    fi_protocols = [
        ("fi_0p04", 0.04, 1500.0, 1000.0, 2500.0, 2000.0),
        ("fi_0p10", 0.10, 1500.0, 1000.0, 2500.0, 2000.0),
        ("fi_0p16", 0.16, 1500.0, 1500.0, 3000.0, 2000.0),
        ("fi_0p20", 0.20, 1500.0, 1000.0, 2500.0, 2000.0),
    ]

    fi_results = {}
    for name, amp, delay, dur, tstop, analysis_start in fi_protocols:
        t_step, v_step = _run_protocol_step(
            create_2024_cell(model=model, params=params, morphology=morphology),
            tstop_ms=tstop,
            amp_nA=amp,
            delay_ms=delay,
            dur_ms=dur,
            temperature_c=temperature_c,
        )
        mask = t_step >= analysis_start
        if not np.any(mask):
            mask = np.ones_like(t_step, dtype=bool)
        fi_results[name] = {
            "amp_nA": amp,
            "frequency_hz": get_frequency_detect_burst(v_step[mask], dt_ms),
            "t_ms": t_step,
            "v_mV": v_step,
        }

    t_hyp, v_hyp = _run_protocol_step(
        create_2024_cell(model=model, params=params, morphology=morphology),
        tstop_ms=2500.0,
        amp_nA=-0.1,
        delay_ms=1600.0,
        dur_ms=500.0,
        temperature_c=temperature_c,
    )

    return {
        "model": model,
        "input_impedance_mohm": input_impedance_mohm,
        "spontaneous_frequency_hz": get_frequency_detect_burst(v_window, dt_ms),
        "spontaneous_trace": {"t_ms": t_spont, "v_mV": v_spont},
        "rest_mV": check_rest(v_window),
        "ahp_mV": check_ahp(v_window),
        "peak_mV": check_peak(v_window),
        "ap2_width_ms": calculate_ap_width(t_window, v_window),
        "hyperpolarization_trace": {"t_ms": t_hyp, "v_mV": v_hyp, "min_mV": float(np.min(v_hyp))},
        "fi_curve": fi_results,
    }


def plot_trace(result: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    has_dbs = result.get("dbs_uA") is not None
    if has_dbs:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(9, 5.5),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 4))
        ax2 = None

    ax1.plot(result["t_ms"], result["soma_v_mV"], label="soma")
    ax1.plot(result["t_ms"], result["initseg_v_mV"], label="AIS")
    ax1.plot(result["t_ms"], result["node0_v_mV"], label="node0")
    for key in sorted(k for k in result if k.startswith("node") and k.endswith("_v_mV") and k != "node0_v_mV"):
        label = key.replace("_v_mV", "")
        ax1.plot(result["t_ms"], result[key], label=label, alpha=0.75)
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(title)
    ax1.legend()

    if has_dbs and ax2 is not None:
        dbs_plot = result.get("dbs_plot_uA", result["dbs_uA"])
        ax2.plot(result["t_ms"], dbs_plot, color="black", lw=1.2)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("DBS at A (uA)")
    else:
        ax1.set_xlabel("Time (ms)")

    fig.tight_layout()
    finish_figure(fig, title)


def plot_population_traces(results: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    has_dbs = results and results[0].get("dbs_uA") is not None
    if has_dbs:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(9, 5.5),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 4))
        ax2 = None

    for result in results:
        ax1.plot(
            result["t_ms"],
            result["soma_v_mV"],
            label=f"cell {result['cell_index']}",
            alpha=0.8,
        )
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(title)
    if len(results) <= 8:
        ax1.legend()

    if has_dbs and ax2 is not None:
        dbs_plot = results[0].get("dbs_plot_uA", results[0]["dbs_uA"])
        ax2.plot(results[0]["t_ms"], dbs_plot, color="black", lw=1.2)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("DBS at A (uA)")
    else:
        ax1.set_xlabel("Time (ms)")

    fig.tight_layout()
    finish_figure(fig, title)


def estimate_fs_from_ms(t_ms: np.ndarray) -> float:
    t_ms = np.asarray(t_ms, dtype=float)
    if len(t_ms) < 2:
        return np.nan
    dt_ms = np.diff(t_ms)
    dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
    if len(dt_ms) == 0:
        return np.nan
    return 1000.0 / float(np.median(dt_ms))


def population_mean_trace(results: list[dict], trace_key: str = "soma_v_mV") -> tuple[np.ndarray, np.ndarray]:
    if not results:
        return np.array([], dtype=float), np.array([], dtype=float)
    t_ms = np.asarray(results[0]["t_ms"], dtype=float)
    traces = []
    for result in results:
        trace = np.asarray(result.get(trace_key, []), dtype=float)
        if trace.shape == t_ms.shape:
            traces.append(trace)
    if not traces:
        return t_ms, np.array([], dtype=float)
    return t_ms, np.mean(np.vstack(traces), axis=0)


def prepare_analysis_trace(
    t_ms: np.ndarray,
    trace: np.ndarray,
    *,
    target_fs_hz: float | None = None,
    lowpass_fraction: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, float]:
    t_ms = np.asarray(t_ms, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if len(t_ms) < 4 or trace.shape != t_ms.shape:
        return t_ms, trace, np.nan

    fs_hz = estimate_fs_from_ms(t_ms)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return t_ms, trace, np.nan

    if target_fs_hz is None or not np.isfinite(target_fs_hz) or target_fs_hz <= 0:
        return t_ms, trace, fs_hz
    if fs_hz <= 1.05 * float(target_fs_hz):
        return t_ms, trace, fs_hz

    target_fs_hz = float(target_fs_hz)
    filtered = trace.copy()
    try:
        from scipy.signal import butter, sosfiltfilt

        cutoff_hz = min(lowpass_fraction * target_fs_hz, 0.45 * fs_hz)
        if 0.0 < cutoff_hz < 0.5 * fs_hz:
            sos = butter(4, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
            filtered = sosfiltfilt(sos, filtered)
    except Exception:
        pass

    dt_ms = 1000.0 / target_fs_hz
    new_t_ms = np.arange(float(t_ms[0]), float(t_ms[-1]) + 0.5 * dt_ms, dt_ms)
    new_t_ms = new_t_ms[new_t_ms <= float(t_ms[-1])]
    if len(new_t_ms) < 4:
        return t_ms, trace, fs_hz
    new_trace = np.interp(new_t_ms, t_ms, filtered)
    return new_t_ms, new_trace, target_fs_hz


def tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if alpha <= 0:
        return np.ones(n, dtype=float)
    if alpha >= 1:
        return np.hanning(n)
    x = np.linspace(0.0, 1.0, n)
    window = np.ones(n, dtype=float)
    left = x < alpha / 2.0
    right = x >= 1.0 - alpha / 2.0
    window[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi * (x[left] / alpha - 0.5)))
    window[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi * (x[right] / alpha - 1.0 / alpha + 0.5)))
    return window


def analytic_signal_fft(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=complex)
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(n, dtype=float)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier)


def fft_bandpass_trace(trace: np.ndarray, fs_hz: float, band_hz: tuple[float, float]) -> np.ndarray | None:
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None
    low, high = float(band_hz[0]), float(band_hz[1])
    if not (0.0 < low < high < 0.5 * fs_hz):
        return None
    x = np.asarray(trace, dtype=float) - float(np.mean(trace))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs_hz)
    spectrum = np.fft.rfft(x)
    keep = (freqs >= low) & (freqs <= high)
    if not np.any(keep) and len(freqs):
        keep[int(np.argmin(np.abs(freqs - 0.5 * (low + high))))] = True
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=len(x))


def manual_spectrogram_psd(trace: np.ndarray, fs_hz: float, nperseg: int, noverlap: int):
    trace = np.asarray(trace, dtype=float)
    step = max(1, int(nperseg) - int(noverlap))
    starts = np.arange(0, len(trace) - int(nperseg) + 1, step, dtype=int)
    if len(starts) == 0:
        starts = np.array([0], dtype=int)
    window = np.hanning(int(nperseg))
    scale = float(fs_hz) * float(np.sum(window ** 2))
    scale = scale if scale > 0 else 1.0
    freqs = np.fft.rfftfreq(int(nperseg), d=1.0 / float(fs_hz))
    power_cols = []
    times_s = []
    for start in starts:
        segment = trace[start : start + int(nperseg)]
        if len(segment) < int(nperseg):
            pad = np.full(int(nperseg), np.nan, dtype=float)
            pad[: len(segment)] = segment
            segment = np.nan_to_num(pad, nan=float(np.nanmean(segment)) if len(segment) else 0.0)
        segment = segment - float(np.mean(segment))
        spectrum = np.fft.rfft(segment * window)
        power = (np.abs(spectrum) ** 2) / scale
        if len(power) > 2:
            power[1:-1] *= 2.0
        power_cols.append(power)
        times_s.append((float(start) + 0.5 * float(nperseg)) / float(fs_hz))
    return freqs, np.asarray(times_s, dtype=float), np.asarray(power_cols, dtype=float).T


def db_from_linear_power(power_linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power_linear, dtype=float), 1e-30))


def compute_pipeline_spectrogram_data(
    t_ms: np.ndarray,
    trace: np.ndarray,
    *,
    nfft_spec: int | None = None,
    analysis_fs_hz: float | None = None,
    window_ms: float = 100.0,
    overlap_frac: float = 0.90,
) -> dict:
    t_ms, trace, fs_hz = prepare_analysis_trace(
        t_ms,
        trace,
        target_fs_hz=analysis_fs_hz,
    )
    if not np.isfinite(fs_hz) or fs_hz <= 0 or len(trace) < 32:
        return {
            "freq_hz": np.array([]),
            "time_ms": np.array([]),
            "power_linear": np.array([]),
            "power_db": np.array([]),
        }

    if nfft_spec is not None and np.isfinite(float(nfft_spec)) and int(nfft_spec) > 0:
        nperseg = int(nfft_spec)
    else:
        nperseg = int(round(max(1.0, float(window_ms)) * float(fs_hz) / 1000.0))
    nperseg = min(max(4, nperseg), len(trace))
    overlap = min(max(float(overlap_frac), 0.0), 0.99)
    noverlap = min(int(round(overlap * nperseg)), max(0, nperseg - 1))
    try:
        from scipy.signal import spectrogram as scipy_spectrogram

        freqs_hz, times_s, power = scipy_spectrogram(
            trace,
            fs=fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
    except ImportError:
        freqs_hz, times_s, power = manual_spectrogram_psd(trace, fs_hz, nperseg, noverlap)
    t0_s = float(t_ms[0]) * 1e-3 if len(t_ms) else 0.0
    return {
        "freq_hz": np.asarray(freqs_hz, dtype=float),
        "time_ms": np.asarray((times_s + t0_s) * 1000.0, dtype=float),
        "power_linear": np.asarray(power, dtype=float),
        "power_db": np.asarray(db_from_linear_power(power), dtype=float),
        "fs_hz": float(fs_hz),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "window_ms": 1000.0 * float(nperseg) / float(fs_hz),
        "overlap_frac": float(noverlap) / float(nperseg) if nperseg else np.nan,
    }


def prepare_spectrogram_display_power(
    power_linear: np.ndarray,
    times_ms: np.ndarray,
    dbs_config: dict | None,
    *,
    mode: str = "relative",
    baseline_mode: str = "pre_stim",
    baseline_pre_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    times_ms = np.asarray(times_ms, dtype=float)
    power_linear = np.asarray(power_linear, dtype=float)
    stim_start_ms = float(dbs_config["start_ms"]) if dbs_config and dbs_config.get("enabled", False) else 0.0
    plot_times_ms = times_ms
    mode = str(mode).lower()
    baseline_mode = str(baseline_mode).lower()
    if mode != "relative":
        return plot_times_ms, db_from_linear_power(power_linear), "power (dB)", ""

    if baseline_mode == "pre_stim":
        if baseline_pre_ms is None:
            baseline_mask = times_ms < stim_start_ms
            baseline_note = "baseline: all available pre-DBS bins"
        else:
            pre_ms = max(0.0, float(baseline_pre_ms))
            baseline_mask = (times_ms < stim_start_ms) & (times_ms >= stim_start_ms - pre_ms)
            baseline_note = f"baseline: {pre_ms:.0f} ms pre-DBS"
        if not np.any(baseline_mask):
            baseline_mask = np.ones_like(plot_times_ms, dtype=bool)
            baseline_note = "baseline: all spectrogram bins (no pre-DBS bin)"
    else:
        baseline_mask = np.ones_like(plot_times_ms, dtype=bool)
        baseline_note = "baseline: time mean"

    with np.errstate(invalid="ignore"):
        baseline_lin = np.nanmean(power_linear[:, baseline_mask], axis=1, keepdims=True)
    ratio = power_linear / np.maximum(baseline_lin, 1e-30)
    display_db = db_from_linear_power(ratio)
    return plot_times_ms, display_db, "relative power (dB re baseline)", baseline_note


def plot_pipeline_spectrogram(
    results: list[dict],
    title: str,
    *,
    dbs_config: dict | None = None,
    nfft_spec: int | None = None,
    analysis_fs_hz: float | None = None,
    window_ms: float = 100.0,
    overlap_frac: float = 0.90,
    f_max_hz: float = 250.0,
    mode: str = "absolute",
    baseline_mode: str = "pre_stim",
    baseline_pre_ms: float | None = None,
    cmap: str = "viridis",
) -> None:
    import matplotlib.pyplot as plt

    t_ms, trace = population_mean_trace(results, "soma_v_mV")
    if len(t_ms) == 0 or len(trace) == 0:
        return
    spec = compute_pipeline_spectrogram_data(
        t_ms,
        trace,
        nfft_spec=nfft_spec,
        analysis_fs_hz=analysis_fs_hz,
        window_ms=window_ms,
        overlap_frac=overlap_frac,
    )
    freqs = np.asarray(spec.get("freq_hz", []), dtype=float)
    times = np.asarray(spec.get("time_ms", []), dtype=float)
    power = np.asarray(spec.get("power_linear", []), dtype=float)
    if power.ndim != 2:
        power_db = np.asarray(spec.get("power_db", []), dtype=float)
        if power_db.ndim == 2:
            power = np.power(10.0, power_db / 10.0)
    if len(freqs) == 0 or len(times) == 0 or power.shape != (len(freqs), len(times)):
        return

    keep = freqs <= float(f_max_hz)
    if np.any(keep):
        freqs = freqs[keep]
        power = power[keep, :]

    times_plot, power_plot, cbar_label, baseline_note = prepare_spectrogram_display_power(
        power,
        times,
        dbs_config,
        mode=mode,
        baseline_mode=baseline_mode,
        baseline_pre_ms=baseline_pre_ms,
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    if len(times_plot) == 1:
        half_width_ms = 500.0 * float(spec.get("nperseg", nfft_spec)) / float(spec.get("fs_hz", 1000.0))
        if len(freqs) > 1:
            df_hz = float(np.median(np.diff(freqs)))
        else:
            df_hz = 1.0
        extent = [
            float(times_plot[0] - half_width_ms),
            float(times_plot[0] + half_width_ms),
            max(0.0, float(freqs[0] - 0.5 * df_hz)),
            float(freqs[-1] + 0.5 * df_hz),
        ]
        mesh = ax.imshow(
            power_plot,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
        )
        ax.text(
            0.01,
            0.98,
            "single spectrogram window; run longer for time structure",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    else:
        mesh = ax.pcolormesh(times_plot, freqs, power_plot, shading="auto", cmap=cmap)
    if dbs_config and dbs_config.get("enabled", False):
        ax.axhline(float(dbs_config["freq_hz"]), color="cyan", ls="--", lw=1.1, alpha=0.95)
        ax.axvline(float(dbs_config["start_ms"]), color="white", ls="--", lw=0.9, alpha=0.9)
    if baseline_note:
        ax.text(
            0.01,
            0.02,
            baseline_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none"},
        )
    ax.set_title(
        f"{title} | window={float(spec.get('window_ms', np.nan)):.1f} ms, "
        f"fs={float(spec.get('fs_hz', np.nan)):.0f} Hz, bins={len(times)}"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0.0, min(float(f_max_hz), float(np.max(freqs))))
    fig.colorbar(mesh, ax=ax, pad=0.02, label=cbar_label)
    fig.tight_layout()
    finish_figure(fig, title)


def pulse_period_ms(dbs_config: dict | None) -> float:
    if not dbs_config or not dbs_config.get("enabled", False):
        return np.nan
    freq_hz = float(dbs_config["freq_hz"])
    if freq_hz <= 0:
        return np.nan
    return 1000.0 / freq_hz


def pta_rel_grid_ms(t_ms: np.ndarray, pre_ms: float, post_ms: float) -> np.ndarray:
    dt_ms = float(np.median(np.diff(t_ms))) if len(t_ms) >= 2 else 0.025
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        dt_ms = 0.025
    n = int(np.floor((float(pre_ms) + float(post_ms)) / dt_ms)) + 1
    return -float(pre_ms) + np.arange(n, dtype=float) * dt_ms


def extract_pulse_segment(
    t_ms: np.ndarray,
    trace: np.ndarray,
    pulse_ms: float,
    rel_grid_ms: np.ndarray,
    baseline_value: float | None = None,
) -> np.ndarray | None:
    target_t = float(pulse_ms) + rel_grid_ms
    if target_t[0] < float(t_ms[0]) or target_t[-1] > float(t_ms[-1]):
        return None
    segment = np.interp(target_t, t_ms, trace)
    if baseline_value is not None and np.isfinite(float(baseline_value)):
        segment = segment - float(baseline_value)
    else:
        pre = segment[rel_grid_ms < 0.0]
        if len(pre):
            segment = segment - float(np.median(pre))
    return segment


def baseline_value_before_time(
    t_ms: np.ndarray,
    trace: np.ndarray,
    event_ms: float,
    baseline_pre_ms: float | None,
) -> float | None:
    if baseline_pre_ms is not None and np.isfinite(float(baseline_pre_ms)) and float(baseline_pre_ms) > 0.0:
        pre_ms = float(baseline_pre_ms)
        mask = (t_ms < float(event_ms)) & (t_ms >= float(event_ms) - pre_ms)
        if np.any(mask):
            return float(np.median(trace[mask]))

    pre = trace[t_ms < float(event_ms)]
    if len(pre):
        return float(np.median(pre))
    return None


def extract_train_analysis_segment(
    t_ms: np.ndarray,
    trace: np.ndarray,
    pulse_times_ms: np.ndarray,
    pulse_index: int,
    rel_grid_ms: np.ndarray,
    *,
    window_scale: float = 1.0,
) -> np.ndarray | None:
    n_periods = max(1, int(round(float(window_scale))))
    if pulse_index < 1 or pulse_index + n_periods >= len(pulse_times_ms):
        return None
    if len(rel_grid_ms) < 4:
        return None

    pre_mask = rel_grid_ms < 0.0
    post_mask = rel_grid_ms >= 0.0
    if not np.any(pre_mask) or not np.any(post_mask):
        return None

    pulse_ms = float(pulse_times_ms[pulse_index])
    target_pre = pulse_ms + rel_grid_ms[pre_mask]
    if target_pre[0] < float(t_ms[0]) or target_pre[-1] > float(t_ms[-1]):
        return None

    segment = np.empty(len(rel_grid_ms), dtype=float)
    segment[pre_mask] = np.interp(target_pre, t_ms, trace)

    post_rows = []
    post_rel = rel_grid_ms[post_mask]
    for j in range(n_periods):
        post_pulse = float(pulse_times_ms[pulse_index + j])
        target_post = post_pulse + post_rel
        if target_post[0] < float(t_ms[0]) or target_post[-1] > float(t_ms[-1]):
            return None
        post_rows.append(np.interp(target_post, t_ms, trace))
    segment[post_mask] = np.nanmean(np.vstack(post_rows), axis=0)

    pre = segment[pre_mask]
    if len(pre):
        segment = segment - float(np.median(pre))
    return segment


def nan_spread(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rows.size == 0:
        return np.array([]), np.array([])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(rows, axis=0)
        if rows.shape[0] >= 2:
            spread = np.nanstd(rows, axis=0, ddof=1)
        else:
            spread = np.full(rows.shape[1], np.nan, dtype=float)
    return mean, spread


def compute_first_pulse_pta(
    results: list[dict],
    dbs_config: dict | None,
    *,
    baseline_pre_ms: float | None = 500.0,
) -> dict:
    if not results or not dbs_config or not dbs_config.get("enabled", False):
        return {}
    t_ms = np.asarray(results[0]["t_ms"], dtype=float)
    period = pulse_period_ms(dbs_config)
    if not np.isfinite(period):
        return {}
    rel_grid = pta_rel_grid_ms(t_ms, period, 3.0 * period)
    pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
    if len(pulses) == 0:
        return {}

    rows = []
    cell_indices = []
    first_pulse = float(pulses[0])
    for result in results:
        trace = np.asarray(result["soma_v_mV"], dtype=float)
        baseline = baseline_value_before_time(t_ms, trace, first_pulse, baseline_pre_ms)
        seg = extract_pulse_segment(t_ms, trace, first_pulse, rel_grid, baseline_value=baseline)
        if seg is None:
            continue
        rows.append(seg)
        cell_indices.append(int(result["cell_index"]))
    if not rows:
        return {}
    row_array = np.vstack(rows)
    mean, spread = nan_spread(row_array)
    return {
        "t_rel_ms": rel_grid,
        "segments": row_array,
        "mean": mean,
        "spread": spread,
        "cell_indices": cell_indices,
        "period_ms": float(period),
        "baseline_pre_ms": float(baseline_pre_ms) if baseline_pre_ms is not None else np.nan,
    }


def compute_train_pulse_pta(
    results: list[dict],
    dbs_config: dict | None,
    *,
    period_fraction: float = 0.95,
    window_scale: float = 1.0,
    display_post_periods: float = 3.0,
) -> dict:
    if not results or not dbs_config or not dbs_config.get("enabled", False):
        return {}
    t_ms = np.asarray(results[0]["t_ms"], dtype=float)
    period = pulse_period_ms(dbs_config)
    if not np.isfinite(period):
        return {}
    frac = min(1.0, max(0.0, float(period_fraction)))
    if frac <= 0.0:
        return {}
    analysis_window = frac * float(period)
    rel_grid = pta_rel_grid_ms(t_ms, analysis_window, analysis_window)
    display_rel_grid = pta_rel_grid_ms(t_ms, period, max(1.0, float(display_post_periods)) * period)
    pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
    if len(pulses) == 0:
        return {}

    cell_means = []
    display_cell_means = []
    cell_indices = []
    pulse_counts = []
    display_pulse_counts = []
    for result in results:
        trace = np.asarray(result["soma_v_mV"], dtype=float)
        local_rows = []
        for pulse_idx in range(1, len(pulses)):
            seg = extract_train_analysis_segment(
                t_ms,
                trace,
                pulses,
                pulse_idx,
                rel_grid,
                window_scale=window_scale,
            )
            if seg is not None:
                local_rows.append(seg)
        local_display_rows = []
        for pulse in pulses:
            seg = extract_pulse_segment(t_ms, trace, float(pulse), display_rel_grid)
            if seg is not None:
                local_display_rows.append(seg)
        if local_rows:
            local_array = np.vstack(local_rows)
            cell_means.append(np.nanmean(local_array, axis=0))
            cell_indices.append(int(result["cell_index"]))
            pulse_counts.append(int(local_array.shape[0]))
        if local_display_rows:
            display_array = np.vstack(local_display_rows)
            display_cell_means.append(np.nanmean(display_array, axis=0))
            display_pulse_counts.append(int(display_array.shape[0]))
    if not cell_means:
        return {}
    row_array = np.vstack(cell_means)
    mean, spread = nan_spread(row_array)
    if display_cell_means:
        display_row_array = np.vstack(display_cell_means)
        display_mean, display_spread = nan_spread(display_row_array)
    else:
        display_row_array = np.array([])
        display_mean = np.array([])
        display_spread = np.array([])
    return {
        "t_rel_ms": rel_grid,
        "segments": row_array,
        "mean": mean,
        "spread": spread,
        "display_t_rel_ms": display_rel_grid,
        "display_segments": display_row_array,
        "display_mean": display_mean,
        "display_spread": display_spread,
        "cell_indices": cell_indices,
        "pulse_counts": pulse_counts,
        "display_pulse_counts": display_pulse_counts,
        "period_ms": float(period),
        "period_fraction": float(frac),
        "window_scale": float(window_scale),
        "display_post_periods": float(display_post_periods),
    }


def plot_pta_result(pta: dict, title: str, *, max_rows: int = 40) -> None:
    import matplotlib.pyplot as plt

    display_t = np.asarray(pta.get("display_t_rel_ms", []), dtype=float)
    display_mean = np.asarray(pta.get("display_mean", []), dtype=float)
    if len(display_t) and display_mean.shape == display_t.shape:
        t_rel = display_t
        mean = display_mean
        spread = np.asarray(pta.get("display_spread", []), dtype=float)
        rows = np.asarray(pta.get("display_segments", []), dtype=float)
    else:
        t_rel = np.asarray(pta.get("t_rel_ms", []), dtype=float)
        mean = np.asarray(pta.get("mean", []), dtype=float)
        spread = np.asarray(pta.get("spread", []), dtype=float)
        rows = np.asarray(pta.get("segments", []), dtype=float)
    if len(t_rel) == 0 or mean.shape != t_rel.shape:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if rows.ndim == 2:
        for row in rows[:max_rows]:
            ax.plot(t_rel, row, color="tab:blue", alpha=0.15, lw=0.7)
    ax.plot(t_rel, mean, color="black", lw=2.0, label="mean")
    if spread.shape == mean.shape and np.any(np.isfinite(spread)):
        ax.fill_between(t_rel, mean - spread, mean + spread, color="black", alpha=0.2, label="SD")
    period = float(pta.get("period_ms", np.nan))
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.1, label="aligned pulse")
    if np.isfinite(period):
        for k in range(1, 4):
            x = k * period
            if t_rel[0] <= x <= t_rel[-1]:
                ax.axvline(x, color="tab:orange", ls="--", lw=1.0, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("Time from pulse (ms)")
    ax.set_ylabel("Soma Vm baseline-corrected (mV)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    finish_figure(fig, title)


def bandpass_trace(trace: np.ndarray, fs_hz: float, band_hz: tuple[float, float], order: int = 3) -> np.ndarray | None:
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None
    low, high = float(band_hz[0]), float(band_hz[1])
    nyq = 0.5 * fs_hz
    high = min(high, 0.95 * nyq)
    if not (0.0 < low < high < nyq):
        return None
    centered = np.asarray(trace, dtype=float) - float(np.mean(trace))
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(order, [low, high], btype="bandpass", fs=fs_hz, output="sos")
        return sosfiltfilt(sos, centered)
    except Exception:
        return fft_bandpass_trace(centered, fs_hz, (low, high))


def analytic_signal_for_display(filtered_trace: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import hilbert

        return hilbert(filtered_trace)
    except ImportError:
        return analytic_signal_fft(filtered_trace)


def stim_band_from_config(
    dbs_config: dict | None,
    fs_hz: float,
    half_band_hz: float,
    fallback_band_hz: tuple[float, float] = (13.0, 30.0),
) -> tuple[float, float]:
    if dbs_config and dbs_config.get("enabled", False):
        center_hz = float(dbs_config["freq_hz"])
        half = max(0.01, float(half_band_hz))
        low = max(0.5, center_hz - half)
        high = min(0.95 * 0.5 * float(fs_hz), center_hz + half)
        return low, high
    return fallback_band_hz


def pseudo_pulse_times_before_dbs(t_start_ms: float, dbs_config: dict | None) -> np.ndarray:
    if not dbs_config or not dbs_config.get("enabled", False):
        return np.array([], dtype=float)
    freq_hz = float(dbs_config["freq_hz"])
    if freq_hz <= 0:
        return np.array([], dtype=float)
    period_ms = 1000.0 / freq_hz
    stim_start_ms = float(dbs_config["start_ms"])
    n_pre = int(np.floor((stim_start_ms - float(t_start_ms)) / period_ms))
    if n_pre <= 0:
        return np.array([], dtype=float)
    pulses = stim_start_ms - period_ms * np.arange(n_pre, 0, -1, dtype=float)
    return pulses[pulses >= float(t_start_ms)]


def sample_unit_phase_at_times(
    t_ms: np.ndarray,
    unit_analytic: np.ndarray,
    sample_times_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_times_ms = np.asarray(sample_times_ms, dtype=float)
    keep = (sample_times_ms >= float(t_ms[0])) & (sample_times_ms <= float(t_ms[-1]))
    sample_times_ms = sample_times_ms[keep]
    if len(sample_times_ms) == 0:
        return sample_times_ms, np.array([], dtype=float)
    ur = np.interp(sample_times_ms, t_ms, np.real(unit_analytic))
    ui = np.interp(sample_times_ms, t_ms, np.imag(unit_analytic))
    phase = np.angle(ur + 1j * ui)
    return sample_times_ms, phase


def phase_locking_value(phases_rad: np.ndarray) -> float:
    phases_rad = np.asarray(phases_rad, dtype=float)
    if len(phases_rad) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * phases_rad))))


def plot_hilbert_phase(
    results: list[dict],
    dbs_config: dict | None,
    *,
    analysis_fs_hz: float | None = None,
    half_band_hz: float = 1.0,
) -> None:
    if not results or not dbs_config or not dbs_config.get("enabled", False):
        return
    import matplotlib.pyplot as plt

    t_ms, trace = population_mean_trace(results, "soma_v_mV")
    t_ms, trace, fs_hz = prepare_analysis_trace(t_ms, trace, target_fs_hz=analysis_fs_hz)
    freq_hz = float(dbs_config["freq_hz"])
    band_hz = stim_band_from_config(dbs_config, fs_hz, half_band_hz)
    filtered = bandpass_trace(trace, fs_hz, band_hz, order=3)
    if filtered is None:
        return
    analytic = analytic_signal_for_display(filtered)
    unit = analytic / np.maximum(np.abs(analytic), 1e-12)
    dbs_pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
    baseline_pulses = pseudo_pulse_times_before_dbs(float(t_ms[0]), dbs_config)
    _, dbs_phase = sample_unit_phase_at_times(t_ms, unit, dbs_pulses)
    _, baseline_phase = sample_unit_phase_at_times(t_ms, unit, baseline_pulses)
    if len(dbs_phase) == 0 and len(baseline_phase) == 0:
        return
    dbs_plv = phase_locking_value(dbs_phase)
    baseline_plv = phase_locking_value(baseline_phase)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    if len(baseline_phase):
        ax0.scatter(
            np.arange(1, len(baseline_phase) + 1),
            baseline_phase,
            s=18,
            color="tab:gray",
            alpha=0.75,
            label=f"baseline pseudo-pulses (PLV={baseline_plv:.3f}, n={len(baseline_phase)})",
        )
    if len(dbs_phase):
        ax0.scatter(
            np.arange(1, len(dbs_phase) + 1),
            dbs_phase,
            s=18,
            color="tab:blue",
            alpha=0.85,
            label=f"DBS pulses (PLV={dbs_plv:.3f}, n={len(dbs_phase)})",
        )
    ax0.set_xlabel("Pulse number")
    ax0.set_ylabel("Hilbert phase (rad)")
    ax0.set_ylim(-np.pi, np.pi)
    ax0.set_title("Stim-band phase at pulse times")
    ax0.legend(loc="best", fontsize=8)
    if len(baseline_phase):
        ax1.hist(baseline_phase, bins=24, range=(-np.pi, np.pi), color="tab:gray", alpha=0.55, label="baseline")
    if len(dbs_phase):
        ax1.hist(dbs_phase, bins=24, range=(-np.pi, np.pi), color="tab:blue", alpha=0.65, label="DBS")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_xlabel("Hilbert phase (rad)")
    ax1.set_ylabel("count")
    ax1.set_title(f"{band_hz[0]:.1f}-{band_hz[1]:.1f} Hz band")
    ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()
    finish_figure(fig, "Stim-band Hilbert phase")


def plot_hilbert_amplitude(
    results: list[dict],
    dbs_config: dict | None,
    *,
    analysis_fs_hz: float | None = None,
    band_hz: tuple[float, float] | None = None,
    half_band_hz: float = 1.0,
) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    t_ms, trace = population_mean_trace(results, "soma_v_mV")
    t_ms, trace, fs_hz = prepare_analysis_trace(t_ms, trace, target_fs_hz=analysis_fs_hz)
    if band_hz is None:
        band_hz = stim_band_from_config(dbs_config, fs_hz, half_band_hz)
    filtered = bandpass_trace(trace, fs_hz, band_hz, order=3)
    if filtered is None:
        return
    envelope = np.abs(analytic_signal_for_display(filtered))
    notes = []
    ylabel = "Hilbert amplitude (mV)"
    bandwidth_hz = float(band_hz[1]) - float(band_hz[0])
    total_s = (float(t_ms[-1]) - float(t_ms[0])) * 1e-3 if len(t_ms) >= 2 else 0.0
    min_s = 3.0 / max(bandwidth_hz, 1e-9)
    if total_s < min_s:
        notes.append(f"{total_s:.2f} s trace is short for a {bandwidth_hz:.1f} Hz Hilbert band")
    if notes:
        print("[WARN] Hilbert amplitude: " + "; ".join(notes))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(t_ms, envelope, color="black", lw=1.4)
    if dbs_config and dbs_config.get("enabled", False):
        pulses = dbs_first_phase_pulse_times_ms(float(t_ms[0]), float(t_ms[-1]), dbs_config)
        if len(pulses):
            ax.axvline(float(dbs_config["start_ms"]), color="tab:red", ls="--", lw=1.0, alpha=0.9)
            ax.axvline(float(pulses[-1]), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    if notes:
        ax.text(
            0.01,
            0.98,
            "\n".join(notes[:2]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    ax.set_title(f"Stim-band Hilbert amplitude envelope | {band_hz[0]:.1f}-{band_hz[1]:.1f} Hz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    finish_figure(fig, "Stim-band Hilbert amplitude")


def _prepare_entrainment_trace(
    t_ms: np.ndarray,
    trace_mV: np.ndarray,
    *,
    f_max_hz: float,
):
    trace = np.asarray(trace_mV, dtype=float)
    time_ms = np.asarray(t_ms, dtype=float)
    if len(trace) < 4 or len(time_ms) < 4:
        return time_ms, trace, 0.0

    dt_ms = float(np.mean(np.diff(time_ms)))
    if dt_ms <= 0:
        return time_ms, trace, 0.0
    fs_hz = 1000.0 / dt_ms

    trace = trace - float(np.mean(trace))

    try:
        from scipy.signal import butter, decimate, sosfiltfilt

        lowpass_hz = min(max(float(f_max_hz) * 1.5, 200.0), 0.45 * fs_hz)
        if 0.0 < lowpass_hz < 0.5 * fs_hz:
            sos = butter(4, lowpass_hz / (0.5 * fs_hz), btype="low", output="sos")
            trace = sosfiltfilt(sos, trace)

        target_fs_hz = max(800.0, 6.0 * float(f_max_hz))
        decim = max(1, int(np.floor(fs_hz / target_fs_hz)))
        if decim > 1 and len(trace) > 8 * decim:
            trace = decimate(trace, decim, ftype="fir", zero_phase=True)
            time_ms = np.linspace(float(time_ms[0]), float(time_ms[-1]), len(trace))
            fs_hz = fs_hz / decim
    except Exception:
        pass

    return time_ms, trace, fs_hz


def _compute_spectrogram(
    trace_mV: np.ndarray,
    fs_hz: float,
    *,
    window_s: float,
    overlap_frac: float = 0.9,
):
    try:
        from scipy.signal import spectrogram as scipy_spectrogram

        nperseg = max(64, min(len(trace_mV), int(round(window_s * fs_hz))))
        noverlap = min(int(round(overlap_frac * nperseg)), max(0, nperseg - 1))
        freqs_hz, times_s, power = scipy_spectrogram(
            trace_mV,
            fs=fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        return freqs_hz, times_s, power
    except Exception:
        import matplotlib.pyplot as plt

        nfft = max(64, min(len(trace_mV), int(round(window_s * fs_hz))))
        noverlap = min(int(round(overlap_frac * nfft)), max(0, nfft - 1))
        fig = plt.figure()
        spec, freqs_hz, times_s, _ = plt.specgram(
            trace_mV,
            Fs=fs_hz,
            NFFT=nfft,
            noverlap=noverlap,
            scale="linear",
            mode="psd",
        )
        plt.close(fig)
        return freqs_hz, times_s, spec


def plot_spectrogram(
    t_ms: np.ndarray,
    trace_mV: np.ndarray,
    title: str,
    *,
    dbs_config: dict | None = None,
    f_max_hz: float = 500.0,
) -> None:
    import matplotlib.pyplot as plt

    if len(t_ms) < 4 or len(trace_mV) < 4:
        return

    prep_t_ms, prep_trace_mV, fs_hz = _prepare_entrainment_trace(
        t_ms,
        trace_mV,
        f_max_hz=f_max_hz,
    )
    if fs_hz <= 0:
        return

    total_duration_s = max(1e-6, float(prep_t_ms[-1] - prep_t_ms[0]) * 1e-3)
    if dbs_config and dbs_config.get("enabled", False):
        stim_freq_hz = float(dbs_config["freq_hz"])
        window_s = max(0.15, 8.0 / max(stim_freq_hz, 1.0))
    else:
        window_s = 0.15
    window_s = min(window_s, max(0.06, 0.5 * total_duration_s))

    freqs_hz, times_s, power = _compute_spectrogram(
        np.asarray(prep_trace_mV, dtype=float),
        fs_hz,
        window_s=window_s,
    )
    times_ms = np.asarray(times_s, dtype=float) * 1000.0 + float(prep_t_ms[0])
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    power = np.asarray(power, dtype=float)
    if power.ndim != 2:
        return

    freq_mask = freqs_hz <= float(f_max_hz)
    if np.any(freq_mask):
        freqs_plot = freqs_hz[freq_mask]
        power_plot = power[freq_mask, :]
    else:
        freqs_plot = freqs_hz
        power_plot = power

    power_plot = np.maximum(power_plot, 1e-18)
    power_db = 10.0 * np.log10(power_plot)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    mesh = ax.pcolormesh(times_ms, freqs_plot, power_db, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0.0, float(np.max(freqs_plot)))
    if dbs_config and dbs_config.get("enabled", False):
        ax.axhline(float(dbs_config["freq_hz"]), color="cyan", ls="--", lw=1.2, alpha=0.95)
    fig.colorbar(mesh, ax=ax, pad=0.02, label="Power (dB)")
    fig.tight_layout()
    finish_figure(fig, title)


def plot_population_spectrogram(results: list[dict], title: str, *, dbs_config: dict | None = None) -> None:
    if not results:
        return
    t_ms = np.asarray(results[0]["t_ms"], dtype=float)
    soma_stack = np.vstack([np.asarray(result["soma_v_mV"], dtype=float) for result in results])
    mean_trace = np.mean(soma_stack, axis=0)
    plot_spectrogram(
        t_ms,
        mean_trace,
        title,
        dbs_config=dbs_config,
    )


def _paper_axis_message(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


def _label_paper_axis(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )


def _draw_paper_trace_panel(
    ax,
    results: list[dict],
    dbs_config: dict | None,
    *,
    trace_pre_ms: float,
    trace_post_ms: float,
) -> None:
    if not results:
        _paper_axis_message(ax, "no traces")
        return

    xlim = None
    if dbs_config and dbs_config.get("enabled", False):
        t_min = min(float(np.asarray(result["t_ms"], dtype=float)[0]) for result in results)
        t_max = max(float(np.asarray(result["t_ms"], dtype=float)[-1]) for result in results)
        start_ms = float(dbs_config["start_ms"])
        xlim = (
            max(t_min, start_ms - max(0.0, float(trace_pre_ms))),
            min(t_max, start_ms + max(0.0, float(trace_post_ms))),
        )

    for result in results:
        t_ms = np.asarray(result["t_ms"], dtype=float)
        v_mV = np.asarray(result["soma_v_mV"], dtype=float)
        if xlim is not None:
            keep = (t_ms >= xlim[0]) & (t_ms <= xlim[1])
            t_ms = t_ms[keep]
            v_mV = v_mV[keep]
        ax.plot(
            t_ms,
            v_mV,
            lw=0.9,
            alpha=0.85 if len(results) <= 10 else 0.45,
        )
    if dbs_config and dbs_config.get("enabled", False):
        ax.axvline(float(dbs_config["start_ms"]), color="tab:red", ls="--", lw=1.0)
        if dbs_config.get("stop_ms", None) is not None:
            ax.axvline(float(dbs_config["stop_ms"]), color="tab:orange", ls="--", lw=1.0)
    if xlim is not None and xlim[1] > xlim[0]:
        ax.set_xlim(*xlim)
    ax.set_title("Population trace (DBS onset)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Soma Vm (mV)")


def _draw_paper_pta_panel(ax, pta: dict, title: str, *, max_rows: int = 25) -> None:
    display_t = np.asarray(pta.get("display_t_rel_ms", []), dtype=float)
    display_mean = np.asarray(pta.get("display_mean", []), dtype=float)
    if len(display_t) and display_mean.shape == display_t.shape:
        t_rel = display_t
        mean = display_mean
        spread = np.asarray(pta.get("display_spread", []), dtype=float)
        rows = np.asarray(pta.get("display_segments", []), dtype=float)
    else:
        t_rel = np.asarray(pta.get("t_rel_ms", []), dtype=float)
        mean = np.asarray(pta.get("mean", []), dtype=float)
        spread = np.asarray(pta.get("spread", []), dtype=float)
        rows = np.asarray(pta.get("segments", []), dtype=float)

    if len(t_rel) == 0 or mean.shape != t_rel.shape:
        _paper_axis_message(ax, "PTA unavailable")
        return

    if rows.ndim == 2:
        for row in rows[:max_rows]:
            ax.plot(t_rel, row, color="tab:blue", alpha=0.12, lw=0.6)
    ax.plot(t_rel, mean, color="black", lw=1.8)
    if spread.shape == mean.shape and np.any(np.isfinite(spread)):
        ax.fill_between(t_rel, mean - spread, mean + spread, color="black", alpha=0.18)
    period = float(pta.get("period_ms", np.nan))
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.0)
    if np.isfinite(period):
        for k in range(1, 4):
            x = k * period
            if t_rel[0] <= x <= t_rel[-1]:
                ax.axvline(x, color="tab:orange", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time from pulse (ms)")
    ax.set_ylabel("Baseline-corrected Vm (mV)")


def _draw_paper_spectrogram_panel(
    fig,
    ax,
    results: list[dict],
    dbs_config: dict | None,
    *,
    nfft_spec: int | None,
    analysis_fs_hz: float | None,
    window_ms: float,
    overlap_frac: float,
    f_max_hz: float,
    mode: str,
    baseline_mode: str,
    baseline_pre_ms: float | None,
    cmap: str,
) -> None:
    t_ms, trace = population_mean_trace(results, "soma_v_mV")
    if len(t_ms) == 0 or len(trace) == 0:
        _paper_axis_message(ax, "spectrogram unavailable")
        return

    spec = compute_pipeline_spectrogram_data(
        t_ms,
        trace,
        nfft_spec=nfft_spec,
        analysis_fs_hz=analysis_fs_hz,
        window_ms=window_ms,
        overlap_frac=overlap_frac,
    )
    freqs = np.asarray(spec.get("freq_hz", []), dtype=float)
    times = np.asarray(spec.get("time_ms", []), dtype=float)
    power = np.asarray(spec.get("power_linear", []), dtype=float)
    if power.ndim != 2:
        power_db = np.asarray(spec.get("power_db", []), dtype=float)
        if power_db.ndim == 2:
            power = np.power(10.0, power_db / 10.0)
    if len(freqs) == 0 or len(times) == 0 or power.shape != (len(freqs), len(times)):
        _paper_axis_message(ax, "spectrogram unavailable")
        return

    keep = freqs <= float(f_max_hz)
    if np.any(keep):
        freqs = freqs[keep]
        power = power[keep, :]
    times_plot, power_plot, cbar_label, _ = prepare_spectrogram_display_power(
        power,
        times,
        dbs_config,
        mode=mode,
        baseline_mode=baseline_mode,
        baseline_pre_ms=baseline_pre_ms,
    )
    if len(times_plot) == 1:
        half_width_ms = 500.0 * float(spec.get("nperseg", nfft_spec)) / float(spec.get("fs_hz", 1000.0))
        df_hz = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
        extent = [
            float(times_plot[0] - half_width_ms),
            float(times_plot[0] + half_width_ms),
            max(0.0, float(freqs[0] - 0.5 * df_hz)),
            float(freqs[-1] + 0.5 * df_hz),
        ]
        mesh = ax.imshow(power_plot, aspect="auto", origin="lower", extent=extent, cmap=cmap)
    else:
        mesh = ax.pcolormesh(times_plot, freqs, power_plot, shading="auto", cmap=cmap)
    if dbs_config and dbs_config.get("enabled", False):
        ax.axhline(float(dbs_config["freq_hz"]), color="cyan", ls="--", lw=1.0, alpha=0.95)
        ax.axvline(float(dbs_config["start_ms"]), color="white", ls="--", lw=0.8, alpha=0.9)
        if dbs_config.get("stop_ms", None) is not None:
            ax.axvline(float(dbs_config["stop_ms"]), color="white", ls=":", lw=0.8, alpha=0.9)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0.0, min(float(f_max_hz), float(np.max(freqs))))
    fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.046, label=cbar_label)


def _draw_paper_hilbert_amplitude_panel(
    ax,
    results: list[dict],
    dbs_config: dict | None,
    *,
    analysis_fs_hz: float | None,
    half_band_hz: float,
) -> None:
    t_ms, trace = population_mean_trace(results, "soma_v_mV")
    t_ms, trace, fs_hz = prepare_analysis_trace(t_ms, trace, target_fs_hz=analysis_fs_hz)
    band_hz = stim_band_from_config(dbs_config, fs_hz, half_band_hz)
    filtered = bandpass_trace(trace, fs_hz, band_hz, order=3)
    if filtered is None:
        _paper_axis_message(ax, "Hilbert amplitude unavailable")
        return
    envelope = np.abs(analytic_signal_for_display(filtered))
    ax.plot(t_ms, envelope, color="black", lw=1.2)
    if dbs_config and dbs_config.get("enabled", False):
        ax.axvline(float(dbs_config["start_ms"]), color="tab:red", ls="--", lw=0.9)
        if dbs_config.get("stop_ms", None) is not None:
            ax.axvline(float(dbs_config["stop_ms"]), color="tab:orange", ls="--", lw=0.9)
    ax.set_title(f"Hilbert amplitude ({band_hz[0]:.1f}-{band_hz[1]:.1f} Hz)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")


def _draw_paper_synaptic_drive_panel(
    ax,
    results: list[dict],
    dbs_config: dict | None,
) -> None:
    hdp_results = collect_coupled_hdp_results(results)
    gpe_results = collect_coupled_gpe_results(results)
    dynamics = compute_pulse_recruitment_dynamics(results, hdp_results, gpe_results, dbs_config)
    if not dynamics.get("enabled", False):
        _paper_axis_message(ax, "synaptic drive unavailable")
        return

    x_s = np.asarray(dynamics["time_from_first_pulse_s"], dtype=float)
    hdp = dynamics["hdp"]
    gpe = dynamics["gpe"]
    plotted = False

    if int(hdp.get("n_axons", 0)) > 0:
        ax.plot(
            x_s,
            100.0 * np.asarray(hdp["drive_fraction"], dtype=float),
            color="tab:blue",
            lw=1.5,
            label="HDP AMPA",
        )
        plotted = True
    if int(gpe.get("n_axons", 0)) > 0:
        ax.plot(
            x_s,
            100.0 * np.asarray(gpe["drive_fraction"], dtype=float),
            color="tab:orange",
            lw=1.5,
            label="GPe GABA-A",
        )
        plotted = True

    if not plotted:
        _paper_axis_message(ax, "synaptic drive unavailable")
        return

    ax.set_title("Synaptic drive")
    ax.set_xlabel("Time from first DBS pulse (s)")
    ax.set_ylabel("Availability-weighted drive (%)")
    ax.set_ylim(-2.0, 102.0)
    ax.legend(loc="upper right", fontsize=8)


def _draw_paper_entrainment_distribution_panel(ax, results: list[dict], dbs_config: dict | None) -> None:
    rates = np.array(
        [
            float(result.get("during_dbs_rate_hz", np.nan))
            for result in results
            if np.isfinite(result.get("during_dbs_rate_hz", np.nan))
        ],
        dtype=float,
    )
    if len(rates) == 0:
        _paper_axis_message(ax, "rate distribution unavailable")
        return
    bins = min(24, max(8, int(np.sqrt(len(rates))) + 4))
    ax.hist(rates, bins=bins, color="0.35", edgecolor="white", alpha=0.9)
    if dbs_config and dbs_config.get("enabled", False):
        stim_freq_hz = float(dbs_config["freq_hz"])
        entrain_threshold_hz = 0.90 * stim_freq_hz
        entrained = int(np.count_nonzero(rates >= entrain_threshold_hz))
        ax.axvline(stim_freq_hz, color="tab:red", ls="--", lw=1.2, label="DBS frequency")
        ax.axvline(entrain_threshold_hz, color="tab:orange", ls=":", lw=1.2, label="90% DBS freq")
        ax.text(
            0.98,
            0.95,
            f"entrained: {entrained}/{len(rates)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Entrainment distribution")
    ax.set_xlabel("During-DBS firing rate (Hz)")
    ax.set_ylabel("Cells")


def plot_paper_figure(
    results: list[dict],
    dbs_config: dict | None,
    title: str,
    *,
    nfft_spec: int | None = None,
    analysis_fs_hz: float | None = None,
    spectrogram_window_ms: float = 100.0,
    spectrogram_overlap_frac: float = 0.95,
    spectrogram_fmax_hz: float = 500.0,
    spectrogram_mode: str = "absolute",
    spectrogram_rel_baseline: str = "pre_stim",
    spectrogram_baseline_pre_ms: float | None = None,
    spectrogram_cmap: str = "magma",
    spta_baseline_pre_ms: float = 500.0,
    mpta_period_fraction: float = 0.95,
    mpta_window_scale: float = 1.0,
    mpta_display_post_periods: float = 3.0,
    hilbert_half_band_hz: float = 1.0,
    paper_trace_pre_ms: float = 40.0,
    paper_trace_post_ms: float = 160.0,
) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    _draw_paper_trace_panel(
        axes[0],
        results,
        dbs_config,
        trace_pre_ms=paper_trace_pre_ms,
        trace_post_ms=paper_trace_post_ms,
    )
    _draw_paper_pta_panel(
        axes[1],
        compute_first_pulse_pta(results, dbs_config, baseline_pre_ms=spta_baseline_pre_ms),
        "First-pulse PTA",
    )
    _draw_paper_pta_panel(
        axes[2],
        compute_train_pulse_pta(
            results,
            dbs_config,
            period_fraction=mpta_period_fraction,
            window_scale=mpta_window_scale,
            display_post_periods=mpta_display_post_periods,
        ),
        "Pulse-train PTA",
    )
    _draw_paper_spectrogram_panel(
        fig,
        axes[3],
        results,
        dbs_config,
        nfft_spec=nfft_spec,
        analysis_fs_hz=analysis_fs_hz,
        window_ms=spectrogram_window_ms,
        overlap_frac=spectrogram_overlap_frac,
        f_max_hz=spectrogram_fmax_hz,
        mode=spectrogram_mode,
        baseline_mode=spectrogram_rel_baseline,
        baseline_pre_ms=spectrogram_baseline_pre_ms,
        cmap=spectrogram_cmap,
    )
    _draw_paper_synaptic_drive_panel(
        axes[4],
        results,
        dbs_config,
    )
    _draw_paper_entrainment_distribution_panel(axes[5], results, dbs_config)
    for label, ax in zip(("a", "b", "c", "d", "e", "f"), axes):
        _label_paper_axis(ax, label)
    fig.tight_layout()
    finish_figure(fig, title)


def plot_dbs_geometry(
    results: list[dict],
    dbs_config: dict,
    title: str,
    *,
    hdp_results: list[dict] | None = None,
    gpe_results: list[dict] | None = None,
    show_dendrites: bool = True,
    max_hdp_axons: int | str | None = None,
    max_gpe_axons: int | str | None = None,
    show_legend: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    from matplotlib.patches import Circle

    if not results:
        return
    hdp_results = hdp_results or []
    gpe_results = gpe_results or []

    def _limited_pathway(pathway_results: list[dict], max_axons: int | str | None) -> list[dict]:
        max_axons = parse_geometry_axon_limit(max_axons)
        if int(max_axons) < 0:
            return pathway_results
        max_axons = int(max_axons)
        if max_axons == 0:
            return []
        ordered = sorted(
            pathway_results,
            key=lambda item: (
                not bool(item.get("activated", False)),
                int(item.get("axon_index", 0)),
            ),
        )
        return ordered[:max_axons]

    hdp_plot_results = _limited_pathway(hdp_results, max_hdp_axons)
    gpe_plot_results = _limited_pathway(gpe_results, max_gpe_axons)
    hdp_parent_color = "#7a1fa2"
    hdp_collateral_color = "#ff7a00"
    gpe_parent_color = "#0057b8"
    gpe_collateral_color = "#00b8d9"
    inactive_alpha = 0.70
    active_alpha = 0.96

    def _cell_positions_for_geometry(result: dict) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(result["segment_positions_mm"], dtype=float)
        phi = np.asarray(result["segment_phi_mV"], dtype=float)
        if show_dendrites:
            return positions, phi
        kinds = np.asarray(result["segment_kinds"])
        mask = kinds != "dend"
        return positions[mask], phi[mask]

    fig = plt.figure(figsize=(12.5, 5.7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    position_blocks = []
    phi_blocks = []
    for result in results:
        positions, phi = _cell_positions_for_geometry(result)
        if len(positions):
            position_blocks.append(positions)
            phi_blocks.append(phi)
    position_blocks.extend(
        [np.asarray(result["segment_positions_mm"], dtype=float) for result in hdp_plot_results if len(result.get("segment_positions_mm", []))]
    )
    position_blocks.extend(
        [np.asarray(result["segment_positions_mm"], dtype=float) for result in gpe_plot_results if len(result.get("segment_positions_mm", []))]
    )
    phi_blocks.extend(
        [np.asarray(result["segment_phi_mV"], dtype=float) for result in hdp_plot_results if len(result.get("segment_phi_mV", []))]
    )
    phi_blocks.extend(
        [np.asarray(result["segment_phi_mV"], dtype=float) for result in gpe_plot_results if len(result.get("segment_phi_mV", []))]
    )
    all_positions = np.vstack(position_blocks)
    all_phi = np.concatenate(phi_blocks)
    vmin = float(np.min(all_phi))
    vmax = float(np.max(all_phi))

    scat = ax3d.scatter(
        all_positions[:, 0],
        all_positions[:, 1],
        all_positions[:, 2],
        c=all_phi,
        cmap="coolwarm",
        s=8,
        alpha=0.14,
        vmin=vmin,
        vmax=vmax,
    )

    elec_a = np.array(dbs_config["electrode_a_pos_mm"], dtype=float)
    elec_b = np.array(dbs_config["electrode_b_pos_mm"], dtype=float)
    fiber_center = np.array(dbs_config["fiber_center_mm"], dtype=float)
    fiber_radius = float(dbs_config["fiber_radius_mm"])
    ax3d.scatter(*elec_a, color="royalblue", s=120, marker="o", label="electrode A")
    ax3d.scatter(*elec_b, color="crimson", s=120, marker="o", label="electrode B")
    ax3d.scatter(*fiber_center, color="mediumseagreen", s=110, marker="^", label="fiber center")

    for result in results:
        positions = np.asarray(result["segment_positions_mm"], dtype=float)
        kinds = np.asarray(result["segment_kinds"])
        soma = np.array(result["placement"]["soma_pos_mm"], dtype=float)
        axon_dir = np.array(result["placement"]["axon_dir"], dtype=float)

        soma_mask = kinds == "soma"
        dend_mask = kinds == "dend"
        axon_mask = kinds == "axon"

        if show_dendrites and np.any(dend_mask):
            ax3d.scatter(
                positions[dend_mask, 0],
                positions[dend_mask, 1],
                positions[dend_mask, 2],
                color="0.55",
                s=10,
                alpha=0.9,
                label="dendrites" if result["cell_index"] == 0 else None,
            )
        if np.any(soma_mask):
            ax3d.scatter(
                positions[soma_mask, 0],
                positions[soma_mask, 1],
                positions[soma_mask, 2],
                color="black",
                s=30,
                alpha=0.95,
                label="soma" if result["cell_index"] == 0 else None,
            )
        if np.any(axon_mask):
            axon_pos = positions[axon_mask]
            axon_proj = (axon_pos - soma) @ axon_dir
            order = np.argsort(axon_proj)
            axon_pos = axon_pos[order]
            ax3d.plot(
                axon_pos[:, 0],
                axon_pos[:, 1],
                axon_pos[:, 2],
                color="limegreen",
                lw=2.2,
                alpha=0.95,
                label="axon" if result["cell_index"] == 0 else None,
            )
            ax3d.scatter(
                axon_pos[:, 0],
                axon_pos[:, 1],
                axon_pos[:, 2],
                color="limegreen",
                s=14,
                alpha=0.95,
            )

        ax3d.scatter(*soma, color="black", s=45, marker="x")

    for result in hdp_plot_results:
        parent_points = np.asarray(result.get("parent_points_mm", []), dtype=float)
        collateral_points = np.asarray(result.get("collateral_points_mm", []), dtype=float)
        activated = bool(result.get("activated", False))
        parent_color = hdp_parent_color
        collateral_color = hdp_collateral_color
        alpha = active_alpha if activated else inactive_alpha
        lw = 2.4 if activated else 1.1
        if parent_points.ndim == 2 and parent_points.shape[0] >= 2:
            ax3d.plot(
                parent_points[:, 0],
                parent_points[:, 1],
                parent_points[:, 2],
                color=parent_color,
                lw=lw,
                alpha=alpha,
                label="HDP parent" if result["axon_index"] == 0 else None,
            )
        if collateral_points.ndim == 2 and collateral_points.shape[0] >= 2:
            ax3d.plot(
                collateral_points[:, 0],
                collateral_points[:, 1],
                collateral_points[:, 2],
                color=collateral_color,
                lw=lw,
                ls="--",
                alpha=alpha,
                label="HDP collateral" if result["axon_index"] == 0 else None,
            )
            terminal = np.asarray(result.get("terminal_point_mm", []), dtype=float)
            if terminal.shape == (3,):
                ax3d.scatter(
                    terminal[0],
                    terminal[1],
                    terminal[2],
                    color=collateral_color,
                    s=24 if activated else 12,
                    alpha=alpha,
                    marker="s",
                )

    for result in gpe_plot_results:
        parent_points = np.asarray(result.get("parent_points_mm", []), dtype=float)
        collateral_points = np.asarray(result.get("collateral_points_mm", []), dtype=float)
        activated = bool(result.get("activated", False))
        parent_color = gpe_parent_color
        collateral_color = gpe_collateral_color
        alpha = active_alpha if activated else inactive_alpha
        lw = 2.4 if activated else 1.1
        if parent_points.ndim == 2 and parent_points.shape[0] >= 2:
            ax3d.plot(
                parent_points[:, 0],
                parent_points[:, 1],
                parent_points[:, 2],
                color=parent_color,
                lw=lw,
                alpha=alpha,
                label="GPe parent" if result["axon_index"] == 0 else None,
            )
        if collateral_points.ndim == 2 and collateral_points.shape[0] >= 2:
            ax3d.plot(
                collateral_points[:, 0],
                collateral_points[:, 1],
                collateral_points[:, 2],
                color=collateral_color,
                lw=lw,
                ls="--",
                alpha=alpha,
                label="GPe collateral" if result["axon_index"] == 0 else None,
            )
            terminal = np.asarray(result.get("terminal_point_mm", []), dtype=float)
            if terminal.shape == (3,):
                ax3d.scatter(
                    terminal[0],
                    terminal[1],
                    terminal[2],
                    color=collateral_color,
                    s=24 if activated else 12,
                    alpha=alpha,
                    marker="D",
                )

    ax3d.set_title(title)
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3d.zaxis.set_major_locator(MaxNLocator(nbins=5))
    if show_legend:
        ax3d.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.96),
            borderaxespad=0.0,
            fontsize=7,
            markerscale=0.75,
            handlelength=1.4,
            labelspacing=0.25,
            borderpad=0.3,
            frameon=True,
        )
    fig.colorbar(scat, ax=ax3d, shrink=0.7, pad=0.08, label="phi (mV) at +DBS amp")

    x_min = min(
        float(np.min(all_positions[:, 0])),
        float(elec_a[0]),
        float(elec_b[0]),
        float(fiber_center[0] - fiber_radius),
    ) - 0.25
    x_max = max(
        float(np.max(all_positions[:, 0])),
        float(elec_a[0]),
        float(elec_b[0]),
        float(fiber_center[0] + fiber_radius),
    ) + 0.25
    y_min = min(
        float(np.min(all_positions[:, 1])),
        float(elec_a[1]),
        float(elec_b[1]),
        float(fiber_center[1] - fiber_radius),
    ) - 0.25
    y_max = max(
        float(np.max(all_positions[:, 1])),
        float(elec_a[1]),
        float(elec_b[1]),
        float(fiber_center[1] + fiber_radius),
    ) + 0.25
    z_plane = float(np.mean([result["placement"]["soma_pos_mm"][2] for result in results]))
    first_phase_current = first_phase_sign_uA(dbs_config) * float(dbs_config["amp_uA"])

    xs = np.linspace(x_min, x_max, 120)
    ys = np.linspace(y_min, y_max, 120)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_phi = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            grid_phi[i, j] = bipolar_phi_uA(
                first_phase_current,
                (float(grid_x[i, j]), float(grid_y[i, j]), z_plane),
                dbs_config,
            )

    contour = ax2d.contourf(grid_x, grid_y, grid_phi, levels=30, cmap="coolwarm")
    ax2d.scatter(elec_a[0], elec_a[1], color="royalblue", s=80, marker="o")
    ax2d.scatter(elec_b[0], elec_b[1], color="crimson", s=80, marker="o")
    ax2d.add_patch(
        Circle(
            (float(fiber_center[0]), float(fiber_center[1])),
            fiber_radius,
            facecolor="mediumseagreen",
            edgecolor="forestgreen",
            alpha=0.14,
            lw=1.6,
        )
    )
    ax2d.scatter(fiber_center[0], fiber_center[1], color="forestgreen", s=60, marker="^")
    for result in results:
        soma = np.array(result["placement"]["soma_pos_mm"], dtype=float)
        axon_dir = np.array(result["placement"]["axon_dir"], dtype=float)
        ax2d.scatter(soma[0], soma[1], color="black", s=24, marker="x")
        ax2d.arrow(
            soma[0],
            soma[1],
            0.12 * axon_dir[0],
            0.12 * axon_dir[1],
            color="limegreen",
            width=0.002,
            head_width=0.02,
            length_includes_head=True,
        )
    for result in hdp_plot_results:
        parent_points = np.asarray(result.get("parent_points_mm", []), dtype=float)
        collateral_points = np.asarray(result.get("collateral_points_mm", []), dtype=float)
        activated = bool(result.get("activated", False))
        parent_color = hdp_parent_color
        collateral_color = hdp_collateral_color
        alpha = active_alpha if activated else inactive_alpha
        if parent_points.ndim == 2 and parent_points.shape[0] >= 2:
            ax2d.plot(parent_points[:, 0], parent_points[:, 1], color=parent_color, lw=1.5, alpha=alpha)
        if collateral_points.ndim == 2 and collateral_points.shape[0] >= 2:
            ax2d.plot(collateral_points[:, 0], collateral_points[:, 1], color=collateral_color, lw=1.5, ls="--", alpha=alpha)
            terminal = np.asarray(result.get("terminal_point_mm", []), dtype=float)
            if terminal.shape == (3,):
                ax2d.scatter(terminal[0], terminal[1], color=collateral_color, s=16 if activated else 8, marker="s", alpha=alpha)
    for result in gpe_plot_results:
        parent_points = np.asarray(result.get("parent_points_mm", []), dtype=float)
        collateral_points = np.asarray(result.get("collateral_points_mm", []), dtype=float)
        activated = bool(result.get("activated", False))
        parent_color = gpe_parent_color
        collateral_color = gpe_collateral_color
        alpha = active_alpha if activated else inactive_alpha
        if parent_points.ndim == 2 and parent_points.shape[0] >= 2:
            ax2d.plot(parent_points[:, 0], parent_points[:, 1], color=parent_color, lw=1.5, alpha=alpha)
        if collateral_points.ndim == 2 and collateral_points.shape[0] >= 2:
            ax2d.plot(collateral_points[:, 0], collateral_points[:, 1], color=collateral_color, lw=1.5, ls="--", alpha=alpha)
            terminal = np.asarray(result.get("terminal_point_mm", []), dtype=float)
            if terminal.shape == (3,):
                ax2d.scatter(terminal[0], terminal[1], color=collateral_color, s=16 if activated else 8, marker="D", alpha=alpha)
    ax2d.set_title(f"Field slice at z={z_plane * 1000.0:.0f} um")
    ax2d.set_xlabel("x (mm)")
    ax2d.set_ylabel("y (mm)")
    ax2d.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2d.yaxis.set_major_locator(MultipleLocator(0.5))
    fig.colorbar(contour, ax=ax2d, shrink=0.85, pad=0.02, label="phi (mV)")

    fig.tight_layout()
    finish_figure(fig, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="2024 Chen et al. STN model")
    parser.add_argument("--model", choices=("gw", "detail"), default=DEFAULT_RUN_CONFIG["model"])
    parser.add_argument(
        "--morphology",
        default=DEFAULT_RUN_CONFIG["morphology"],
        help="SWC file name/path for detail model, or 'auto' to cycle through available reconstructions",
    )
    parser.add_argument("--n-cells", type=int, default=DEFAULT_RUN_CONFIG["n_cells"])
    parser.add_argument("--param-index", type=int, default=DEFAULT_RUN_CONFIG["param_index"])
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override an optimized parameter, e.g. --set soma_gNa=0.007",
    )
    parser.add_argument(
        "--param-jitter-frac",
        type=float,
        default=DEFAULT_RUN_CONFIG["param_jitter_frac"],
        help="Per-cell random multiplicative jitter fraction, e.g. 0.1 = +/-10%%",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_CONFIG["seed"])
    parser.add_argument("--tstop", type=float, default=DEFAULT_RUN_CONFIG["tstop_ms"])
    parser.add_argument("--amp", type=float, default=DEFAULT_RUN_CONFIG["amp_nA"], help="IClamp amplitude in nA")
    parser.add_argument("--delay", type=float, default=DEFAULT_RUN_CONFIG["delay_ms"])
    parser.add_argument("--dur", type=float, default=DEFAULT_RUN_CONFIG["dur_ms"])
    parser.add_argument("--temp", type=float, default=DEFAULT_RUN_CONFIG["temperature_c"])
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot"],
    )
    parser.add_argument(
        "--plot-geometry",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_geometry"],
        help="Show a static geometry/field view",
    )
    parser.add_argument(
        "--geometry-show-dendrites",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["geometry_show_dendrites"],
        help="Show dendritic segment cloud in the static geometry view",
    )
    parser.add_argument(
        "--geometry-max-hdp-axons",
        type=parse_geometry_axon_limit,
        default=DEFAULT_RUN_CONFIG["geometry_max_hdp_axons"],
        help="HDP axons to draw in the geometry view: none/0, all/-1, or a positive number",
    )
    parser.add_argument(
        "--geometry-max-gpe-axons",
        type=parse_geometry_axon_limit,
        default=DEFAULT_RUN_CONFIG["geometry_max_gpe_axons"],
        help="GPe axons to draw in the geometry view: none/0, all/-1, or a positive number",
    )
    parser.add_argument(
        "--geometry-show-legend",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["geometry_show_legend"],
        help="Show legend in the static geometry view",
    )
    parser.add_argument(
        "--plot-spectrogram",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_spectrogram"],
        help="Show a pipeline-style spectrogram of the soma trace (or mean soma trace for populations)",
    )
    parser.add_argument(
        "--plot-spta",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_spta"],
        help="Show first-pulse PTA of soma voltage",
    )
    parser.add_argument(
        "--plot-mpta",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_mpta"],
        help="Show pulse-train PTA of soma voltage",
    )
    parser.add_argument(
        "--plot-hilbert-phase",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_hilbert_phase"],
        help="Show stim-band Hilbert phase sampled at DBS pulses",
    )
    parser.add_argument(
        "--plot-hilbert-amp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_hilbert_amp"],
        help="Show stim-band Hilbert amplitude envelope",
    )
    parser.add_argument(
        "--plot-post-rate-distribution",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_post_rate_distribution"],
        help="Show histogram of post-DBS firing rates across STN cells",
    )
    parser.add_argument(
        "--plot-plv-histogram",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_plv_histogram"],
        help="Show histogram of per-cell PLV at the DBS frequency",
    )
    parser.add_argument(
        "--plot-recruitment-dynamics",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["plot_recruitment_dynamics"],
        help="Show pulse-by-pulse STN/HDP/GPe recruitment and depression-weighted synaptic drive",
    )
    parser.add_argument(
        "--paper-plot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["paper_plot"],
        help="Show one paper-style multi-panel figure instead of separate trace/PTA/spectrogram summary plots",
    )
    parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["save_figure"],
        help="Save each generated figure as a 300 dpi PNG before showing it",
    )
    parser.add_argument(
        "--save-figure-dir",
        default=DEFAULT_RUN_CONFIG["save_figure_dir"],
        help="Directory for saved figures when --save-figure is enabled",
    )
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["print_summary"],
        help="Print a compact paper-style population summary",
    )
    parser.add_argument(
        "--print-activation-origin",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["print_activation_origin"],
        help="Print soma/AIS/axon activation-origin diagnostics",
    )
    parser.add_argument(
        "--print-synapse-details",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_CONFIG["print_synapse_details"],
        help="Print detailed HDP/GPe synapse and axon summaries after the compact cell table",
    )
    parser.add_argument("--analysis-fs-hz", type=float, default=DEFAULT_RUN_CONFIG["analysis_fs_hz"])
    parser.add_argument("--spectrogram-nfft", type=int, default=DEFAULT_RUN_CONFIG["spectrogram_nfft"])
    parser.add_argument("--spectrogram-window-ms", type=float, default=DEFAULT_RUN_CONFIG["spectrogram_window_ms"])
    parser.add_argument("--spectrogram-overlap-frac", type=float, default=DEFAULT_RUN_CONFIG["spectrogram_overlap_frac"])
    parser.add_argument("--spectrogram-fmax-hz", type=float, default=DEFAULT_RUN_CONFIG["spectrogram_fmax_hz"])
    parser.add_argument(
        "--spectrogram-mode",
        choices=["absolute", "relative"],
        default=DEFAULT_RUN_CONFIG["spectrogram_mode"],
    )
    parser.add_argument(
        "--spectrogram-rel-baseline",
        choices=["pre_stim", "time_mean"],
        default=DEFAULT_RUN_CONFIG["spectrogram_rel_baseline"],
    )
    parser.add_argument(
        "--spectrogram-baseline-pre-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["spectrogram_baseline_pre_ms"],
    )
    parser.add_argument("--spectrogram-cmap", default=DEFAULT_RUN_CONFIG["spectrogram_cmap"])
    parser.add_argument(
        "--spta-baseline-pre-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["spta_baseline_pre_ms"],
        help="Baseline window before the first DBS pulse for sPTA; <=0 uses all available pre-pulse data",
    )
    parser.add_argument(
        "--mpta-period-fraction",
        type=float,
        default=DEFAULT_RUN_CONFIG["mpta_period_fraction"],
        help="Fraction of one inter-pulse interval used for mPTA analysis, matching the pipeline default",
    )
    parser.add_argument(
        "--mpta-window-scale",
        type=float,
        default=DEFAULT_RUN_CONFIG["mpta_window_scale"],
        help="Number of post-pulse periods to fold into the mPTA analysis response",
    )
    parser.add_argument(
        "--mpta-display-post-periods",
        type=float,
        default=DEFAULT_RUN_CONFIG["mpta_display_post_periods"],
        help="Post-pulse periods shown in the mPTA plot without changing the one-period analysis",
    )
    parser.add_argument("--hilbert-half-band-hz", type=float, default=DEFAULT_RUN_CONFIG["hilbert_half_band_hz"])
    parser.add_argument(
        "--dbs-rate-edge-window-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["dbs_rate_edge_window_ms"],
        help="Window length used for early/late during-DBS firing-rate summaries",
    )
    parser.add_argument(
        "--paper-trace-pre-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["paper_trace_pre_ms"],
        help="Milliseconds shown before DBS onset in paper-plot trace panel",
    )
    parser.add_argument(
        "--paper-trace-post-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["paper_trace_post_ms"],
        help="Milliseconds shown after DBS onset in paper-plot trace panel",
    )
    parser.add_argument("--show-params", action="store_true")
    parser.add_argument("--show-defaults", action="store_true")
    parser.add_argument("--list-morphologies", action="store_true")
    parser.add_argument("--suite", action="store_true", default=DEFAULT_RUN_CONFIG["suite"], help="Run the paper-style protocol suite")
    parser.add_argument(
        "--dbs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DBS_CONFIG["enabled"],
        help="Enable analytic bipolar DBS field",
    )
    parser.add_argument(
        "--a-positive-first",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DBS_CONFIG["a_positive_first"],
        help="If true, electrode A is positive/anodic in the first phase",
    )
    parser.add_argument("--dbs-start", type=float, default=DEFAULT_RUN_CONFIG["dbs_start_ms"])
    parser.add_argument(
        "--dbs-stop-ms",
        type=float,
        default=DEFAULT_RUN_CONFIG["dbs_stop_ms"],
        help="Absolute DBS stop time in ms; unset or <=0 keeps DBS on until simulation end",
    )
    parser.add_argument("--dbs-freq", type=float, default=DEFAULT_DBS_CONFIG["freq_hz"])
    parser.add_argument("--dbs-pw", type=float, default=DEFAULT_DBS_CONFIG["pw_ms"])
    parser.add_argument("--dbs-ipg", type=float, default=DEFAULT_DBS_CONFIG["ipg_ms"])
    parser.add_argument("--dbs-amp-uA", type=float, default=DEFAULT_DBS_CONFIG["amp_uA"])
    parser.add_argument(
        "--omit-pulse",
        type=int,
        default=DEFAULT_DBS_CONFIG["omit_pulse"],
        help="Omit one 1-based DBS pulse number after DBS start; <=0 disables omission",
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_DBS_CONFIG["sigma_S_per_m"])
    parser.add_argument("--electrode-a", nargs=3, type=float, default=DEFAULT_DBS_CONFIG["electrode_a_pos_mm"])
    parser.add_argument("--electrode-b", nargs=3, type=float, default=DEFAULT_DBS_CONFIG["electrode_b_pos_mm"])
    parser.add_argument("--fiber-center", nargs=3, type=float, default=DEFAULT_DBS_CONFIG["fiber_center_mm"])
    parser.add_argument("--fiber-radius-mm", type=float, default=DEFAULT_DBS_CONFIG["fiber_radius_mm"])
    parser.add_argument("--soma-pos", nargs=3, type=float, default=None)
    parser.add_argument("--axon-dir", nargs=3, type=float, default=None)
    parser.add_argument("--dend-dir", nargs=3, type=float, default=None)
    parser.add_argument("--shell-min-r-mm", type=float, default=DEFAULT_DBS_CONFIG["min_r_mm"])
    parser.add_argument("--shell-max-r-mm", type=float, default=DEFAULT_DBS_CONFIG["max_r_mm"])
    parser.add_argument(
        "--hdp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HDP_CONFIG["enabled"],
        help="Enable standalone parametric hyperdirect pathway axons",
    )
    parser.add_argument(
        "--hdp-axons",
        type=int,
        default=DEFAULT_HDP_CONFIG["n_axons"],
        help="Number of standalone parametric hyperdirect pathway axons to simulate",
    )
    parser.add_argument("--hdp-parent-diameter-um", type=float, default=DEFAULT_HDP_CONFIG["parent_diameter_um"])
    parser.add_argument("--hdp-diameter-jitter-frac", type=float, default=DEFAULT_HDP_CONFIG["diameter_jitter_frac"])
    parser.add_argument("--hdp-collateral-diameter-frac", type=float, default=DEFAULT_HDP_CONFIG["collateral_diameter_frac"])
    parser.add_argument("--hdp-parent-length-mm", type=float, default=DEFAULT_HDP_CONFIG["parent_length_mm"])
    parser.add_argument("--hdp-parent-nodes", type=int, default=DEFAULT_HDP_CONFIG["parent_nodes"])
    parser.add_argument("--hdp-collateral-nodes", type=int, default=DEFAULT_HDP_CONFIG["collateral_nodes"])
    parser.add_argument("--hdp-parent-pass-radius-mm", type=float, default=DEFAULT_HDP_CONFIG["parent_pass_radius_mm"])
    parser.add_argument("--hdp-terminal-radius-mm", type=float, default=DEFAULT_HDP_CONFIG["terminal_radius_mm"])
    parser.add_argument("--hdp-direction-jitter-frac", type=float, default=DEFAULT_HDP_CONFIG["direction_jitter_frac"])
    parser.add_argument(
        "--hdp-synapses",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HDP_CONFIG["synapses_enabled"],
        help="Couple HDP terminal spikes to STN glutamatergic synapses during each STN-cell simulation",
    )
    parser.add_argument(
        "--hdp-inputs-per-cell",
        type=int,
        default=DEFAULT_HDP_CONFIG["inputs_per_cell"],
        help="Number of HDP terminal-to-STN synapses per STN cell",
    )
    parser.add_argument(
        "--hdp-syn-target",
        choices=("distal", "dendrite", "soma"),
        default=DEFAULT_HDP_CONFIG["syn_target"],
        help="STN target compartment class for HDP synapses; soma is a debug option",
    )
    parser.add_argument(
        "--hdp-syn-distal-frac",
        type=float,
        default=DEFAULT_HDP_CONFIG["syn_distal_frac"],
        help="Automatic distal cutoff as a fraction of max dendritic path length",
    )
    parser.add_argument(
        "--hdp-syn-min-dist-um",
        type=float,
        default=DEFAULT_HDP_CONFIG["syn_min_dist_um"],
        help="Override distal dendrite cutoff in um; default uses hdp-syn-distal-frac",
    )
    parser.add_argument("--hdp-syn-weight-uS", type=float, default=DEFAULT_HDP_CONFIG["syn_weight_uS"])
    parser.add_argument("--hdp-syn-tau1-ms", type=float, default=DEFAULT_HDP_CONFIG["syn_tau1_ms"])
    parser.add_argument("--hdp-syn-tau2-ms", type=float, default=DEFAULT_HDP_CONFIG["syn_tau2_ms"])
    parser.add_argument("--hdp-syn-delay-ms", type=float, default=DEFAULT_HDP_CONFIG["syn_delay_ms"])
    parser.add_argument(
        "--hdp-syn-depression",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HDP_CONFIG["syn_depression_enabled"],
        help="Use depression-only DepExp2Syn for HDP AMPA synapses",
    )
    parser.add_argument("--hdp-syn-depression-u", type=float, default=DEFAULT_HDP_CONFIG["syn_depression_u"])
    parser.add_argument(
        "--hdp-syn-depression-tau-rec-ms",
        type=float,
        default=DEFAULT_HDP_CONFIG["syn_depression_tau_rec_ms"],
    )
    parser.add_argument(
        "--hdp-syn-depression-tau-facil-ms",
        type=float,
        default=DEFAULT_HDP_CONFIG["syn_depression_tau_facil_ms"],
        help="Recorded no-facilitation setting; DepExp2Syn uses U and tau_rec only",
    )
    parser.add_argument(
        "--gpe",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GPE_CONFIG["enabled"],
        help="Enable selected parametric GPe->STN axons",
    )
    parser.add_argument("--gpe-axons", type=int, default=DEFAULT_GPE_CONFIG["n_axons"])
    parser.add_argument("--gpe-parent-diameter-um", type=float, default=DEFAULT_GPE_CONFIG["parent_diameter_um"])
    parser.add_argument("--gpe-diameter-jitter-frac", type=float, default=DEFAULT_GPE_CONFIG["diameter_jitter_frac"])
    parser.add_argument("--gpe-collateral-diameter-frac", type=float, default=DEFAULT_GPE_CONFIG["collateral_diameter_frac"])
    parser.add_argument("--gpe-parent-length-mm", type=float, default=DEFAULT_GPE_CONFIG["parent_length_mm"])
    parser.add_argument("--gpe-parent-nodes", type=int, default=DEFAULT_GPE_CONFIG["parent_nodes"])
    parser.add_argument("--gpe-collateral-nodes", type=int, default=DEFAULT_GPE_CONFIG["collateral_nodes"])
    parser.add_argument("--gpe-parent-pass-radius-mm", type=float, default=DEFAULT_GPE_CONFIG["parent_pass_radius_mm"])
    parser.add_argument("--gpe-terminal-radius-mm", type=float, default=DEFAULT_GPE_CONFIG["terminal_radius_mm"])
    parser.add_argument("--gpe-direction-jitter-frac", type=float, default=DEFAULT_GPE_CONFIG["direction_jitter_frac"])
    parser.add_argument(
        "--gpe-synapses",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GPE_CONFIG["synapses_enabled"],
        help="Couple selected GPe terminal spikes to STN inhibitory synaptic contacts",
    )
    parser.add_argument("--gpe-inputs-per-cell", type=int, default=DEFAULT_GPE_CONFIG["inputs_per_cell"])
    parser.add_argument("--gpe-contacts-per-input", type=int, default=DEFAULT_GPE_CONFIG["contacts_per_input"])
    parser.add_argument("--gpe-target-soma-frac", type=float, default=DEFAULT_GPE_CONFIG["target_soma_frac"])
    parser.add_argument("--gpe-target-proximal-frac", type=float, default=DEFAULT_GPE_CONFIG["target_proximal_frac"])
    parser.add_argument("--gpe-target-distal-frac", type=float, default=DEFAULT_GPE_CONFIG["target_distal_frac"])
    parser.add_argument("--gpe-syn-proximal-frac", type=float, default=DEFAULT_GPE_CONFIG["syn_proximal_frac"])
    parser.add_argument("--gpe-syn-distal-frac", type=float, default=DEFAULT_GPE_CONFIG["syn_distal_frac"])
    parser.add_argument("--gpe-syn-proximal-max-dist-um", type=float, default=DEFAULT_GPE_CONFIG["syn_proximal_max_dist_um"])
    parser.add_argument("--gpe-syn-distal-min-dist-um", type=float, default=DEFAULT_GPE_CONFIG["syn_distal_min_dist_um"])
    parser.add_argument("--gpe-syn-weight-uS", type=float, default=DEFAULT_GPE_CONFIG["syn_weight_uS"])
    parser.add_argument("--gpe-syn-tau1-ms", type=float, default=DEFAULT_GPE_CONFIG["syn_tau1_ms"])
    parser.add_argument("--gpe-syn-tau2-ms", type=float, default=DEFAULT_GPE_CONFIG["syn_tau2_ms"])
    parser.add_argument("--gpe-syn-e-mV", type=float, default=DEFAULT_GPE_CONFIG["syn_e_mV"])
    parser.add_argument("--gpe-syn-delay-ms", type=float, default=DEFAULT_GPE_CONFIG["syn_delay_ms"])
    parser.add_argument(
        "--gpe-syn-depression",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GPE_CONFIG["syn_depression_enabled"],
        help="Use depression-only DepExp2Syn for GPe GABA-A synapses",
    )
    parser.add_argument("--gpe-syn-depression-u", type=float, default=DEFAULT_GPE_CONFIG["syn_depression_u"])
    parser.add_argument(
        "--gpe-syn-depression-tau-rec-ms",
        type=float,
        default=DEFAULT_GPE_CONFIG["syn_depression_tau_rec_ms"],
    )
    parser.add_argument(
        "--gpe-syn-depression-tau-facil-ms",
        type=float,
        default=DEFAULT_GPE_CONFIG["syn_depression_tau_facil_ms"],
        help="Recorded no-facilitation setting; DepExp2Syn uses U and tau_rec only",
    )
    args = parser.parse_args()
    configure_figure_saving(args.save_figure, args.save_figure_dir)
    DEFAULT_RUN_CONFIG["dbs_rate_edge_window_ms"] = max(0.0, float(args.dbs_rate_edge_window_ms))

    if args.list_morphologies:
        for path in list_available_morphologies():
            print(path.name)
        return

    if args.show_defaults:
        print("run_config:")
        for key, value in DEFAULT_RUN_CONFIG.items():
            print(f"{key}: {value}")
        print("dbs_config:")
        for key, value in DEFAULT_DBS_CONFIG.items():
            print(f"{key}: {value}")
        print("hdp_config:")
        for key, value in DEFAULT_HDP_CONFIG.items():
            print(f"{key}: {value}")
        print("gpe_config:")
        for key, value in DEFAULT_GPE_CONFIG.items():
            print(f"{key}: {value}")
        return

    overrides = parse_parameter_overrides(args.set)
    params = build_parameter_vector(args.param_index, overrides=overrides)
    dbs_config = build_dbs_config_from_args(args)
    hdp_config = build_hdp_config_from_args(args)
    gpe_config = build_gpe_config_from_args(args)
    if hdp_config.get("synapses_enabled", False) and not hdp_config.get("enabled", False):
        raise ValueError("HDP synapses require HDP axons. Use --hdp together with --hdp-synapses.")
    if gpe_config.get("synapses_enabled", False) and not gpe_config.get("enabled", False):
        raise ValueError("GPe synapses require GPe axons. Use --gpe together with --gpe-synapses.")
    idx = best_parameter_index() if args.param_index is None else args.param_index
    print(f"Using 2024 parameter vector index: {idx}")
    if args.show_params:
        for name, value in parameter_dict(params).items():
            print(f"{name}: {value}")

    if args.suite:
        if args.n_cells != 1:
            raise ValueError("--suite currently supports one cell at a time. Use --n-cells 1.")
        if dbs_config["enabled"]:
            raise ValueError("DBS is not wired into --suite yet. Use the standard simulation mode instead.")
        suite = run_protocol_suite(
            model=args.model,
            params=params,
            morphology=args.morphology,
            temperature_c=args.temp,
        )
        print(f"Input impedance: {suite['input_impedance_mohm']:.2f} MOhm")
        print(f"Spontaneous frequency: {suite['spontaneous_frequency_hz']:.2f} Hz")
        print(f"Rest: {suite['rest_mV']:.2f} mV")
        print(f"AHP: {suite['ahp_mV']:.2f} mV")
        print(f"Peak: {suite['peak_mV']:.2f} mV")
        if suite["ap2_width_ms"] is None:
            print("AP2 width: unavailable (install efel to enable)")
        else:
            print(f"AP2 width: {suite['ap2_width_ms']:.2f} ms")
        for key, value in suite["fi_curve"].items():
            print(f"{key}: {value['frequency_hz']:.2f} Hz")
        if args.plot:
            plot_trace(
                {
                    "t_ms": suite["spontaneous_trace"]["t_ms"],
                    "soma_v_mV": suite["spontaneous_trace"]["v_mV"],
                    "initseg_v_mV": np.full_like(suite["spontaneous_trace"]["v_mV"], np.nan),
                    "node0_v_mV": np.full_like(suite["spontaneous_trace"]["v_mV"], np.nan),
                },
                f"2024 STN protocol suite ({args.model})",
            )
    else:
        results = run_population_simulation(
            n_cells=args.n_cells,
            model=args.model,
            base_params=params,
            morphology=args.morphology,
            tstop_ms=args.tstop,
            amp_nA=args.amp,
            delay_ms=args.delay,
            dur_ms=args.dur,
            temperature_c=args.temp,
            param_jitter_frac=args.param_jitter_frac,
            seed=args.seed,
            dbs_config=dbs_config,
            hdp_config=hdp_config,
            gpe_config=gpe_config,
        )
        if hdp_config.get("synapses_enabled", False):
            hdp_results = collect_coupled_hdp_results(results)
        else:
            hdp_results = run_hdp_population_simulation(
                hdp_config=hdp_config,
                dbs_config=dbs_config,
                tstop_ms=args.tstop,
                temperature_c=args.temp,
                seed=args.seed,
            )
        gpe_results = collect_coupled_gpe_results(results)
        summary = summarize_population_results(results)

        print(f"Cells: {int(summary['n_cells'])}")
        print(
            f"Spike count mean/min/max: {summary['mean_spikes']:.2f} / "
            f"{int(summary['min_spikes'])} / {int(summary['max_spikes'])}"
        )
        print(f"Soma Vm range: {summary['soma_min_mV']:.2f} to {summary['soma_max_mV']:.2f} mV")
        if dbs_config["enabled"]:
            phase_label = "A+/B- first" if dbs_config["a_positive_first"] else "A-/B+ first"
            omit_label = ""
            if dbs_config.get("omit_pulse", None) is not None:
                omit_label = f", omit pulse {int(dbs_config['omit_pulse'])}"
            stop_label = ""
            if dbs_config.get("stop_ms", None) is not None:
                stop_label = f", stop {float(dbs_config['stop_ms']):.1f} ms"
            print(
                f"DBS: {dbs_config['amp_uA']:.2f} uA, {dbs_config['freq_hz']:.2f} Hz, "
                f"PW {dbs_config['pw_ms']:.3f} ms, {phase_label}{omit_label}{stop_label}"
            )
        print_compact_cell_table(results, dbs_config)
        if args.print_summary:
            print_paper_style_summary(results, dbs_config, hdp_results, gpe_results)
        if args.print_synapse_details:
            print_stn_local_axon_activation_summary(results, dbs_config)
            print_hdp_synapse_summary(results)
            print_hdp_summary(hdp_results)
            print_gpe_synapse_summary(results)
            print_gpe_summary(gpe_results)
        if args.print_activation_origin:
            print_activation_origin_summary(results, dbs_config)
        if args.paper_plot:
            plot_paper_figure(
                results,
                dbs_config,
                f"2024 STN paper plot ({args.model})",
                nfft_spec=args.spectrogram_nfft,
                analysis_fs_hz=args.analysis_fs_hz,
                spectrogram_window_ms=args.spectrogram_window_ms,
                spectrogram_overlap_frac=args.spectrogram_overlap_frac,
                spectrogram_fmax_hz=args.spectrogram_fmax_hz,
                spectrogram_mode=args.spectrogram_mode,
                spectrogram_rel_baseline=args.spectrogram_rel_baseline,
                spectrogram_baseline_pre_ms=args.spectrogram_baseline_pre_ms,
                spectrogram_cmap=args.spectrogram_cmap,
                spta_baseline_pre_ms=args.spta_baseline_pre_ms,
                mpta_period_fraction=args.mpta_period_fraction,
                mpta_window_scale=args.mpta_window_scale,
                mpta_display_post_periods=args.mpta_display_post_periods,
                hilbert_half_band_hz=args.hilbert_half_band_hz,
                paper_trace_pre_ms=args.paper_trace_pre_ms,
                paper_trace_post_ms=args.paper_trace_post_ms,
            )
        else:
            if args.plot:
                if len(results) == 1:
                    plot_trace(results[0], f"2024 STN model ({args.model})")
                else:
                    plot_population_traces(results, f"2024 STN population ({args.model})")
            if args.plot_spta:
                plot_pta_result(
                    compute_first_pulse_pta(
                        results,
                        dbs_config,
                        baseline_pre_ms=args.spta_baseline_pre_ms,
                    ),
                    f"2024 STN first-pulse PTA ({args.model}, soma)",
                )
            if args.plot_mpta:
                plot_pta_result(
                    compute_train_pulse_pta(
                        results,
                        dbs_config,
                        period_fraction=args.mpta_period_fraction,
                        window_scale=args.mpta_window_scale,
                        display_post_periods=args.mpta_display_post_periods,
                    ),
                    f"2024 STN pulse-train PTA ({args.model}, soma)",
                )
            if args.plot_spectrogram:
                plot_pipeline_spectrogram(
                    results,
                    f"2024 STN spectrogram ({args.model}, mean soma)",
                    dbs_config=dbs_config,
                    nfft_spec=args.spectrogram_nfft,
                    analysis_fs_hz=args.analysis_fs_hz,
                    window_ms=args.spectrogram_window_ms,
                    overlap_frac=args.spectrogram_overlap_frac,
                    f_max_hz=args.spectrogram_fmax_hz,
                    mode=args.spectrogram_mode,
                    baseline_mode=args.spectrogram_rel_baseline,
                    baseline_pre_ms=args.spectrogram_baseline_pre_ms,
                    cmap=args.spectrogram_cmap,
                )
            if args.plot_hilbert_phase:
                plot_hilbert_phase(
                    results,
                    dbs_config,
                    analysis_fs_hz=args.analysis_fs_hz,
                    half_band_hz=args.hilbert_half_band_hz,
                )
            if args.plot_hilbert_amp:
                plot_hilbert_amplitude(
                    results,
                    dbs_config,
                    analysis_fs_hz=args.analysis_fs_hz,
                    half_band_hz=args.hilbert_half_band_hz,
                )
            if args.plot_post_rate_distribution:
                plot_post_rate_distribution(
                    results,
                    dbs_config,
                    f"2024 STN post-DBS firing-rate distribution ({args.model})",
                )
            if args.plot_plv_histogram:
                plot_plv_histogram(
                    results,
                    dbs_config,
                    f"2024 STN DBS-frequency PLV histogram ({args.model})",
                )
        if args.plot_recruitment_dynamics:
            plot_recruitment_dynamics(
                results,
                hdp_results,
                gpe_results,
                dbs_config,
                f"2024 STN pulse recruitment dynamics ({args.model})",
            )
        if args.plot_geometry:
            plot_dbs_geometry(
                results,
                dbs_config,
                "STN geometry",
                hdp_results=hdp_results,
                gpe_results=gpe_results,
                show_dendrites=args.geometry_show_dendrites,
                max_hdp_axons=args.geometry_max_hdp_axons,
                max_gpe_axons=args.geometry_max_gpe_axons,
                show_legend=args.geometry_show_legend,
            )


if __name__ == "__main__":
    main()
