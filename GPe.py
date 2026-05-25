from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_DIR = Path(__file__).resolve().parent
ARTICLES_DIR = Path("/home/dtorbin/Downloads/articles")
COMBINED_MECH_DIR = REPO_DIR / "STN+GPe"
DEFAULT_KOELMAN_ROOT = REPO_DIR / "external" / "koelman-stn-gpe-model-frontiers"
KOELMAN_ROOT = Path(os.environ.get("KOELMAN_STN_GPE_ROOT", DEFAULT_KOELMAN_ROOT)).expanduser().resolve()
GPE_ROOT = KOELMAN_ROOT / "bgcellmodels" / "models" / "GPe" / "Gunay2008"

# NEURON auto-loads ./x86_64 on import. Running from sth-model can therefore
# load the STN-only library before the combined STN+GPe library. Step away first.
LAUNCH_CWD = Path.cwd()
if (LAUNCH_CWD / "x86_64").exists():
    os.chdir("/tmp")

from neuron import h, load_mechanisms


LITERATURE_TARGETS = {
    "gunay_2008_slice": {
        "paper": "Gunay, Edgerton & Jaeger 2008",
        "species": "rat",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5771640/",
        "notes": (
            "Rat GP/GPe slice recordings and model database. Reported spontaneous "
            "firing ranged from 0 to 25.51 Hz, mean 5.45 Hz, SD 5.89 Hz. The model "
            "database was tuned to spike shape, f-I behavior, sag, rebound, and "
            "conductance-density variability."
        ),
        "spontaneous_rate_hz": [0.0, 25.51],
    },
    "tachibana_2011_in_vivo": {
        "paper": "Tachibana et al. 2011",
        "species": "6-OHDA rat / control rat",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3113166/",
        "notes": (
            "In vivo GPe firing in control and 6-OHDA rats was much faster than "
            "slice autonomous firing: approximately 29.3 +/- 12.2 Hz in controls "
            "and 26.2 +/- 10.2 Hz in 6-OHDA rats, with stronger bursting/pauses "
            "in parkinsonian animals. This is a network-state target, not a pure "
            "cell-intrinsic target."
        ),
        "in_vivo_rate_hz_mean_sd": [[29.3, 12.2], [26.2, 10.2]],
    },
    "koelman_lowery_2019": {
        "paper": "Koelman & Lowery 2019",
        "species": "computational STN-GPe network, rodent-grounded GPe model",
        "source": "https://www.frontiersin.org/articles/10.3389/fncom.2019.00077/full",
        "notes": (
            "Uses detailed STN and GPe conductance-based cells for beta resonance "
            "and intrinsic STN-GPe oscillation analysis. The GPe implementation "
            "is derived from the Gunay/Hendrickson lineage and is the main model "
            "family wrapped here."
        ),
    },
    "kumaravelu_2016": {
        "paper": "Kumaravelu et al. 2016",
        "species": "6-OHDA rat computational BG-thalamus model",
        "source": "https://doi.org/10.1007/s10827-016-0593-9",
        "notes": (
            "Useful later for full basal ganglia network validation under DBS, "
            "but not a one-to-one target for the standalone detailed GPe cell."
        ),
    },
    "kang_lowery_2014": {
        "paper": "Kang & Lowery 2014",
        "species": "computational cortico-STN-GPe DBS model",
        "source": "https://www.frontiersin.org/articles/10.3389/fncom.2014.00032/full",
        "notes": (
            "Useful later for DBS-mediated orthodromic/antidromic afferent effects "
            "and STN-GPe network behavior. It is not a strict standalone GPe-cell "
            "calibration target."
        ),
    },
}


DEFAULT_OUTPUT_DIR = ARTICLES_DIR / "gpe_sanity"


def load_nonstrict_json(path: Path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//.*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def convert_to_neuron_units(value: float, units: str | None) -> float:
    """Convert GENESIS/SI-style config units into NEURON units."""

    if units is None:
        return float(value)
    if units == "V":
        return float(value) * 1e3
    if units == "Ohm*m":
        return float(value) * 1e2
    if units == "F/m^2":
        return float(value) * 1e2
    if units == "S/m^2":
        return float(value) * 1e-4
    if units == "seconds":
        return float(value) * 1e3
    return float(value)


def eval_genesis_expression(expr, genesis_params: dict[str, float]) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    resolved = str(expr).format(**genesis_params)
    return float(eval(resolved, {"__builtins__": {}}, {"math": math}))


def gpe_mech_name(name: str) -> str:
    return f"gpe_{name}"


def ensure_source_files() -> None:
    missing = []
    for path in [
        GPE_ROOT / "config" / "mechanisms.json",
        GPE_ROOT / "config" / "params_gunay2008_GENESIS.json",
        GPE_ROOT / "config" / "map_params_gunay2008_v2.json",
        GPE_ROOT / "config" / "locations.json",
        GPE_ROOT / "morphology" / "bg0121b_axonless_GENESIS_import.swc",
    ]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "GPe model assets are incomplete. Missing:\n" + "\n".join(missing)
        )


class GunayGPeCell:
    """Rodent GPe cell wrapper based on the Koelman/Gunay/Hendrickson model.

    This class wraps the Gunay et al. 2008 rat GP/GPe conductance model as
    ported in the Koelman/Lowery codebase. The mechanisms are loaded from the
    combined STN+GPe mechanism library, where GPe mechanism suffixes are
    prefixed with ``gpe_`` to avoid collisions with the STN mechanisms.

    Important modeling note:
    The morphology is detailed enough to preserve soma/dendrite/axon-stub
    channel placement, but the original GENESIS-compatible port normalizes
    compartment dimensions after parameter application. Treat this as a trusted
    conductance-based GPe cell, not as a final visually realistic GPe cable.
    """

    mechanisms_ready = False

    def __init__(self, *, clear_existing_sections: bool = False, normalize_dimensions: bool = True):
        self.ensure_neuron_ready()
        if clear_existing_sections:
            clear_sections()
        self.normalize_dimensions = bool(normalize_dimensions)
        self._load_morphology()
        self.section_lists = {
            "all": list(self.all),
            "somatic": list(self.soma),
            "basal": list(self.dend),
            "axonal": list(self.axon),
            "apical": [],
        }
        self.raw_geometry_summary = self.geometry_summary()
        self.location_specs = {
            spec["loc_name"]: spec
            for spec in load_nonstrict_json(GPE_ROOT / "config" / "locations.json")
        }
        self.location_hit_counts: dict[str, int] = {}
        self._insert_mechanisms()
        self._apply_parameters()
        if self.normalize_dimensions:
            self._fix_compartment_dimensions()
        self.normalized_geometry_summary = self.geometry_summary()

    @classmethod
    def ensure_neuron_ready(cls) -> None:
        if cls.mechanisms_ready:
            return
        ensure_source_files()
        lib_path = COMBINED_MECH_DIR / "x86_64" / "libnrnmech.so"
        if not lib_path.exists():
            subprocess.run(["nrnivmodl"], cwd=COMBINED_MECH_DIR, check=True)
        try:
            load_mechanisms(str(COMBINED_MECH_DIR))
        except RuntimeError as exc:
            if "already exists" not in str(exc):
                raise
            # If the combined library was already loaded this is fine. If only
            # the STN library was loaded, the prefixed GPe mechanisms will be
            # absent and the next check will fail loudly.
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        if not hasattr(h, "gpe_NaF") or not hasattr(h, "gpe_Kv3"):
            raise RuntimeError(
                "GPe mechanisms were not available. Run GPe.py in a fresh Python "
                "process before importing 2024_full.py, or load the combined "
                "STN+GPe mechanism library first."
            )
        cls.mechanisms_ready = True

    def _load_morphology(self) -> None:
        reader = h.Import3d_SWC_read()
        reader.input(str(GPE_ROOT / "morphology" / "bg0121b_axonless_GENESIS_import.swc"))
        importer = h.Import3d_GUI(reader, 0)
        importer.instantiate(self)

    def geometry_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for name, secs in {
            "soma": list(getattr(self, "soma", [])),
            "dend": list(getattr(self, "dend", [])),
            "axon": list(getattr(self, "axon", [])),
            "all": list(getattr(self, "all", [])),
        }.items():
            lengths = [float(sec.L) for sec in secs]
            diams = [float(sec.diam) for sec in secs]
            summary[name] = {
                "sections": int(len(secs)),
                "segments": int(sum(sec.nseg for sec in secs)),
                "length_sum_um": float(np.sum(lengths)) if lengths else 0.0,
                "diam_mean_um": float(np.mean(diams)) if diams else 0.0,
                "diam_min_um": float(np.min(diams)) if diams else 0.0,
                "diam_max_um": float(np.max(diams)) if diams else 0.0,
            }
        return summary

    def _insert_mechanisms(self) -> None:
        mechanism_specs = load_nonstrict_json(GPE_ROOT / "config" / "mechanisms.json")
        for section_list_name, mechanisms in mechanism_specs.items():
            for sec in self.section_lists.get(section_list_name, []):
                for mechanism in mechanisms:
                    sec.insert("pas" if mechanism == "pas" else gpe_mech_name(mechanism))

    def _segments_for_location(self, location_name: str) -> list:
        spec = self.location_specs[location_name]
        lower_d = float(spec["lower_distance"])
        upper_d = float(spec["upper_distance"])
        lower_diam = float(spec["lower_diameter"])
        upper_diam = float(spec["upper_diameter"])

        soma_sec = self.soma[0]
        h.distance(0, soma_sec(0.5))
        hits = []
        for sec in self.section_lists.get(spec["sectionlist"], []):
            for seg in sec:
                distance = float(h.distance(float(seg.x), sec=sec))
                diam = float(seg.diam)
                if lower_d <= distance < upper_d and lower_diam <= diam < upper_diam:
                    hits.append(seg)
        self.location_hit_counts[location_name] = len(hits)
        return hits

    def _targets_for_spec(self, spec: dict):
        if "location" in spec:
            return self._segments_for_location(spec["location"])
        return self.section_lists.get(spec.get("sectionlist", ""), [])

    @staticmethod
    def _set_attr(target, attr_name: str, value: float) -> bool:
        try:
            setattr(target, attr_name, value)
            return True
        except Exception:
            return False

    @staticmethod
    def _attr_name_for_spec(spec: dict) -> str:
        if "mech" in spec:
            return f"{spec['mech_param']}_{gpe_mech_name(spec['mech'])}"
        return str(spec["param_name"])

    def _apply_parameters(self) -> None:
        genesis_params = load_nonstrict_json(GPE_ROOT / "config" / "params_gunay2008_GENESIS.json")
        param_specs = load_nonstrict_json(GPE_ROOT / "config" / "map_params_gunay2008_v2.json")

        self.applied_parameters: list[dict] = []
        self.skipped_parameters: list[dict] = []
        for spec in param_specs:
            attr_name = self._attr_name_for_spec(spec)
            value = convert_to_neuron_units(
                eval_genesis_expression(spec["value"], genesis_params),
                spec.get("units"),
            )
            targets = self._targets_for_spec(spec)
            applied = 0
            if spec["type"] == "section":
                for sec in targets:
                    applied += int(self._set_attr(sec, attr_name, value))
            elif spec["type"] in ("range", "segment"):
                for target in targets:
                    if hasattr(target, "x") and hasattr(target, "sec"):
                        applied += int(self._set_attr(target, attr_name, value))
                    else:
                        for seg in target:
                            applied += int(self._set_attr(seg, attr_name, value))
            else:
                self.skipped_parameters.append({"spec": spec, "reason": "unknown type"})
                continue

            record = {
                "name": attr_name,
                "type": spec["type"],
                "sectionlist": spec.get("sectionlist"),
                "location": spec.get("location"),
                "value": value,
                "applied_count": applied,
            }
            if applied:
                self.applied_parameters.append(record)
            else:
                self.skipped_parameters.append({**record, "reason": "no targets or attr failed"})

    def _fix_compartment_dimensions(self) -> None:
        for sec in self.section_lists["all"]:
            sec.L = 1.0
            sec.diam = 1.0


def clear_sections() -> None:
    for sec in list(h.allsec()):
        h.delete_section(sec=sec)


def threshold_crossings(t_ms: np.ndarray, v_mV: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    crossings = np.flatnonzero((v_mV[:-1] < threshold) & (v_mV[1:] >= threshold))
    if len(crossings) == 0:
        return np.array([], dtype=float)
    times = []
    for idx in crossings:
        dv = float(v_mV[idx + 1] - v_mV[idx])
        frac = 0.0 if abs(dv) < 1e-12 else (threshold - float(v_mV[idx])) / dv
        times.append(float(t_ms[idx]) + frac * float(t_ms[idx + 1] - t_ms[idx]))
    return np.array(times, dtype=float)


def half_width_ms(t_ms: np.ndarray, v_mV: np.ndarray, spike_times_ms: np.ndarray) -> float:
    widths = []
    for spike_time in spike_times_ms[: min(20, len(spike_times_ms))]:
        center = int(np.argmin(np.abs(t_ms - spike_time)))
        lo = max(0, center - int(round(5.0 / np.median(np.diff(t_ms)))))
        hi = min(len(t_ms) - 1, center + int(round(8.0 / np.median(np.diff(t_ms)))))
        local_v = v_mV[lo : hi + 1]
        local_t = t_ms[lo : hi + 1]
        if len(local_v) < 3:
            continue
        peak = float(np.max(local_v))
        base = float(np.min(local_v[: max(1, center - lo)]))
        half = base + 0.5 * (peak - base)
        above = np.flatnonzero(local_v >= half)
        if len(above) >= 2:
            widths.append(float(local_t[above[-1]] - local_t[above[0]]))
    return float(np.mean(widths)) if widths else np.nan


def run_current_clamp(
    *,
    tstop_ms: float = 1000.0,
    amp_nA: float = 0.0,
    delay_ms: float = 250.0,
    dur_ms: float = 500.0,
    temperature_c: float = 37.0,
    dt_ms: float = 0.025,
    v_init_mV: float = -60.0,
    normalize_dimensions: bool = True,
) -> dict:
    clear_sections()
    cell = GunayGPeCell(clear_existing_sections=False, normalize_dimensions=normalize_dimensions)
    h.dt = float(dt_ms)
    h.celsius = float(temperature_c)
    h.CVode().active(0)

    stim = h.IClamp(cell.soma[0](0.5))
    stim.delay = float(delay_ms)
    stim.dur = float(dur_ms)
    stim.amp = float(amp_nA)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.finitialize(float(v_init_mV))
    while float(h.t) < float(tstop_ms):
        h.fadvance()

    t_ms = np.asarray(t_vec, dtype=float)
    v_mV = np.asarray(v_vec, dtype=float)
    spike_times = threshold_crossings(t_ms, v_mV, threshold=0.0)
    analysis_mask = t_ms >= min(100.0, 0.2 * float(tstop_ms))
    stim_mask = (t_ms >= float(delay_ms)) & (t_ms <= float(delay_ms) + float(dur_ms))
    post_mask = t_ms > float(delay_ms) + float(dur_ms)
    baseline_spikes = spike_times[spike_times >= 100.0]
    stim_spikes = spike_times[(spike_times >= float(delay_ms)) & (spike_times <= float(delay_ms) + float(dur_ms))]
    post_spikes = spike_times[spike_times > float(delay_ms) + float(dur_ms)]
    isi = np.diff(baseline_spikes)
    metrics = {
        "tstop_ms": float(tstop_ms),
        "temperature_c": float(temperature_c),
        "dt_ms": float(dt_ms),
        "amp_nA": float(amp_nA),
        "delay_ms": float(delay_ms),
        "dur_ms": float(dur_ms),
        "spike_count": int(len(spike_times)),
        "spontaneous_rate_after_100ms_hz": float(len(baseline_spikes) / max(1e-9, (float(tstop_ms) - 100.0) * 1e-3)),
        "stim_rate_hz": float(len(stim_spikes) / max(1e-9, float(dur_ms) * 1e-3)),
        "post_rate_hz": float(len(post_spikes) / max(1e-9, (float(tstop_ms) - float(delay_ms) - float(dur_ms)) * 1e-3)),
        "isi_cv_after_100ms": float(np.std(isi) / np.mean(isi)) if len(isi) >= 2 and np.mean(isi) > 0 else np.nan,
        "v_mean_after_100ms_mV": float(np.mean(v_mV[analysis_mask])) if np.any(analysis_mask) else np.nan,
        "v_min_mV": float(np.min(v_mV)),
        "v_max_mV": float(np.max(v_mV)),
        "stim_v_mean_mV": float(np.mean(v_mV[stim_mask])) if np.any(stim_mask) else np.nan,
        "post_v_mean_mV": float(np.mean(v_mV[post_mask])) if np.any(post_mask) else np.nan,
        "half_width_ms": half_width_ms(t_ms, v_mV, spike_times),
        "applied_parameter_count": int(len(cell.applied_parameters)),
        "skipped_parameter_count": int(len(cell.skipped_parameters)),
        "raw_geometry_summary": cell.raw_geometry_summary,
        "normalized_geometry_summary": cell.normalized_geometry_summary,
        "location_hit_counts": cell.location_hit_counts,
    }
    return {
        "cell": cell,
        "t_ms": t_ms,
        "v_mV": v_mV,
        "spike_times_ms": spike_times,
        "metrics": metrics,
        "stim": {
            "delay_ms": float(delay_ms),
            "dur_ms": float(dur_ms),
            "amp_nA": float(amp_nA),
        },
    }


def plot_trace(result: dict, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    t_ms = result["t_ms"]
    v_mV = result["v_mV"]
    stim = result["stim"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.3))
    ax.plot(t_ms, v_mV, color="black", lw=1.0)
    if stim["amp_nA"] != 0.0:
        ax.axvspan(
            stim["delay_ms"],
            stim["delay_ms"] + stim["dur_ms"],
            color="tab:blue" if stim["amp_nA"] > 0 else "tab:purple",
            alpha=0.12,
            label=f"IClamp {stim['amp_nA']:.3g} nA",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("GPe soma Vm (mV)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fi_curve(records: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    amps = np.array([record["metrics"]["amp_nA"] for record in records], dtype=float)
    stim_rates = np.array([record["metrics"]["stim_rate_hz"] for record in records], dtype=float)
    spont_rates = np.array([record["metrics"]["spontaneous_rate_after_100ms_hz"] for record in records], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax.plot(amps, stim_rates, marker="o", color="black", label="during current step")
    ax.plot(amps, spont_rates, marker="s", color="tab:gray", label="whole trace after 100 ms")
    ax.axhspan(0.0, 25.51, color="tab:green", alpha=0.08, label="Gunay slice spontaneous range")
    ax.axhline(5.45, color="tab:green", ls="--", lw=1.0, label="Gunay slice mean")
    ax.set_xlabel("IClamp amplitude (nA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("GPe sanity f-I curve")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_literature_comparison(metrics: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rate = metrics["spontaneous_rate_after_100ms_hz"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    ax.axhspan(0.0, 25.51, color="tab:green", alpha=0.18, label="Gunay 2008 slice range")
    ax.axhline(5.45, color="tab:green", ls="--", lw=1.3, label="Gunay 2008 slice mean")
    ax.axhspan(29.3 - 12.2, 29.3 + 12.2, color="tab:orange", alpha=0.15, label="Control in vivo rat mean +/- SD")
    ax.axhspan(26.2 - 10.2, 26.2 + 10.2, color="tab:red", alpha=0.12, label="6-OHDA in vivo rat mean +/- SD")
    ax.scatter([0], [rate], color="black", s=90, zorder=5, label="This standalone model")
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["GPe.py"])
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Standalone GPe firing-rate target check")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_sanity_suite(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    temperature_c: float = 37.0,
    tstop_ms: float = 1000.0,
    dt_ms: float = 0.025,
    normalize_dimensions: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    spontaneous = run_current_clamp(
        tstop_ms=tstop_ms,
        amp_nA=0.0,
        delay_ms=250.0,
        dur_ms=500.0,
        temperature_c=temperature_c,
        dt_ms=dt_ms,
        normalize_dimensions=normalize_dimensions,
    )
    plot_trace(
        spontaneous,
        output_dir / "gpe_spontaneous_trace.png",
        "GPe.py standalone sanity: spontaneous activity",
    )
    plot_literature_comparison(spontaneous["metrics"], output_dir / "gpe_literature_rate_comparison.png")

    fi_records = []
    # The GENESIS-compatible port normalizes compartment dimensions. Tiny
    # pA-scale NEURON currents are therefore the right sanity range; larger nA
    # steps are useful only as stress tests and can distort the f-I readout.
    for amp_nA in [-0.002, -0.001, 0.0, 0.001, 0.002, 0.005]:
        record = run_current_clamp(
            tstop_ms=tstop_ms,
            amp_nA=amp_nA,
            delay_ms=250.0,
            dur_ms=500.0,
            temperature_c=temperature_c,
            dt_ms=dt_ms,
            normalize_dimensions=normalize_dimensions,
        )
        fi_records.append(record)
        amp_pA = int(round(1000.0 * amp_nA))
        plot_trace(
            record,
            output_dir / f"gpe_iclamp_{amp_pA:+d}pA_trace.png",
            f"GPe.py standalone sanity: IClamp {amp_pA:+d} pA",
        )
    plot_fi_curve(fi_records, output_dir / "gpe_fi_curve.png")

    metrics = {
        "model": "Koelman/Gunay/Hendrickson rodent GPe wrapper",
        "temperature_c": float(temperature_c),
        "normalize_dimensions": bool(normalize_dimensions),
        "literature_targets": LITERATURE_TARGETS,
        "spontaneous": spontaneous["metrics"],
        "fi_curve": [
            {
                "amp_nA": record["metrics"]["amp_nA"],
                "stim_rate_hz": record["metrics"]["stim_rate_hz"],
                "spontaneous_rate_after_100ms_hz": record["metrics"]["spontaneous_rate_after_100ms_hz"],
                "v_min_mV": record["metrics"]["v_min_mV"],
                "v_max_mV": record["metrics"]["v_max_mV"],
                "isi_cv_after_100ms": record["metrics"]["isi_cv_after_100ms"],
                "half_width_ms": record["metrics"]["half_width_ms"],
            }
            for record in fi_records
        ],
        "interpretation": interpret_sanity(spontaneous["metrics"]),
        "outputs": {
            "spontaneous_trace": str(output_dir / "gpe_spontaneous_trace.png"),
            "literature_rate_comparison": str(output_dir / "gpe_literature_rate_comparison.png"),
            "fi_curve": str(output_dir / "gpe_fi_curve.png"),
            "metrics_json": str(output_dir / "gpe_sanity_metrics.json"),
            "summary_md": str(output_dir / "gpe_sanity_summary.md"),
        },
    }
    (output_dir / "gpe_sanity_metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2), encoding="utf-8")
    write_summary_md(metrics, output_dir / "gpe_sanity_summary.md")
    return metrics


def interpret_sanity(metrics: dict) -> list[str]:
    notes = []
    rate = float(metrics.get("spontaneous_rate_after_100ms_hz", np.nan))
    if 0.0 <= rate <= 25.51:
        notes.append("Standalone spontaneous rate is within the Gunay et al. 2008 rat slice range.")
    elif np.isfinite(rate):
        notes.append(
            "Standalone spontaneous rate is outside the Gunay et al. 2008 slice range; "
            "check temperature, initialized voltage, and parameter mapping before using in a network."
        )
    if np.isfinite(rate) and 15.0 <= rate <= 45.0:
        notes.append("Rate is also near in vivo rat GPe ranges, but this should be interpreted as a network-state comparison.")
    elif np.isfinite(rate) and rate > 45.0:
        notes.append("Rate is much higher than typical in vivo rat GPe firing, so this configuration is not sane for network use.")
    else:
        notes.append("Rate is lower than typical in vivo rat GPe firing, which is expected for a slice-tuned autonomous model.")
    skipped = int(metrics.get("skipped_parameter_count", 0))
    if skipped:
        notes.append(
            f"{skipped} parameter specs did not apply, mostly from empty diameter/distance bins in this morphology; "
            "this occurred in the internal wrapper too and is acceptable if active mechanisms and firing remain sane."
        )
    return notes


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_summary_md(metrics: dict, path: Path) -> None:
    spont = metrics["spontaneous"]
    lines = [
        "# GPe.py Standalone Sanity Summary",
        "",
        "## Model",
        "",
        "- Koelman/Gunay/Hendrickson-style rodent GPe conductance model.",
        "- GPe mechanisms are prefixed and loaded from the combined STN+GPe mechanism library.",
        "- This is suitable as the first full GPe-cell candidate for STN-GPe coupling, but should still be validated in-network.",
        "",
        "## Main Standalone Result",
        "",
        f"- Temperature: `{metrics['temperature_c']}` C",
        f"- Spontaneous rate after 100 ms: `{spont['spontaneous_rate_after_100ms_hz']:.2f}` Hz",
        f"- Spike count: `{spont['spike_count']}`",
        f"- Vm range: `{spont['v_min_mV']:.2f}` to `{spont['v_max_mV']:.2f}` mV",
        f"- ISI CV after 100 ms: `{spont['isi_cv_after_100ms']:.3f}`",
        f"- AP half-width estimate: `{spont['half_width_ms']:.3f}` ms",
        f"- Applied/skipped parameter specs: `{spont['applied_parameter_count']}` / `{spont['skipped_parameter_count']}`",
        "",
        "## Interpretation",
        "",
    ]
    for note in metrics["interpretation"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Literature Targets", ""])
    for target in LITERATURE_TARGETS.values():
        lines.append(f"- **{target['paper']}** ({target['species']}): {target['notes']} Source: {target['source']}")
    lines.extend(["", "## Output Files", ""])
    for key, value in metrics["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone rodent GPe sanity runner")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp", type=float, default=37.0)
    parser.add_argument("--tstop", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument(
        "--no-normalize-dimensions",
        action="store_true",
        help="Keep imported SWC dimensions instead of the GENESIS-compatible normalized dimensions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON metrics instead of the compact console summary.",
    )
    args = parser.parse_args()

    metrics = run_sanity_suite(
        output_dir=args.output_dir,
        temperature_c=args.temp,
        tstop_ms=args.tstop,
        dt_ms=args.dt,
        normalize_dimensions=not args.no_normalize_dimensions,
    )
    if args.json:
        print(json.dumps(to_jsonable(metrics), indent=2))
    else:
        spont = metrics["spontaneous"]
        print("GPe.py sanity complete")
        print(f"Output dir: {args.output_dir}")
        print(f"Spontaneous rate: {spont['spontaneous_rate_after_100ms_hz']:.2f} Hz")
        print(f"Vm range: {spont['v_min_mV']:.2f} to {spont['v_max_mV']:.2f} mV")
        print(f"ISI CV: {spont['isi_cv_after_100ms']:.3f}")
        print("Interpretation:")
        for note in metrics["interpretation"]:
            print(f"- {note}")


if __name__ == "__main__":
    main()
