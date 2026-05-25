# STN DBS Afferent Model

This repository contains the thesis version of a NEURON-based subthalamic nucleus
(STN) deep brain stimulation (DBS) model. The current model simulates detailed
STN cells, local STN axons, hyperdirect pathway (HDP) afferent axons, GPe afferent
axons, and short-term synaptic depression.

The model is intended for mechanistic exploration of DBS-evoked STN responses,
including pulse-locked firing, first-pulse and pulse-train peri-triggered averages,
axon recruitment, and depression-weighted synaptic drive.

## Main Files

| File | Purpose |
| --- | --- |
| `STN.py` | Main simulation script. This is the renamed thesis version of the STN afferent model. |
| `sucs.py` | Runs multiple `STN.py` simulations in succession for overnight/batch runs. |
| `TEED.py` | Helper script for TEED-balanced DBS parameter calculations. |
| `GPe.py` | Standalone detailed GPe model wrapper. This is still under construction and is not yet part of the main thesis simulations. |

## Folders

| Folder | Contents |
| --- | --- |
| `external/STN-Neuron-main/` | Local copy of the STN morphology, parameter pool, and base STN mechanisms used by `STN.py`. |
| `external/STN-Neuron-main/Detailed Morphology/` | STN morphology `.swc` files. |
| `external/STN-Neuron-main/sth/` | STN NEURON mechanisms, hoc files, and `sth-data` channel distribution files. |
| `st_depression/` | Custom depression-only `DepExp2Syn.mod` synapse mechanism. |
| `STN+GPe/` | Combined STN/GPe mechanism sources for future GPe integration. |
| `external/koelman-stn-gpe-model-frontiers/` | Minimal Koelman/Gunay GPe configuration and morphology assets used by `GPe.py`. |

## Requirements

This code was developed with Python and NEURON. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

You also need a working NEURON installation with `nrnivmodl` available on the command line.

## First-Time Setup

The model compiles NEURON mechanism folders automatically when needed. If you want to
compile them manually:

```bash
cd external/STN-Neuron-main/sth
nrnivmodl

cd ../../../st_depression
nrnivmodl
```

The default STN asset path is local to this repository:

```text
external/STN-Neuron-main
```

To use a different STN asset folder, set:

```bash
STN_NEURON_ROOT=/path/to/STN-Neuron-main python STN.py
```

## Running One Simulation

From the repository root:

```bash
python STN.py
```

The script prints a compact cell-by-cell table and a paper-style summary. Depending
on the plot toggles in `STN.py`, it can also generate:

- population trace
- first-pulse PTA
- pulse-train PTA
- spectrogram
- synaptic drive / short-term depression plot
- firing-rate or entrainment distribution
- paper-style multi-panel figure

Figures are saved to `/home/dtorbin/Downloads/articles` when `save_figure` is enabled.

## Running Batch Simulations

Edit the `RUN_PLAN` list in `sucs.py`, then run:

```bash
python sucs.py
```

Each run gets its own output folder under `/home/dtorbin/Downloads/articles`.

## Current Scope

Included in the main model:

- detailed STN cells
- local STN axons
- HDP afferent axons and AMPA synapses
- GPe afferent axons and GABA-A synapses
- short-term synaptic depression
- DBS parameter sweeps and paper-style plotting

Not yet finalized:

- full recurrent STN-GPe network coupling
- fully integrated detailed GPe population model
- GABA-B synapses

`GPe.py` is kept in the repository as a work-in-progress candidate for future full
GPe integration, but the thesis-ready runs should currently use `STN.py` and `sucs.py`.

## Provenance Notes

The files under `external/STN-Neuron-main/` include upstream README files from the
STN model source. They are kept for provenance and are not the main README for this
repository. The main repository instructions are the present file.
