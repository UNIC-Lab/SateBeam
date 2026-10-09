# SateBeam

Reference code and frozen publication data for **SateBeam: Beam-Aware Radio Map Estimation With Physics-Consistent Parametric Modeling for Unknown Multiple Satellites**.

SateBeam uses known satellite ephemerides as two complementary forms of sensing structure. Propagated positions determine link geometry and spatial beam footprints, while radial velocities determine candidate-specific Doppler trajectories. The solver combines Doppler screening, GPU-accelerated grid proposals, continuous beam refinement, Gamma-likelihood BIC model-order selection, backward deletion, and support exchange.

The experiments use real TLE-derived satellite positions and velocities with simulated received-power measurements. They are not field-measurement experiments.

## What this repository contains

```text
SateBeam/
├── satbeam_study/              # final simulator, estimator, baselines, metrics
├── satbeam/                    # low-level TLE and geometry helpers
├── data/
│   ├── tle/                    # frozen 10,713-record Starlink TLE snapshot
│   ├── publication/            # frozen data passed to paper tables/figures
│   ├── manifests/              # archived numerical summary
│   └── checksums.sha256        # integrity hashes for frozen inputs
├── scripts/
│   ├── reproduce_paper.py      # rebuild key tables and figures
│   └── verify_release.py       # check hashes and headline claims
├── tests/
│   └── test_release.py         # focused physics and data checks
├── run_experiment.py           # end-to-end experiment entry point
├── environment.yml
└── requirements.txt
```

## Environment

The reference environment is Python 3.11.15. Create a clean environment with either command below.

```bash
conda env create -f environment.yml
conda activate satebeam
```

or

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux and macOS before installing the requirements.

The CPU path is sufficient for verification and smoke testing. The optional GPU path uses PyTorch when a compatible CUDA installation is already available. PyTorch is intentionally not pinned in `requirements.txt` because its wheel must match the local CUDA runtime.

## Fast verification

Verify the frozen inputs and manuscript-facing numerical claims:

```bash
python scripts/verify_release.py
```

Rebuild the key tables and figures from the frozen publication data:

```bash
python scripts/reproduce_paper.py --output reproduced
```

Run focused tests:

```bash
python tests/test_release.py
```

## End-to-end algorithm smoke test

This command propagates the bundled TLE catalogue, simulates one small measurement scenario, runs SateBeam and selected comparators, saves raw observations and fitted outputs, and writes metrics and provenance.

```bash
python run_experiment.py --smoke --cpu-only --output outputs/smoke
```

The smoke test checks execution and persistence. It is not a substitute for the full statistical experiment.

## Main nominal experiment

The following command uses the paper's nominal physical settings and runs 30 paired scenarios. The GPU path is recommended for SateBeam and full-grid baselines.

```bash
python run_experiment.py \
  --suite validation \
  --conditions four_by_60s \
  --trials 30 \
  --methods SateBeam OMP-Grid Lasso-Grid Orbit-NNLS Peak IDW Kriging SoftImpute DeepRM-TD Oracle-Support \
  --output outputs/nominal
```

Add `--cpu-only` for a CPU run. The command preserves raw scenario inputs, per-method fits, metrics, the complete configuration, environment information, and a TLE manifest under the selected output directory.

Additional reviewer-requested suites are available through `--suite mismatch`, `--suite scaling`, `--suite selection`, `--suite grid`, `--suite identifiability`, and `--suite physical`.

## Frozen publication data

`data/publication/` contains the exact CSV views used by the final publication plotting stage. `scripts/reproduce_paper.py` rebuilds publication-style summaries without rerunning orbital propagation or nonlinear optimization. This makes table and figure inspection fast and deterministic.

The minimal repository does not bundle the much larger per-scenario raw and fit archives. The included experiment entry point regenerates those archives from the frozen TLE snapshot and declared seeds. If the complete archived run is distributed separately through a GitHub Release or research-data repository, its checksum can be added to `data/checksums.sha256` without changing the code.

## Data integrity

The bundled TLE snapshot has SHA-256:

```text
E9A02010F992C24CA767195441F36B55FEEA898080E0F21C2857F3AAB609CF11
```

Run `python scripts/verify_release.py` before using the frozen data.

## Citation

Citation metadata are provided in `CITATION.cff`.
