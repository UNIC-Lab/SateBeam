# Data included in the minimal release

`tle/catalog_starlink_20260911.tle` is the fixed Starlink TLE catalogue used by the revision experiments. The code propagates this catalogue with SGP4 and records the catalogue path, record count, and epoch in each run manifest.

`publication/*.csv` contains the exact derived arrays passed to the final table and figure generation stage. These files cover the main comparison, mismatch, candidate/source scaling, grid sensitivity, empirical coherence, per-scenario solver audit, qualitative scenario matrices, and temporal matrices.

`manifests/summary.json` records the archived headline statistics and paired tests for the revised solver.

The minimal release deliberately excludes the large optimizer fit archives and raw per-scenario arrays. They can be regenerated with `run_experiment.py`. Frozen publication data allow immediate inspection of the reported paper artifacts, while the smoke and full commands exercise the scientific pipeline itself.
