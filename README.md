# Functional Hierarchy Analysis — Allen Brain Observatory Neuropixels

This repository reproduces a core analysis using the Allen Brain Observatory Neuropixels (ecephys) dataset accessed via AllenSDK. We test whether functional spiking metrics follow the anatomical visual hierarchy across areas.

Research question (H1): Do functional metrics (drifting gratings modulation and natural scenes image selectivity) exhibit monotonic trends consistent with the anatomical hierarchy across mouse visual areas?

Target areas: LGN, LP, V1, LM, AL, RL, PM, AM (Allen acronyms: LGd, LP, VISp, VISl, VISal, VISrl, VISpm, VISam).

Metrics:
- Drifting gratings modulation index: `mod_idx_dg`
- Natural scenes image selectivity: `image_selectivity_ns`

Data source:
- AllenSDK EcephysProjectCache (warehouse), manifest created at `data/allen_ecephys/manifest.json`.
- Project documentation: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html

Directory layout:
- `src/` — analysis code
  - `analyze_hierarchy.py` — end-to-end pipeline
- `data/allen_ecephys/` — AllenSDK cache directory (manifest stored here)
- `results/` — generated figures and summary CSV/JSON

Setup
1. Create a Python environment (example using conda/micromamba) with packages:
   - allensdk >= 2.16
   - numpy, pandas, scipy, matplotlib, seaborn, scipy.stats
2. Ensure internet access to download the warehouse manifest and tables.

Run analysis
```
python src/analyze_hierarchy.py \
  --base_dir data/allen_ecephys \
  --output_dir results \
  --session_type brain_observatory_1.1
```
This will:
- Instantiate AllenSDK cache using `data/allen_ecephys/manifest.json`
- Load units and precomputed unit analysis metrics for the selected session type
- Map Allen acronyms to target areas
- Apply quality filters: presence_ratio>0.95, isi_violations<0.5, amplitude_cutoff<0.1
- Aggregate medians per area for `mod_idx_dg` and `image_selectivity_ns`
- Compute Spearman correlations vs anatomical rank (LGN=0 < LP=1 < V1=2 < LM=3 < AL=4 < RL=5 < PM=6 < AM=7)
- Save figures and summary outputs in `results/`

Outputs
- `results/mod_idx_dg_by_area.png` — Box+strip: drifting gratings modulation by area
- `results/image_selectivity_ns_by_area.png` — Box+strip: natural scenes image selectivity by area
- `results/median_mod_idx_dg_trend.png` — Trend of area medians vs rank
- `results/median_image_selectivity_ns_trend.png` — Trend of area medians vs rank
- `results/area_summary.csv` — Per-area N and medians
- `results/results.json` — Summary and Spearman statistics

Findings (using brain_observatory_1.1)
- Drifting gratings modulation (mod_idx_dg): strong negative correlation with rank (r ≈ -0.93, p < 0.001). Modulation is highest in LGN and decreases along the hierarchy into cortex.
- Natural scenes image selectivity (image_selectivity_ns): weak, non-significant positive trend (r ≈ 0.17, p ≈ 0.69). Suggests higher selectivity in early cortex vs thalamus but not a clear monotonic increase across all areas under current filters.

Extensions
- Run with `--session_type functional_connectivity` for validation.
- Add bootstrapped confidence intervals and session-level aggregation.
- Explore additional metrics (e.g., f1_f0_dg, lifetime_sparseness_ns) and behavioral modulation.

Citation
If you use this analysis, please cite the Allen Brain Observatory: Neuropixels Visual Coding dataset and AllenSDK documentation.
