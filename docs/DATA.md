# Archived data

Everything described here is downloaded by `download_data.py` from the Zenodo deposition accompanying the paper: [https://doi.org/10.1016/j.matdes.2026.116984](https://doi.org/10.1016/j.matdes.2026.116984).

```bash
python download_data.py          # fetch and extract
python download_data.py --list   # archive, size, extraction targets
python download_data.py --check  # verify what is present
```

The deposition is [https://doi.org/10.5281/zenodo.19009893](https://doi.org/10.5281/zenodo.19009893) (CC BY 4.0). The record id and the MD5 checksum live in `zenodo_manifest.json`, and downloads are checksum-verified before extraction. Everything ships as a single archive, `baygds-data.tar.gz`.

---

## Inputs

The microstructures and the oracle responses are the inputs to the pipeline. Every other archived file is derived from them.

### `data/structures/structures.npz`

| Array          | Shape           | Dtype | Meaning                                                     |
| -------------- | --------------- | ----- | ----------------------------------------------------------- |
| `structures` | (50000, 96, 96) | uint8 | Binary microstructure images.`1` is solid, `0` is void. |

### `data/oracle/oracle_0deg.npz` and `data/oracle/oracle_45deg.npz`

Precomputed high-fidelity responses. `oracle.py` reads from these in place of a live simulation.

| Array                | Shape              | Dtype   | Meaning                                                                    |
| -------------------- | ------------------ | ------- | -------------------------------------------------------------------------- |
| `stresses`         | (50000, 105, 2, 2) | float64 | First Piola–Kirchhoff stress in MPa, per structure and deformation state. |
| `deformation_grid` | (105, 2, 2)        | float64 | The applied deformation gradients.                                         |
| `deformation_tags` | (105,)             | object  | Loading path of each state.                                                |

The tags take five values: `tension_x`, `tension_y`, `equibiaxial`, `off_x`, `off_y`.

The two files differ in the orientation of the applied loading: `oracle_0deg` drives the surrogate training, while `oracle_45deg` supplies the held-out targets for inverse design. Using distinct loading orientations for training and for target definition is what evaluates the inverse design for generalization to unseen loading directions.

### `data/pca/pc_scores.npz`

| Array              | Shape      | Dtype   | Meaning                                                                             |
| ------------------ | ---------- | ------- | ----------------------------------------------------------------------------------- |
| `PCA_components` | (50000, 8) | float64 | PC scores of the phase and interface two-point correlations, one row per structure. |

Descriptors derived from the structures by `experiments/baygds/run_feature_engineering.py`, archived so that the later stages can be run without recomputing them. Eight components are stored; the framework uses the **first six** (`n_pca_components` in `output/surrogate/active_learning/al_config.json`), a choice justified by the PC dimension study in `experiments/sensitivity/run_pc_dimension.py`. These descriptors were introduced in Danesh et al., *Physical Review Materials* **9**, 075201 (2025), [https://doi.org/10.1103/8zzt-4b7z](https://doi.org/10.1103/8zzt-4b7z).

---

## Cached results

These are the outputs of the computationally expensive stages. `reproduction/reproduce.py` consumes them to reproduce the manuscript's figures and results.

### `output/surrogate/active_learning/` — the selected surrogate

| File                        | Contents                                                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `al_model_selected.pt`    | The surrogate checkpoint used everywhere in the manuscript: the model at**200 observed structures** (iteration 190). |
| `al_config.json`          | Full training configuration and the selection record.                                                                      |
| `al_state.npz`            | Active learning trajectory: MAE history, acquisition order, index splits.                                                  |
| `normalization_stats.npz` | Descriptor and stress standardization statistics.                                                                          |
| `mae_history.png`         | Diagnostic plot of the run.                                                                                                |

A re-run also writes `al_model_current.pt`, the model at the latest completed acquisition, so an interrupted run still leaves a usable checkpoint. It is a recovery artifact rather than a result and is not archived.

Key entries in `al_state.npz`:

| Array                  | Shape               | Meaning                                     |
| ---------------------- | ------------------- | ------------------------------------------- |
| `eval_mae_hist`      | one per iteration   | Held-out MAE in MPa after each acquisition. |
| `picked_indices`     | one per acquisition | Structure indices in acquisition order.     |
| `initial_train_idx`  | (10,)               | The initial design.                         |
| `selected_train_idx` | (200,)              | Training set of the selected checkpoint.    |
| `test_holdout_idx`   | (500,)              | Held-out structures used for the MAE curve. |

`normalization_stats.npz` carries `z_mean`/`z_std` over the six descriptors and `stress_mean`/`stress_std` over the 202 fitted stress rows, plus their `row_labels`.

### `output/inverse_design/` — cached inverse design batches

`sweep_settings.json` records the settings shared by every batch: the 1,000 target indices, the acceptance tolerance (`eta = 5` %), the equal component weights, and the evaluation budgets. `reproduce.py` cross-checks these against its own constants and warns on any disagreement.

Batches are stored as `runs/<combo>/eval_<budget>/inverse_design.npz`, for seven target component combinations (`p11`, `p22`, `p12`, `p11_p22`, `p11_p12`, `p22_p12`, `p11_p22_p12`) and 21 oracle budgets (1, then 10 to 200 in steps of 10).

| Array                                | Shape          | Meaning                                          |
| ------------------------------------ | -------------- | ------------------------------------------------ |
| `random_target_indices`            | (1000,)        | The target structures.                           |
| `random_best_indices`              | (1000,)        | Best structure found for each target.            |
| `random_oracle_eval_counts`        | (1000,)        | Oracle calls spent per target.                   |
| `random_oracle_eval_threshold_met` | (1000,)        | Whether the target was matched within tolerance. |
| `random_oracle_eval_indices`       | (1000,) object | Candidates evaluated, per target.                |
| `random_oracle_eval_mismatches`    | (1000,) object | Mismatch at each evaluation.                     |

The hit rate reported in the manuscript is the mean of `random_oracle_eval_threshold_met`.

### `output/studies/` — sensitivity and baselines

| Directory                | Key arrays                                                                                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc_dimension/`        | `metrics.npz`: `n_pcs` (1–8), `test_mae_mpa`, `test_rmse_mpa`, `per_structure_mae_mpa`, plus `split_indices.npz`, `metrics.csv`, and the per-dimension training histories. |
| `lambda_sensitivity/`  | `results.npz`: `gamma_values` (0, 0.25, 0.5, 1, 2, 4), `eval_steps` (50), `hit_rate_percent` (6 × 50), plus per-target mismatch means, standard deviations, and rankings.        |
| `baseline_comparison/` | One directory per method, each with its own`results.npz`: `proposed/`, `random_search/`, `bo_ei_engineered_features/`, and `bo_ei_raw_image_pcs/`.                              |

Each study directory also stores a `config.json` recording the settings the cache was produced with.

Every per-method baseline file carries the shared context it needs to stand on its own — `config_signature`, `target_indices`, `eval_steps`, `best_nmae_percentile_levels` — together with its own `method_label`, `eval_counts` (1,000 targets), `hit_rate_percent` (50 budgets), `best_nmae_percentiles` (3 × 50), and its method-specific traces. Additionally, `bo_ei_raw_image_pcs/` holds the `raw_structure_pca_06.npz` descriptors it is built on.

---

## Notes on the caches

**Overwriting.** Re-running an experiment script overwrites its cache. Restore the archived values with:

```bash
python download_data.py --force
```

**What is not archived.** `output/figures/` is not in the deposition: every figure is regenerated by `reproduction/reproduce.py` from the cached results.
