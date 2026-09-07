# Data-efficient Bayesian-guided design selection from large candidate sets: Application to hyperelastic stochastic metamaterials

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.matdes.2026.116984-B31B1B)](https://doi.org/10.1016/j.matdes.2026.116984) [![Data](https://img.shields.io/badge/Data-10.5281%2Fzenodo.19009893-1682D4)](https://doi.org/10.5281/zenodo.19009893) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains the code accompanying the paper:

> H. Danesh and H. Wessels, *Data-efficient Bayesian-guided design selection from large candidate sets: Application to hyperelastic stochastic metamaterials*, **Materials & Design** (2026), 116984. [https://doi.org/10.1016/j.matdes.2026.116984](https://doi.org/10.1016/j.matdes.2026.116984)

![Graphical abstract: a target stress response and a candidate set of 50,000 metamaterial designs feed a Bayesian-guided selection loop, where a surrogate trained by feature engineering and active learning narrows the pool of admissible designs, reaching over 90 percent success within about 20 high-fidelity evaluations.](assets/graphical_abstract.png)

The repository holds the source code. The dataset, the trained surrogate, and the cached experiment results are archived on Zenodo and fetched by `download_data.py`. Together they regenerate every figure and table in the paper.

The method learns low-dimensional descriptors of  binary microstructures from two-point correlations, trains a multi-output Gaussian process surrogate of the nonlinear stress response via active learning, and uses the surrogate's posterior mean and uncertainty to rank candidate structures against a target stress response. The high-fidelity solver (oracle) then selects the optimal design.

---

## Setting up the environment

Create and activate a conda environment, then install the pinned dependencies into it:

```bash
conda create -n baygds python=3.13
conda activate baygds
python -m pip install -r requirements.txt
```

Python 3.11 or newer works; 3.13 is what the manuscript results were produced with. Every command in this README assumes the `baygds` environment is active.

**LaTeX.** The figures are typeset through Matplotlib's `text.usetex`, so a working LaTeX installation (TeX Live or MacTeX, with `amsmath` and `amssymb`) is needed to regenerate them. Without one, set `"text.usetex": False` in `setup_latex_style()` in `reproduction/plotting.py`; the figures then render with Matplotlib's own math text and differ cosmetically from the published versions.

The manuscript results were produced with Python 3.13.7 on macOS, CPU only, using NumPy 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2, pandas 2.3.2, Matplotlib 3.10.6, PyTorch 2.5.1, and GPyTorch 1.14.3.

---

## Quick start

With the environment active, reproduce every code-produced figure and table in the manuscript in two commands:

```bash
python download_data.py            # fetches the archive from Zenodo
python reproduction/reproduce.py
```

Figures are written to `output/figures/` as paired PDF and PNG files, the summary tables to `output/figures/inverse_design_summary.{csv,md}`, and a full run log to `output/reproduction.log`.

`reproduction/reproduce.ipynb` is the annotated notebook version of the same pipeline, with explanatory text for each step. Both produce identical output. To run it, register the environment as a kernel first:

```bash
python -m ipykernel install --user --name baygds
jupyter lab reproduction/reproduce.ipynb
```

No GPU is required, and no step of the reproduction retrains a model or calls the oracle: it loads cached results and regenerates the figures from them.

---

## Repository layout

```
.
├── download_data.py          Fetch and extract the Zenodo archive
├── zenodo_manifest.json      Archive name, size, and MD5 checksum
├── requirements.txt          Python dependencies
├── assets/                   Graphical abstract
│
├── src/                      Reusable, dataset-independent method code
│   ├── feature_engineering.py    Two-point statistics and PCA descriptors
│   ├── gp_core.py                Multi-output variational GP with an LMC structure
│   ├── active_learning.py        Uncertainty-driven acquisition loop
│   ├── inverse_design.py         Surrogate-guided candidate ranking
│   ├── oracle.py                 Interface to the high-fidelity response
│   └── data_utils.py             Dataset and descriptor loading
│
├── experiments/              Dataset-specific runs reported in the manuscript
│   ├── baygds/       			  The three stages of the proposed method
│   ├── sensitivity/              PC dimension and uncertainty weight studies
│   └── baselines/                Random search and BO--EI comparisons
│
├── reproduction/             Figure and table generation
│   ├── reproduce.py              Non-interactive full reproduction
│   ├── reproduce.ipynb           Annotated notebook, same output
│   └── plotting.py               All manuscript plotting functions
│
└── docs/
    └── DATA.md               Contents and format of every archived file
```

`data/` and `output/` are created by `download_data.py` and are excluded from version control.

---

## Data

`download_data.py` retrieves a single archive, `baygds-data.tar.gz`, and unpacks it in place. It contains:

- the 50,000 binary microstructures and their PC score descriptors,
- the oracle stress responses at 0° and 45° loading,
- the selected surrogate checkpoint and its active learning trajectory,
- the cached inverse design, sensitivity, and baseline results.

Useful options:

```bash
python download_data.py --list     # show the archive and where it extracts
python download_data.py --check    # verify what is already present
python download_data.py --force    # re-download and overwrite
```

Each download is verified against the MD5 checksum in `zenodo_manifest.json` before extraction. Zenodo occasionally answers with a 502, 503 or 504 while a gateway is busy; the download retries with a growing delay and resumes a partial transfer where it stopped, so re-running the command is enough. If the service is down for longer, fetch the archive from the record page by hand and point the script at it with `--from-dir`. `docs/DATA.md` documents every array, its shape, and its units.

The archive is deposited at [https://doi.org/10.5281/zenodo.19009893](https://doi.org/10.5281/zenodo.19009893) (CC BY 4.0).

---

## Re-running the experiments from scratch

The reproduction above reuses the cached results. To recompute those caches from the inputs instead, run the stages below in order from the repository root. These are the computationally expensive stages, and none of them is needed to reproduce the manuscript figures.

**1. Feature engineering** — performs two-point statistics computation and PCA over 50,000 structures.

```bash
python -m experiments.baygds.run_feature_engineering
```

**2. Active learning** — trains the multi-output variational GP surrogate using active sampling.

```bash
python -m experiments.baygds.run_active_learning
```

**3. Design selection** — selects candidates for 1,000 random targets across seven target-component combinations and oracle budgets up to 200 evaluations.

```bash
python -m experiments.baygds.run_inverse_design
```

**Sensitivity studies** — surrogate accuracy against descriptor dimension, and inverse design hit rate against the uncertainty weight.

```bash
python -m experiments.sensitivity.run_pc_dimension
python -m experiments.sensitivity.run_lambda_sensitivity
```

**Search baselines** — random search, Bayesian optimization using expected improvement (BO–EI) on the proposed features, and BO–EI on raw-image PCs, followed by the combined comparison figure.

```bash
python -m experiments.baselines.run_baseline_comparison --n-jobs 4
```

Recomputing a stage overwrites the corresponding cached result. Keep a copy of `output/` first if you want to compare against the archived values, or restore them afterwards with `python download_data.py --force`.

---

## Documentation

- `docs/DATA.md` — every archived array, its shape, and its units.
- Each module in `src/` documents its purpose, parameters, and outputs in its own docstring.

---

## Citation

If you use this code or the archived data in your work, please cite the paper:

```bibtex
@article{danesh2026dataefficient,
  title   = {Data-efficient Bayesian-guided design selection from large
             candidate sets: Application to hyperelastic stochastic metamaterials},
  author  = {Danesh, Hooman and Wessels, Henning},
  journal = {Materials \& Design},
  year    = {2026},
  pages   = {116984},
  issn    = {0264-1275},
  doi     = {10.1016/j.matdes.2026.116984},
}
```

If you use the archived data, please also cite the deposition:

```bibtex
@dataset{danesh2026baygdsdata,
  title     = {Dataset for the publication "Data-efficient Bayesian-guided
               design selection from large candidate sets: Application to
               hyperelastic stochastic metamaterials"},
  author    = {Danesh, Hooman and Wessels, Henning},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19009893},
}
```

The descriptor construction follows Danesh et al., *Physical Review Materials* **9**, 075201 (2025), [https://doi.org/10.1103/8zzt-4b7z](https://doi.org/10.1103/8zzt-4b7z).

---

## License

MIT License. See `LICENSE`.

## Contact

Hooman Danesh — <hooman.danesh@tu-braunschweig.de> Division of Data-Driven Modeling of Mechanical Systems, Institute of Applied Mechanics, Technische Universität Braunschweig
