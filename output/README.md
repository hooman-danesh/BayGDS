# `output/` — cached results and generated figures

This directory is populated by the downloader and by the analysis scripts, not by version control.

```bash
python download_data.py
```

After that it contains:

```
output/
├── surrogate/active_learning/   Selected surrogate checkpoint and AL trajectory
├── inverse_design/              Cached inverse design batches, 7 combos x 21 budgets
└── studies/
    ├── pc_dimension/                Descriptor-dimension sensitivity
    ├── lambda_sensitivity/          Uncertainty weight sensitivity
    └── baseline_comparison/         One directory per search method:
        ├── proposed/                    Proposed-framework sequences
        ├── random_search/               Full library random search
        ├── bo_ei_engineered_features/   BO--EI on the six proposed features
        └── bo_ei_raw_image_pcs/         BO--EI on six raw-image PCs
```

`reproduction/reproduce.py` then creates:

```
output/
├── figures/             All manuscript figures as paired PDF and PNG,
│                        plus inverse_design_summary.{csv,md}
└── reproduction.log     Full run log with the tables and selected checkpoint
```

Figures are regenerated, never archived. See `docs/DATA.md` for what each cached file holds.
