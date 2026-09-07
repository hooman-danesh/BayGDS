# `data/` — raw inputs

This directory is populated by the downloader, not by version control:

```bash
python download_data.py
```

After that it contains:

```
data/
├── structures/structures.npz    50,000 binary microstructures, 96 x 96
├── pca/pc_scores.npz            PC score descriptors, 50,000 x 8
└── oracle/
    ├── oracle_0deg.npz          Oracle stress responses, 0 deg loading
    └── oracle_45deg.npz         Oracle stress responses, 45 deg loading
```

See `docs/DATA.md` for the arrays inside each file, their shapes, and their units.
