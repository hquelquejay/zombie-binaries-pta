# Zombie binaries in PTA

Scripts and data supporting the results of [https://doi.org/10.1103/9wq2-gqkh](https://doi.org/10.1103/9wq2-gqkh).

## Goal

This project aims to evaluate the detection efficiency of zombie supermassive black-hole binaries in a PTA as a function of binary mass and redshift.

## Reproducing the results

1. Use the environment in the `environment/` folder. The provided `python-env.def` can be used to build the container/image used for the runs.
2. Pick a run configuration from `config/`, for example `config/EPTA-noRN-logz.yaml`.
3. Run the efficiency script:

```bash
python zombie-efficiency.py config/EPTA-noRN-logz.yaml --outdir results/EPTA-noRN-logz --ncpu 5
```

This reads the YAML run settings, loads the PTA definition named by `PTA_config` from `config/PTA/`, computes the efficiency grid, saves `efficiency-grid.npz`, and writes plots to `results/EPTA-noRN-logz/plots/`.

The `PTA_config` entry in each YAML file should match a filename in `config/PTA/`.

The paper figures can be regenerated with `paper-figures.ipynb`.

## Data

The PTA input arrays are stored in `data/PTA/` as NumPy files with shape `(N_pulsar, 4)`, containing the unit position vectors and distances (in kpc) for each pulsar.