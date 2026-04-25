# Zombie binaries in PTA

Scripts and data supporting the findings of [https://arxiv.org/pdf/2604.20975](https://arxiv.org/pdf/2604.20975).

## Scripts

The zombie-efficiency script can be run using

`apptainer exec ./environment/zombie-env.sif python zombie-efficiency.py ./config/SKA-130-noRN-logz.yaml --ncpu 5 --outdir ./results/SKA-130-noRN-logz/`

The `zombie-env` singularity image can be built in the `environment` folder.

## Data

The PTA configurations used in this work can be found in the `./data/PTA/` folder.
Format: numpy array of size (N_pulsar, 4), storing the unit position vector and distance (in kpc) to each pulsar.

Figures of the paper can be reproduced using the `paper-figures.ipynb` notebook.