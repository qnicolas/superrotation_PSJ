# Mechanisms of Superrotation in Slowly-Rotating and Tidally-Locked Planets - code

## Overview

This folder contains code to run two-level Dedalus simulations of planetary atmospheres and reproduce the figures appearing in Nicolas & Vallis (2026): Mechanisms of Superrotation in Slowly-Rotating and Tidally-Locked Planets.  

## Requirements

A .yml file is included that contains all necessary python packages to run the code. You can create the conda environment using `conda env create -f environment.yml`, then activate with `conda activate dedalus3`. See [this link](https://www.anaconda.com/docs/getting-started/miniconda/main) for more info on how to install conda.

## Running simulations

The dedalus code to run the two-level model, which defines in particular various settings for the resolution, stop time, output, etc, is `two_level_nondimensional.py`. More info on Dedalus [here](https://dedalus-project.readthedocs.io/en/latest/).

You can run the code serially using `python two_level_nondimensional.py --Ro=1 --Trad=100 --type=locked`. The `Ro` and `Trad` arguments set the thermal Rossby number and nondimensional radiative relaxation timescale. `type` sets whether the planet is tidally locked (`type=locked`) or axisymmetrically forced  (`type=axi`). `E` (Ekman number) and `stratification` are optional arguments. Use `python two_level_nondimensional.py -h` to see the list of all arguments and default values.

You can run the code in parallel with MPI simply by calling `mpiexec -n 4 python two_level_nondimensional.py --Ro=1 --Trad=100 --type=locked` (here using 4 cores, but you may use more or less depening on your setup and the resolution you use).

For thermal Rossby numbers higher than 10, you may have to reduce the time step and perhaps increase the resolution.

## Reproducing Figures

Several Jupyter notebooks are provided with instructions and code to produce the Figures. Use:
 - `figures_BVP.ipynb` to reproduce Figures 2 and 3,
 - `figures_EVP.ipynb` to reproduce Figure 4,
 - `figures_tidally_locked.ipynb` to reproduce Figures 5--9,
 - `figures_axi.ipynb` to reproduce Figures 10--12,
 - `figures_transition.ipynb` to reproduce Figures 13--14.

## Data availability

Simulation output is too heavy to be made publicly available, but you can recreate it by running the simulations yourself (see above). It is also available upon request.

## Citation

If you use any of the code in your work, we simply ask you to cite the paper.

For any questions, do not hesitate to email me!
