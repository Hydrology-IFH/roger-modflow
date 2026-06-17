## RoGeR-MODFLOW
RoGeR-MODFLOW is a modelling system that couples the soil-vegetation-atmosphere continuum with groundwater.

## Installation

Install the required Python libraries in an environment using conda:
```
conda env create -f conda-environment.yml
conda activate roger-modflow
pip install roger
```

In order to run MODFLOW the related binary files are required. MODFLOW6 binary files can be downloaded using `get_modflow.py` or manually from [MODFLOW executables](https://github.com/MODFLOW-USGS/executables). After extracting you have to put the files in `\bin`. Please note that these files are build for x86-64 CPU architectures.

For ARM64 CPU architectures, it is important to use MODFLOW6 binary files build for ARM64. These files can be downloaded manually from [MODFLOW for ARM](https://github.com/MODFLOW-USGS/modflow6-nightly-build). After extracting you have to put the files in `\bin`.

## Features
RoGeR-MODFLOW provides
-   steady-state simulations using MODFLOW6
-   **offline coupling** between RoGeR and MODFLOW6 using flopy and xmipy
-   **online coupling** between RoGeR and MODFLOW6 using flopy and xmipy

## Basic usage
See README in examples.

## Contact
I have left academia to find a permanent position outside of academia. I will still fix minor bugs, if time is available. For major requests or questions please contact [Markus Weiler](markus.weiler@hydrology.uni-freiburg.de).

## License
This software can be distributed freely under the MIT license. Please read the LICENSE for further information.
© 2024, Robin Schwemmle (<robin.schwemmle@hydrology.uni-freiburg.de>)