# Porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

Simulation of the soil water and groundwater of the porous aquifer of the Dreisam-Moehlin-Neumagen catchment (Germany) using MODFLOW6 coupled with RoGeR. The project contains a steady-state groundwater model and a transient groundwater model. For transient case, the two models can be coupled offline (i.e. simulations are performed sequentially) or online (i.e. variables/boundary conditions are updated after every time step).

Short description of the folders:
- `steady-state/`: steady-state groundwater model
- `transient/`: transient groundwater model

Data of `input/` or `output/` is stored on FUHYS018 in `StressRes_RoGeR-ModFlow/` since GitHub is not meant to be a large data storage facility. Please contact [Jürgen Strub](juergen.strub@hydrology.uni-freiburg.de) or [Markus Weiler](markus.weiler@hydrology.uni-freiburg.de) to access the data.

In order to run the Python scripts, you have to install the anaconda environment using `../conda-environment.yml`. You can follow the instructions provided in `../README.md`.

See READMEs in the subfolders for more information. I have tried my best to document everything as good as possible and for sure the you will encounter some bugs or incomplete documentation. My advice is that for larger modelling projects a single person is not sufficient and I suggest the four eye principle. Many conceptual and technical issues can be avoided if you are working in development teams.