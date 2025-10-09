from pathlib import Path
import pandas as pd
import geopandas as gpd
import yaml

base_path = Path(__file__).parent

# load config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

# load observation points
file = base_path / "input" / "sfr_observation_points.csv"
df_observation_points = pd.read_csv(file, sep=";")

# load SFR reach data
file = base_path / "input" / "sfr_packagedata.csv"
df_reaches = pd.read_csv(file, sep=";")
df_reaches["rno"] = df_reaches["rno"] - 1
df_reaches["i"] = df_reaches["i"] - 1
df_reaches["j"] = df_reaches["j"] - 1

rnos = df_reaches.loc[:, "rno"].values.tolist()
labels = df_reaches.loc[:, "rno"].values.astype(str).tolist()

modflow_config["outlet_rnos"] = [int(rno) for rno in rnos]
dict_obs_stage_rnos = {}
dict_obs_flow_rnos = {}
dict_sfr_obs = {}

for label, rno in zip(labels, rnos):
    key = f"{str(label).upper()}_STAGE"
    dict_obs_stage_rnos[key] = int(rno)

for label, rno in zip(labels, rnos):
    key = f"{str(label).upper()}_FLOW"
    dict_obs_flow_rnos[key] = int(rno)

modflow_config["dict_obs_stage_rnos"] = dict_obs_stage_rnos
modflow_config["dict_obs_flow_rnos"] = dict_obs_flow_rnos

for label, rno in zip(labels, rnos):
    key = f"{label}_stage"
    ll = ["stage", int(rno)]
    dict_sfr_obs[key] = ll

    key = f"{label}_flow"
    ll = ["downstream-flow", int(rno)]
    dict_sfr_obs[key] = ll

modflow_config["sfr_obs"] = dict_sfr_obs

# add SFR observation points to config
file_config = base_path / "config.yml"
with open(file_config, "w") as file:
    yaml.dump(modflow_config, file)


