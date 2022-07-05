### EDIT HERE ###
wandb_mode = "disabled" # online
wandb_project = ""
wandb_entity = ""

smurf_directory = '../sota/smurf'
raft_directory = '../sota/RAFT'

smurf_weights_path = "../sota/smurf/ckpt/smurf/sintel-smurf"
raft_weights_path = "../sota/RAFT/models/"
flownets_weights_path = "../sota/flownets/"


### DO NOT EDIT BELOW ###
import pathlib
smurf_parent_directory = str(pathlib.Path(smurf_directory).parent)
raft_core_directory = str(pathlib.Path(raft_directory).joinpath('core'))
#####################