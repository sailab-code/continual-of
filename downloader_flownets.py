import gdown
import pathlib
# Download FlowNetS weights, trained on FlyingChairs, by Clement Pinard [https://github.com/ClementPinard/FlowNetPytorch]
from settings import flownets_weights_path

url = 'https://drive.google.com/uc?id=1jbWiY1C_nqAUJRYZu7mwzV6CK7ugsa5v'

path = pathlib.Path(flownets_weights_path)
path.mkdir(parents=True, exist_ok=True)
output = path.joinpath('net.pth')

gdown.download(url, str(output), quiet=False)