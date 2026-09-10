import cv2
import numpy as np 
import random
import torch

print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
from pathlib import Path
random.seed(42)
np.random.seed(42)
# raw images 
path = Path(__file__).parent.parent
data_path = path / "data/raw/Kvasir-SEG"
yolo_data_path = path / "data/processed/Kvasir-SEG"



