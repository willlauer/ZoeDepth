import torch
import numpy as np
from os import listdir, path
from PIL import Image
from tqdm import tqdm

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

data_directory = '/home/wlauer/Documents/cobalt-data/cobalt/train2017/'

files = list(filter(lambda x: x.endswith('.jpg'), listdir(data_directory)))

def convert_depth(file):
    image = Image.open(path.join(data_directory, file)).convert("RGB")
    depth_numpy = zoe.infer_pil(image)
    np.save(path.join(data_directory, file.replace('.jpg', '.npy')), depth_numpy)

for file in tqdm(files):
    convert_depth(file)