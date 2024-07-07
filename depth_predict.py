import torch
import numpy as np

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("/media/wlauer/cobalt/coco/images/train2017/000000000192.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

print('depth numpy', depth_numpy.min(), depth_numpy.max())

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth, background_color=(255, 255, 255, 255), invalid_val=255)

# save colored output
fpath_colored = "output/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
np.save('output/output.npy', depth_numpy)