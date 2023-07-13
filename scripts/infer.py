import os
from modules import scripts_postprocessing

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from repositories.clothSegmentation.data.base_dataset import Normalize_image
from repositories.clothSegmentation.utils.saving_utils import load_checkpoint_mgpu
from repositories.clothSegmentation.networks import U2NET

# device = "cuda"
device = "cpu"

image_dir = "input_images"
result_dir = "output_images"
checkpoint_path = os.path.join("./repositories/clothSegmentation/trained_checkpoint", "cloth_segm_u2net_latest.pth")
do_palette = True

palette1 = [ 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, ]
palette2 = [ 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, ]
palette3 = [0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255]

def run( img: scripts_postprocessing.PostprocessedImage ):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    pbar = tqdm(total=1)

    img = img.convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")

    masks = []
    mask1 = output_img.copy()
    mask1.putpalette(palette1)
    mask1 = mask1.convert("RGB").convert("L")
    masks.append(mask1)

    mask2 = output_img.copy()
    mask2.putpalette(palette2)
    mask2 = mask2.convert("RGB").convert("L")
    masks.append(mask2)

    mask3 = output_img.copy()
    mask3.putpalette(palette3)
    mask3 = mask3.convert("RGB").convert("L")
    masks.append(mask3)

    pbar.update(1)
    pbar.close()

    return masks
