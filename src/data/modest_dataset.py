from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
from time import time


class DepthMapDataset(Dataset):
  
  # Image Segmentation Part
  def __init__(self, data, bgfg_transforms, mask_transforms): #bg_transforms, depthmap_transforms):
    self.bgfg_images, self.mask_images = zip(*data) # do the unzipping part, add this during depth estimation self.depth_maps, self.bg_images,
    # self.bg_transforms = bg_transforms
    self.bgfg_transforms = bgfg_transforms
    self.mask_transforms = mask_transforms
    # self.depthmap_transforms = depthmap_transforms.


  def __len__(self):
    """
    returns len of the dataset
    """
    return len(self.bgfg_images)


  def __getitem__(self, idx):
    """
    returns image data & target for the corresponding index
    """
    overall = 0 
    try:
      start = time()
      # print('bg fg image', self.bgfg_images[idx])
      # bg_image = Image.open(self.bg_images[idx]) # BG Images
      bgfg_image = Image.open(self.bgfg_images[idx]) # BG_FG Images
      mask_image = Image.open(self.mask_images[idx]) # Mask Images
      # depth_map = Image.open(self.depth_maps[idx]) # Depth Images

      ## Transformed...
      # bg_image = self.bg_transforms(bg_image)
      bgfg_image = self.bgfg_transforms(bgfg_image)
      mask_image = self.mask_transforms(mask_image)
      # # depth_map = self.depthmap_transforms(depth_map)

      end = time()
      overall += (end-start)
      # print('Image transformed to a Tensor.')
      print("Took {0} seconds to load and transform".format(overall))
      return { "bgfg": bgfg_image, "mask": mask_image} #, "depthmap": depth_map} # dict way of returning Depth Map
      # also add "bg": bg_image,
      


    except Exception as e:
      print("Image {0} skipped due to {1}".format(self.bgfg_images[idx],e)) # this is only if some images can't be identified by PIL

    
