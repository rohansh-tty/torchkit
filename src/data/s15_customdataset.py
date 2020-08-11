from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile


class DepthMapDataset(Dataset):
  
  # Image Segmentation Part
  def __init__(self, data, bg_transforms, bgfg_transforms, mask_transforms, depthmap_transforms): 
    self.bg_images, self.bgfg_images, self.mask_images, self.depth_maps = zip(*data) # do the unzipping part, add this during depth estimation self.depth_maps, self.bg_images,
    self.bg_transforms = bg_transforms
    self.bgfg_transforms = bgfg_transforms
    self.mask_transforms = mask_transforms
    self.depthmap_transforms = depthmap_transforms


  def __len__(self):
    """
    returns len of the dataset
    """
    return len(self.bgfg_images)


  def __getitem__(self, idx):
    """
    returns image data & target for the corresponding index
    """
    try:
      bg_image = Image.open(self.bg_images[idx])#.convert('L') # BG Images
      
      bgfg_image = Image.open(self.bgfg_images[idx])#.convert('L') # BG_FG Images  
      
      mask_image = Image.open(self.mask_images[idx]).convert('L') # Mask Images

      depth_map = Image.open(self.depth_maps[idx]).convert('L') # Depth Images

      ## Transformed Images
      bg_image = self.bg_transforms(bg_image)
      bgfg_image = self.bgfg_transforms(bgfg_image)
      mask_image = self.mask_transforms(mask_image) # first transform the image
      depth_map = self.depthmap_transforms(depth_map)

      input_image = torch.cat((bg_image, bgfg_image), dim=0)

      return {"input":input_image, "bg": bg_image, "bgfg": bgfg_image, "mask": mask_image, "dpmap": depth_map} # dict way of returning Depth Map
      # also add "bg": bg_image,
    

    except Exception as e:
      print("Image {0} skipped due to {1}".format(self.bgfg_images[idx],e)) # this is only if some images can't be identified by PIL

    
