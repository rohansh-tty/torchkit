import torch
from IPython.display import Image, clear_output 
import time as t
import zipfile
import os


# Hardware Properties
def hardware_specs():
    return 'PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# Unzip the File
def zip_data(zipfile, path_to_zipfile, directory_to_extract):
  with zipfile.ZipFile(path_to_zipfile, 'r') as zip:
    zip.extractall(directory_to_extract)
  print("Zipfile extracted to ", directory_to_extract)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# For BG 
# BG DATASET
bg_path ="/content/drive/My Drive/ZipFiles/BG DATASET.zip"
zip_data(zipfile, bg_path, directory_to_extract)

# FG DATASET
fg_path =  "/content/drive/My Drive/ZipFiles/FG DATASET.zip"
zip_data(zipfile, fg_path, directory_to_extract)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Set the path
zip_path = "/content/drive/My Drive/ZipExtract/data_TenK"
subd = os.listdir(zip_path) #Returns sub-directories
bg_path, bgfg_mask_path = os.path.join(zip_path, subd[0]), os.path.join(zip_path, subd[1])
bgfg_path, depthmap_path = os.path.join(zip_path, subd[2]), os.path.join(zip_path, subd[3])



# something extra you might need in the future
# # BG 
# bg_zip_path = "/content/drive/My Drive/ZipExtract/BG DATASET"
# bg_subd = os.listdir(bg_zip_path) #Returns sub-directories

# # set the paths
# bg_path, bg_mask_path = os.path.join(bg_zip_path, bg_subd[0]), os.path.join(bg_zip_path, bg_subd[1])

# # FG
# fg_zip_path = "/content/drive/My Drive/ZipExtract/FG DATASET"
# fg_subd = os.listdir(fg_zip_path) #Returns sub-directories

# # set the paths
# fg_path, fg_mask_path = os.path.join(fg_zip_path, fg_subd[0]), os.path.join(fg_zip_path, fg_subd[1])




#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Zipping Dataset for image display

# BACKGROUND
bg_mask_files = [os.path.join(bg_mask_path, bg_mask) for bg_mask in os.listdir(bg_mask_path)] # fg_mask Files
bg_files = [os.path.join(bg_path, bg) for bg in os.listdir(bg_path)] # fg files
bg_zip = list(zip(bg_files, bg_mask_files))
bg_zip # consists of Backgrund with corresponding mask images(Black) zipped in


# FOREGROUND
fg_mask_files = [os.path.join(fg_mask_path, fg_mask) for fg_mask in os.listdir(fg_mask_path)] # fg_mask Files
fg_files = [os.path.join(fg_path, fg) for fg in os.listdir(fg_path)] # fg files
fg_zip = list(zip(fg_files, fg_mask_files))
fg_zip # consists of Foregrounds with corresponding mask images zipped in

# DEPTH MAPS
dpmap_path = '/content/drive/My Drive/ZipExtract/data_TenK/DepthMaps'
depth_map_files = [os.path.join(dpmap_path, dpmap) for dpmap in os.listdir(dpmap_path)]
depth_map_files[-1]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Display Grid Images

import matplotlib.pyplot as plt
from PIL import Image

def gridshow(file: list, num: int,  indices=None, imgsize=(20,3)):
  file_len = len(file)
  if indices is None:  
    indices = [randint(1,file_len) for img in range(num)]
    print('indices', indices)

  images = [file[indices[i]] for i in range(len(indices))]
  r,c = 1, num

  fig = plt.figure(figsize=imgsize)
  fig.subplots_adjust(hspace=0.01, wspace=0.05)

  for i in range(len(images)):
      img = Image.open(images[i])
      ax = plt.subplot(r, c, i+1) 
      # ax.text(70, -4, image_name[i], fontsize=14, fontfamily='monospace')  

      plt.subplot(r, c,  i+1) 
      plt.axis('off')
      plt.imshow(img, cmap='gray', aspect='auto')
      
  plt.show()

# Foreground
gridshow(fg_files, 5, imgsize=(10,2))

# Background
gridshow(bg_files, 5, imgsize=(10,2))

# Foreground Mask
gridshow(fg_mask_files, 5, indices=[177, 112, 160, 74, 194], imgsize=(10,2))

# FGBG 
gridshow(bgfg_files, 5, imgsize=(10,2))

# FGBG Mask
gridshow(bgfg_mask_files, 5, [1532, 2050, 5643, 5280, 1014])

# Depth Maps
gridshow(depth_map_files, 5, [1532, 2050, 5643, 5280, 1014], imgsize=(10,2))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# DEPTH ESTIMATION FILE

bgfg_files = [os.path.join(bgfg_path, bgfg) for bgfg in os.listdir(bgfg_path)]
dpmap_files = [os.path.join(depthmap_path, map) for map in os.listdir(depthmap_path)]

dp_est = list(zip(bgfg_files,dpmap_files))
# # # ziplist = list(zip1)
