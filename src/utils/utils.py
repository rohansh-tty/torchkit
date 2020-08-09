import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from random import randint

# you need to import DepthMapDataset

image_name = ('Background', 'Background-Foreground', 'Mask', 'Depth Map' )

def display_images(dataset_obj, idx :int, ax=None, title=None):
  """
  Displays Images of 3/4 Different Version on passing the index
  """
  if isinstance(idx, int):

      # Initialize something like this and pass to display_images()
    # dataset_obj = DepthMapDataset(img_seg, bg_transform, bgfg_transform, bgfg_mask_transform, dpmap_transform) # Instance of Dataset Class
    
    bg, bgfg, mask, dpmap = dataset_obj[idx]['bg'], dataset_obj[idx]['bgfg'], dataset_obj[idx]['mask'], dataset_obj[idx]['depthmap'] # Tensors of BG,BGFG,MASK
    image_set = (bg, bgfg, mask, dpmap)
    print('image set', image_set)

    # images = [image_name for img in image_set]
    image_len = len(image_set)
    np_images = [i.detach().cpu().numpy() for i in image_set] # converting tensor images to numpy format
    axes=[]
    r,c=1,4 # r=1, c=4 with 4 classes

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig = plt.figure(figsize=(14,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    print('len of np_images:',len(np_images))
    for i in range(len(np_images)):
      ax = plt.subplot(r, c, i+1) 
      ax.text(70, -4, image_name[i], fontsize=14, fontfamily='monospace')  

      plt.subplot(r, c, i+1) 
      plt.axis('off')
      print('i=',i)
      print('np_images',np_images[i])
      plt.imshow(np.clip(std*(np_images[i].transpose((1,2,0)))+mean, 0, 1))
      
    plt.show()



def gridshow(file: list, num: int,  indices=None, imgsize=(20,3)):
    """
    Func to display "n" images in grid fashion
    Arguments:-
    file: Zip File with Images
    num: Number of Images
    indices: Only the file has some images zipped in, eg: FG_BG and it's Mask Images
    imgsize: Tuple of Image Size
    """
    
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
