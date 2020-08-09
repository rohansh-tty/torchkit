from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np

class AlbumentationTransforms:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms_list=[]):
    transforms_list.append(AP.ToTensor()) # Transform the normalized image to Tensor
    self.transforms = A.Compose(transforms_list)
    print('transforms list', transforms_list)


  def __call__(self, img):
    img = np.array(img) 
    return self.transforms(image=img)['image']

  

class SingleChannel_ImageTransforms:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms_list=[]):
    transforms_list.append(AP.ToTensor()) # Transform the normalized image to Tensor
    self.transforms = A.Compose(transforms_list)
    print('transforms list', transforms_list)


  def __call__(self, img):
    img = np.array(img)
    img = img[:,:,np.newaxis] 
    return self.transforms(image=img)['image']

  