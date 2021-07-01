import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm as tqdm

def plot_images(data,file, size=(32,32)):
  r = len(data)//5
  # fig, axes = plt.subplots(12, 5, figsize=(25, 25)) # this means that create a figure of 4 rows with 5 columns and figsize is 15x15
  fig, axes = plt.subplots(r, 5, figsize=(15, 15))
  
  # Plot the images
  for i, img in enumerate(data):
      # Get the image
      image, label, pred = img['image'], img['label'].cpu().numpy()[0], img['pred'].cpu().numpy()[0]
      img = image.cpu().numpy().astype(np.uint8).reshape(size) # convert the image tensor to np array of shape 28x28, 2d image
      
      # Get the appropriate subplot
      x  = i%5         # Subplot x-coordinate
      y  = int(i/5)    # Subplot y-coordinate
      ax = axes[y][x]
      ax.imshow(img, cmap='gray')

      # Format the plot
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      ax.set_title(f'Label: {label} Prediction: {pred}')

   # save the plot
  plt.savefig(file+'.png')
#   files.download(file+'.png')
      

cifar_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def plot_rgb(class_='plane', config=None):
  _data = get_cifar_image_data(config.trainloader, classname=class_)
  fig = plt.figure(figsize=(45,20))
  for idx in np.arange(len(_data[:10])):
    ax = fig.add_subplot(12,12, idx+1, xticks=[], yticks=[]) 
    plt.imshow(_data[idx]) #converting to numpy array as plt needs it.
    ax.set_title(str(class_))

def get_cifar_image_data(data, classname='dog'):
  images, labels = next(iter(data))
  image_index = []
  class_index = cifar_classes.index(classname)
  count = 0
  
  for label in labels:
    if label == class_index:
      image_index.append(count)
    count+=1
  
  images = [images[idx] for idx in image_index]
  np_image_array = [detach_transpose(image) for image in images]
  return np_image_array



def detach_transpose(image):
  img =  image.cpu().detach().numpy()
  img = img.transpose(1,2,0).clip(0,1)
  return img


def calc_mean_sd(data, image_size=32):
  sum_ = torch.tensor([0.0,0.0,0.0])
  sum_sq = torch.tensor([0.0,0.0,0.0])
  batch_sizes = []

  # count = (390*128*32*32)+(80*32*32) # last batch has 80 images
  count = 0
  image_size=32

  for images, labels in tqdm(data):
    # images shape = (batch_size x channels x image_size x image_size)
    batch_sizes.append(images.shape[0])
    count += images.shape[0]*image_size*image_size
    sum_ += images.sum(axis=[0,2,3]) # calculating mean channelwise
    sum_sq += (images**2).sum(axis=[0,2,3])

  total_mean = sum_/count
  total_var = (sum_sq/count) -  (total_mean**2)
  total_std = torch.sqrt(total_var)

  return f'Dataset Mean: {total_mean}, Dataset SD: {total_std}'



def calc_mean_sd(data, image_size=32):
  sum_ = torch.tensor([0.0,0.0,0.0])
  sum_sq = torch.tensor([0.0,0.0,0.0])
  batch_sizes = []

  # count = (390*128*32*32)+(80*32*32) # last batch has 80 images
  count = 0
  image_size=32

  for images, labels in tqdm(data):
  # images shape = (batch_size x channels x image_size x image_size)
    batch_sizes.append(images.shape[0])
    count += images.shape[0]*image_size*image_size
    sum_ += images.sum(axis=[0,2,3]) # calculating mean channelwise
    sum_sq += (images**2).sum(axis=[0,2,3])

  total_mean = sum_/count
  total_var = (sum_sq/count) -  (total_mean**2)
  total_std = torch.sqrt(total_var)

  return f'Dataset Mean: {total_mean}, Dataset SD: {total_std}'



def plot_misclassified_rgb(data,file, size=(32,32,3), config=None):
  r = len(data)//5
  fig, axes = plt.subplots(r, 5, figsize=(15, 15))
  
  # Plot the images
  for i, img in enumerate(data):
      # Get the image
      image, label, pred = img['image'].cpu(), img['label'].cpu().numpy()[0], img['pred'].cpu().numpy()[0]
      # img = image.cpu().numpy().astype(np.uint8).reshape(size) # convert the image tensor to np array of shape 28x28, 2d image
      img = np.transpose(image, (1, 2, 0)) / 2 + 0.5
      label = config.classes[label]
      pred = config.classes[pred]
      
      # Get the appropriate subplot
      x  = i%5         # Subplot x-coordinate
      y  = int(i/5)    # Subplot y-coordinate
      ax = axes[y][x]
      ax.imshow(img, cmap='gray')

      # Format the plot
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      
      ax.set_title(f'Label: {label}\n Prediction: {pred}')

   # save the plot
  plt.savefig(file+'.png')
      
     

def plot_data(data, config, cols=8, rows=5, transform=None):
  figure = plt.figure(figsize=(cols*2, rows*2))
  for i in range(1, cols * rows + 1):
    img, label = data[i]
    figure.add_subplot(rows, cols, i)
    plt.title(config.classes[label])
    plt.axis("off")
    
    img = np.transpose(img, (1,2,0)).numpy().astype(np.float32)
    plt.imshow(img, cmap="gray")

  plt.tight_layout()
  plt.show()