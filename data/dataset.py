from torch.utils.data import Dataset, DataLoader
import torch

class CIFAR_Dataset(Dataset):
    def __init__(self, data, targets, transforms=[]):
        self.image_data = data # np array 
        self.image_targets = targets
        self.image_transforms = transforms
    
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_ = self.image_data[idx]
        target_ = self.image_targets[idx]

        if self.image_transforms:
            image_ = self.image_transforms(image=image_)['image']

        return image_, target_ 
