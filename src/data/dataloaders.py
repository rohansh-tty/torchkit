import torch
import torchvision


class DataLoaders:
  def __init__(self, 
              batch_size=512,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              seed=1):
  
    """
    Arguments:-
    batch_size: Number of images to be passed in each batch
    shuffle(boolean):  If True, then shuffling of the dataset takes place
    num_workers(int): Number of processes that generate batches in parallel
    pin_memory(boolean):
    seed: Random Number, this is to maintain the consistency
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')  # set device to cuda

    if use_cuda:
      torch.manual_seed(seed)
    
    self.dataLoader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True ) if use_cuda else dict(batch_size=1, shuffle=True, num_workers = 1, pin_memory = True)


  def dataLoader(self, data):
    return torch.utils.data.DataLoader(data,**self.dataLoader_args)



def Data_To_Dataloader(trainset,testset,seed=1,batch_size=128,num_workers=4,pin_memory=True):
	"""
	Conv DataSet Obj to DataLoader
	"""

	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
			torch.cuda.manual_seed(SEED)

	dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)

	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)


	return  trainloader, testloader