import torch
from torchvision import datasets
from .transforms import train_transforms, test_transforms

def convert(trainset,testset,seed=1,batch_size=128, num_workers=2,pin_memory=True):
	"""
	Converts DataSet Object to DataLoader
	"""
	SEED = 1
	cuda = torch.cuda.is_available()
	torch.manual_seed(SEED)

	if cuda:
			torch.cuda.manual_seed(SEED)

	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
	return  trainloader, testloader





def MNIST_Loader(batch: int, train_transforms=train_transforms, test_transforms=test_transforms):
	torch.manual_seed(1)
	batch_size = batch

	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
						transform=train_transforms),
		batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=test_transforms),
		batch_size=batch_size, shuffle=True, **kwargs)
	
	return train_loader, test_loader