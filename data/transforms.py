from torchvision import transforms

train_transforms = transforms.Compose([
							transforms.RandomRotation((-6.90,6.90), fill=(1,)),
				# 			transforms.RandomHorizontalFlip(p=0.5),
							transforms.ColorJitter(brightness=0.7, contrast=0.3),
							transforms.ToTensor(),
							transforms.Normalize((0.1307,), (0.3081,))
						])


test_transforms = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1307,), (0.3081,))
						])


