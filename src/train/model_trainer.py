import torch
import torchvision
from tqdm import tqdm_notebook
import torch.nn.Functional as F


class TrainModel:
    def __init__(self, model, trainloader, optimizer, scheduler, mask_criterion):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mask_criterion = mask_criterion

        # initialize training variables
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.learning_rate = []


    def trainit(self):
        self.model.train()
        pbar = tqdm_notebook(enumerate(trainloader))

        for batch_idx, data in pbar:
            bgfg, mask = data['bgfg'].to(self.model.device), data['mask'].to'(self.model.device)

            # set optimzer to zero grad
            self.optimizer.zero_grad()
            
            # get the predictions
            pred_mask = self.model(bgfg)

            # calculate DICE LOSS
            mask_loss = self.mask_criterion(mask_pred, mask_target) 
            loss = mask_loss
            mask_coeff += dice_coefficient(mask_pred,mask_target, mask = True).item() # Segmentation Metric

            # BackProp
            loss.backward()
            self.optimizer.step()
            
            # Batch LR
            _lr = self.optimizer.param_groups[0]['lr']

            # Update PBAR-TQDM
            pbar.set_description(f'Loss={loss:0.3f}')
        print('Train set: Average loss: {:.4f}, Coef: ({:.5f})\n'.format((mask_loss),  (mask_coeff) /(total_length)))
        self.train_losses.append((mask_loss))
        self.train_acc.append( mask_coeff/total_length)
        self.learning_rate.append(_lr)


class TestModel:
    def __init__(self, model, testloader, mask_criterion, scheduler):
        self.model = model
        self.testloader = testloader
        self.mask_criterion = mask_criterion
        self.scheduler = scheduler

        def testit(self):
            self.model.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(testloader):
                    bgfg, mask = data['bgfg'].to(self.model.device), data['mask'].to(self.model.device)

                    # Mask Predictions
                    pred_mask = self.model(bgfg)

                    # Calculate DICE LOSS
                    mask_loss = self.mask_criterion(mask_pred, mask_target) 
                    loss = mask_loss
                    mask_coeff += dice_coefficient(mask_pred,mask_target, mask = True).item() # Segmentation Metric
                
                    if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss)
                self.test_losses.append((mask_loss))
                self.test_acc.append( mask_coeff/total_length)
                self.learning_rate.append(_lr)

class EvaluateModel:
    def __init__(self, model, trainloader, testloader, mask_criterion, optimizer, scheduler):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.mask_criterion = mask_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train = TrainModel(self.model, self.trainloader, self.optimizer, self.scheduler, self.mask_criterion)
        self.test = TestModel(self.model, self.testloader, self.mask_criterion, self.scheduler)

    
    def train_test(self, epochs=10):
        pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
        for epoch in pbar:
            # gc.collect()
            self.train.run()
            self.test.run()
            lr = self.optimizer.param_groups[0]['lr']         
            if self.scheduler and not self.batch_scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            pbar.write(f"Learning Rate = {lr:0.6f}"
