import torch
import torchvision
from tqdm import notebook
import torch.nn.functional as F
from time import time


class TrainModel:
    def __init__(self, model, device, trainloader, optimizer, scheduler, mask_criterion):
        self.model = model
        self.device = device
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mask_criterion = mask_criterion

        # initialize training variables
        self.train_losses = []
        self.train_acc = []
        self.learning_rate = []
        self.mask_coeff = []

        # pass time
        self.pass_time = []

    def train(self):
        self.model.train()
        pbar = notebook.tqdm(enumerate(self.trainloader))
        mask_coeff=0
        start = time()
        for batch_idx, data in pbar:
            bgfg, mask_target = data['bgfg'].to(self.device), data['mask'].to(self.device)
            
            # set optimzer to zero grad
            self.optimizer.zero_grad()
            
            # get the predictions
            pred_mask = self.model(bgfg)


            # calculate DICE LOSS
            mask_loss, mask_coeff = self.mask_criterion(pred_mask, mask_target) 
            loss = mask_loss

            # print('pred_mask size', pred_mask.size())
            # print('mask_target', mask_target.size())

            mask_coeff = dice_no_threshold(pred_mask, mask_target).item() # Segmentation Metric
            dice_score += mask_coeff*(batch_idx) # trying to calculate the dice score


            # BackProp
            loss.backward()
            self.optimizer.step()
            
            # Batch LR
            _lr = self.optimizer.param_groups[0]['lr']
            

            # Update PBAR-TQDM
            pbar.set_description(f'Loss={loss:0.3f}')
            end = time()
            total_time = end-start


            # if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.scheduler.step(loss)
  
            
        print('Train set: Average loss: {:.4f}, Coef: ({:.5f})\n'.format((mask_loss),  mask_coeff))
        # print('Train set: Mask Coefficient : {:.3f}'.format(1-mask_loss))
        self.pass_time.append(total_time)
        self.train_losses.append((mask_loss))
        self.mask_coeff.append(1-mask_loss)
        self.learning_rate.append(_lr)


class TestModel:
    def __init__(self, model, device, testloader, mask_criterion, scheduler):
        self.model = model
        self.device = device
        self.testloader = testloader
        self.mask_criterion = mask_criterion
        self.scheduler = scheduler
        self.test_losses = []
        self.test_acc = []
        self.mask_coeff = []

        

    def test(self):
        self.model.eval()
        mask_coeff=0
        with torch.no_grad():
          for batch_idx, data in enumerate(self.testloader):
                bgfg, mask_target = data['bgfg'].to(self.device), data['mask'].to(self.device)
                
            
                # Mask Predictions
                pred_mask = self.model(bgfg)
                # print('pred_mask type test', type(pred_mask))
                

                # Calculate DICE LOSS
                mask_loss, mask_coeff = self.mask_criterion(pred_mask, mask_target) 
                loss = mask_loss
                # mask_coeff += dice_coefficient(pred_mask,mask_target, mask = True).item() # Segmentation Metric
            
                # if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                #     self.scheduler.step(loss)

                if batch_idx < 1:
                  for i in range(self.testloader.batch_size):
                    _samples.append({'pred_mask': pred_mask[i] ,'bgfg': bgfg[i]})


              
          print('Test Set: Average loss: {:.4f}, Coef: ({:.5f})\n'.format((mask_loss), mask_coeff))
          # print('Test Set: Mask Coefficient: {:.3f}'.format(1-mask_loss))

          self.mask_coeff.append(1-mask_loss)
          self.test_losses.append((mask_loss))
          # self.test_acc.append( mask_coeff/len(self.testloader))
          # self.learning_rate.append(_lr)


class EvaluateModel:
    def __init__(self, model, trainloader, device, testloader, mask_criterion, optimizer, scheduler):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.mask_criterion = mask_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train = TrainModel(self.model, self.device, self.trainloader, self.optimizer, self.scheduler, self.mask_criterion)
        self.test = TestModel(self.model, self.device,self.testloader, self.mask_criterion, self.scheduler)

    
    def train_test(self, epochs=10):
        pbar = notebook.tqdm(range(1, epochs+1), desc="Epochs")
        for epoch in pbar:
            # gc.collect()
            print('EPOCH:', epoch)
            self.train.train()
            self.test.test()
            lr = self.optimizer.param_groups[0]['lr']     
            print('lr', lr)    
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print('Stepping scheduler...')
                self.scheduler.step()
            # pbar.write(f"Learning Rate = {lr:0.6f}")
