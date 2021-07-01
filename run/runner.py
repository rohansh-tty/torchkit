import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from tqdm import tqdm_notebook
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

misclassified_images = []
test_loss_min = np.inf
tb = SummaryWriter()

def train(model, epoch, config=None):
    
  model.train()
#   train_loss, train_acc = [], []
  pbar = tqdm(config.trainloader)
  correct = 0
  processed = 0
  train_loss, running_loss = [], 0
  correct, count = 0, 0
  train_misc_images = []

  optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
  scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
  
 
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(config.device), target.to(config.device)
    
    # init by setting the gradients to zero
    optimizer.zero_grad()

    #Predict
    output = model(data)

    #Loss Calculation
    if config.loss_function == 'CrossEntropyLoss':
          loss_ = nn.CrossEntropyLoss() 
          loss = loss_(output, target)
    elif config.loss_function == 'NLLoss':
        loss = F.nll_loss(output, target)
    
    running_loss += loss
    #Implementing L1 Regularization
    if config.L1Lambda and config.channel_norm=='BatchNorm2d':
        with torch.enable_grad():
          l1_loss = 0
          for param in model.parameters():
            l1_loss += torch.sum(param.abs())
          _l1_reg_loss = (1e-5 * l1_loss)
          running_loss += _l1_reg_loss
     

    train_loss.append(loss)

    
    # BackProp
    loss.backward()
    optimizer.step()

    # lr changes
    if config.lr_scheduler=='OneCycleLR':
      scheduler.step()

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    # Misclassified Images
    if config.misclassified:
      result = pred.eq(target.view_as(pred))
      if count > 430 and count < 460:
          for i in range(0, config.trainloader.batch_size):
              if not result[i]:
                  train_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
    pbar.set_description(desc= f'Train set: batch_id={batch_idx}  Average loss: {loss} Accuracy: {round(100*correct/len(config.trainloader.dataset),3)}')
    # pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    

  config.model_results['TrainAccuracy'].append(100*correct/processed)
  config.model_results['TrainLoss'].append(float(running_loss))
  train_acc_value = correct/len(config.trainloader.dataset)
  train_loss_value = running_loss/len(config.trainloader.dataset)
  

  tb.add_scalar('Train Loss', train_loss_value, epoch)
  tb.add_scalar('Train Accuracy', train_acc_value, epoch)
  
  return train_misc_images



def test(model, epoch, config=None):
    model.eval()
    running_loss = 0
    correct = 0
    # test_loss, test_acc = [], []
    count = 0
    test_misc_images = []
    processed=0
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
    img, label = next(iter(config.testloader))
    test_input = img.to(config.device)

    with torch.no_grad():
        for data, target in config.testloader:
            count += 1
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            if config.loss_function == 'CrossEntropyLoss':
                loss_ = nn.CrossEntropyLoss() #getattr(torch.nn, 'CrossEntropyLoss')
                running_loss += loss_(output, target).item()
                
            elif config.loss_function == 'NLLoss':
                running_loss += F.nll_loss(output, target, reduction='sum').item()

            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            result = pred.eq(target.view_as(pred))
            processed += len(data)

            if config.misclassified:
              if count >40  and count < 70:
                  for i in range(0, config.testloader.batch_size):
                      if not result[i]:
                          test_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
                          


    config.model_results['TestAccuracy'].append(100*correct/processed)
    config.model_results['TestLoss'].append(float(running_loss))
    running_loss /= len(config.testloader.dataset)
    test_acc_value = 100 * correct/len(config.testloader.dataset)
    
    # save model if validation loss has decreased
    global test_loss_min
    if running_loss <= test_loss_min:
          print(f'Validation loss has  decreased from {round(test_loss_min,4)} to {round(running_loss,4)}. Saving the model')
          torch.save(model.state_dict(), 'MNIST.pt')
          test_loss_min = running_loss

    tb.add_scalar('Test Loss', running_loss, epoch)
    tb.add_scalar('Test Accuracy', test_acc_value, epoch)
    tb.add_graph(model, test_input)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        running_loss, correct, len(config.testloader.dataset),
        100. * correct / len(config.testloader.dataset)))
    
    return test_misc_images
    

    
    
    
def fit(model, config):

    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
    
    model_results = defaultdict()
    
    for epoch in range(1,config.EPOCHS+1):
        try:
            lr_val = scheduler.get_last_lr()
        except:
            lr_val = optimizer.param_groups[0]['lr']
            
        
        print('\nEPOCH: ', epoch,'LR: ',lr_val)
      
        # Train and Test Model
        train_misc_images = train(model, epoch, config=config)
        # if config.lr_scheduler=='OneCycleLR':
        #     scheduler.step()
        test_misc_images = test(model, epoch, config=config)
        
        # Scheduler Step, update Learning Rate
        if config.lr_scheduler=='ReduceLROnPlateau':
            scheduler.step(config.model_results['TestLoss'][-1])
        if config.lr_scheduler=='StepLR':
            scheduler.step()
            
         # add lr to tensorboard
        lr = np.array(lr_val)
        tb.add_scalar('Learning Rate', lr, epoch)
        
    tb.close()
        

    return train_misc_images, test_misc_images, model_results
