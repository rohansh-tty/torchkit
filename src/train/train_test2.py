import dill as dill
import torch
import time
from evaluation_metrics.accuracy import dice_coefficient

#Training & Testing Loops
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook


# MODEL STATS
train_losses = []
test_losses = []
train_acc = []
test_acc = []
running_mask_loss=0.0
# running_depth_loss=0.0

# Training

def train(model, device, train_loader, optimizer, mask_criterion, epoch, scheduler = False):
  running_mask_loss = 0.0
#   running_depth_loss=0.0
  
  model.train()
  mask_coef = 0
#   depth_coef = 0
  
  pbar = tqdm(train_loader)
  total_length = len(train_loader)
  for batch_idx, (data, mask_target) in enumerate(pbar):
    
    # get samples
    data, mask_target, depth_target = data.to(device), mask_target.to(device), depth_target.to(device)

    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    mask_target = mask_target.unsqueeze_(1)
    # depth_target = depth_target.unsqueeze_(1)

    mask_pred = model(data)

    # Calculate loss and DICE coefficient
    mask_loss = mask_criterion(  mask_pred,mask_target,)
    # depth_loss = depth_criterion(depth_pred,depth_target)
    loss = mask_loss 
    mask_coef += dice_coefficient(mask_pred,mask_target, mask= True).item()
    # depth_coef += dice_coefficient(depth_pred, depth_target, mask=False).item()
    
    # Backpropagation
    torch.autograd.backward([mask_loss])

    optimizer.step()
    if(scheduler):
      scheduler.step()

    pbar.set_description(f'Loss={loss:0.4f}')
  print('\nTrain set: Average loss: {:.4f}, Coef: ({:.5f})\n'.format((mask_loss,  (mask_coef)/(total_length))))
  train_losses.append((mask_loss/total_length))
  train_acc.append((mask_coef/total_length))
#   tb.add_scalar('Mask Train Loss', mask_loss/total_length, epoch)
#   tb.add_scalar('Depth Train Loss', depth_loss/total_length, epoch)
#   tb.add_scalar('Total Train Loss', (mask_loss+depth_loss)/(2*total_length), epoch)
#   tb.add_scalar('Mask Train Coef',mask_coef/total_length, epoch)
#   tb.add_scalar('Depth Train Coef', depth_coef/total_length, epoch)
#   tb.add_scalar('Total Train Coef', (mask_coef+depth_coef)/(2*total_length), epoch)
  return train_losses, train_acc
  



# Testing
def test(model, device, mask_criterion, depth_criterion, test_loader, epoch):
    model.eval()
    mask_loss = 0
    # depth_loss = 0
    
    correct = 0
    mask_coef = 0
    # depth_coef = 0
    
    total_length = len(test_loader)

    with torch.no_grad():
        for data, mask_target in tqdm(test_loader):
            data, mask_target = data.to(device), mask_target.to(device)
            
            mask_target = mask_target.unsqueeze_(1)
            # depth_target = depth_target.unsqueeze_(1)

            mask_pred= model(data) # predict masks
            mask_loss += mask_criterion(mask_pred, mask_target,).item()  # sum up batch loss
            # depth_loss += depth_criterion(depth_pred,mask_target,).item()
            
            # calculate test loss
            test_loss = mask_loss
          
            # calculate dice coefficient
            mask_coef += dice_coefficient(mask_pred,mask_target, mask= True).item()
            # depth_coef += dice_coefficient(depth_pred, depth_target, mask=False).item()
            
    # print(test_loss)
    test_loss /= (total_length)
    test_losses.append((mask_loss/total_length))

    print('\nTest set: Average loss: {:.4f}, Coef: ({:.5f})\n'.format(
        test_loss, 
         (mask_coef) /(total_length)))
    
    test_acc.append((mask_coef/total_length))
    # tb.add_scalar('Mask Test Loss', mask_loss/total_length, epoch)
    # tb.add_scalar('Depth Test Loss', depth_loss/total_length, epoch)
    # tb.add_scalar('Total Test Loss', (mask_loss+depth_loss)/(2*total_length), epoch)
    # tb.add_scalar('Mask Test Coef',mask_coef/total_length, epoch)
    # tb.add_scalar('Depth Test Coef', depth_coef/total_length, epoch)
    # tb.add_scalar('Total Test Coef', (mask_coef+depth_coef)/(2*total_length), epoch)
    return test_losses,test_acc




LR = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
train_scheduler = False


# Runner for both train and test

def train_model(model,device,trainloader,testloader,optimizer,mask_criterion,EPOCHS,scheduler = False,batch_scheduler = False ,best_loss = 1000,path = "/content/gdrive/My Drive/API/bestmodel.pt"):
  start = time.time()
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch+1,'LR:',optimizer.param_groups[0]['lr'])
      LR.append(optimizer.param_groups[0]['lr'])
      train_scheduler = False

      if(batch_scheduler):
        train_scheduler = scheduler
      train_loss, train_acc = train(model, device, trainloader, optimizer, mask_criterion, epoch,train_scheduler)
      
      test_loss , test_acc = test(model, device, mask_criterion, testloader,epoch)
      if(scheduler and not batch_scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)): 
        scheduler.step()

      elif(scheduler and not batch_scheduler and  isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step(test_loss[-1][2])
    
      
      if(test_loss[-1][2]<best_loss):
        print("loss reduced, Saving model....")
        best_loss = test_loss[-1][2]
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': test_loss,
              'acc' : test_acc
              }, path,pickle_module=dill)
        
      print('----------------------------------------------------------------------------------')
    
  end = time.time()
  print(f"Traning took {round((end - start),3)}s for {EPOCHS} epochs")

 
