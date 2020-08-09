
import time
from tqdm import notebook
class Trainer:
  def __init__(self, config):
    self.model = config.model
    self.device = config.device
    self.trainloader = config.trainloader
    self.testloader = config.testloader
    self.epochs = config.epochs
    self.optimizer = config.optimizer
    self.scheduler = config.scheduler
    self.mask_criterion = config.mask_criterion
    self.depth_criterion = config.depth_criterion
    self.valid_loss_min = np.inf

    # initialize training variables
    self.train_losses = []
    self.test_losses = []
    self.learning_rate = []

    # loss coeffs 
    ## TRAIN
    self.dice_scores = []
    self.iou_scores = [] 
    self.d_ssim_index = []
  
    ## TEST 
    self.test_dice_scores = []
    self.test_iou_scores = []
    self.test_d_ssim_index = []
 

    # Sample Container
    self._trainsamples = []
    self._testsamples = []


    # train_loss, valid_loss, dice_score, iou_score, mask_ssim_index
    self.train_loss = 0.0
    self.valid_loss = 0.0

    self.iou_score = 0.0
    self.depth_ssim_index = 0.0
    

    # # TensorboardColab
    # self.tb = TensorboardColab()
    # self.train_iter = 0
    # self.test_iter = 0

    # other variables
    self.epoch = 0


  def train(self):
    self.epoch += 1
    self.model.train()  
    self.epoch_start = time.time()    
    pbar = notebook.tqdm(enumerate(self.trainloader))

    for batch_idx, data in enumerate(pbar):     

        # assigning bgfg and mask with a bgfg and mask image from trainloader
        bg, bgfg, mask, dpmap = data[1]['bg'].to(self.device), data[1]['bgfg'].to(self.device), data[1]['mask'].to(self.device), data[1]['dpmap'].to(self.device)
        
        input_image = data[1]['input'].to(self.device)
      
        # clear the gradients of all optimized variables
        self.optimizer.zero_grad()

        # Run the model on this input batch
        pred_mask, pred_dpmap = self.model(input_image) # 96x96x2
        
        # calculate loss using criterion
        maskloss = self.mask_criterion(pred_mask, mask) # Output, Target
        dpmaploss = self.depth_criterion(pred_dpmap, dpmap)

        depth_ssim_index = ssim(pred_dpmap, dpmap) # ssim loss function
        depth_ssim = torch.clamp((1 - ssim(pred_dpmap, dpmap)) * 0.5, 0, 1)

        
        loss =  (depth_ssim *0.9 + dpmaploss *0.1 + maskloss)

        # for backpropagation
        loss.backward()

        # parameter update
        self.optimizer.step()

        # For checking input images
        # For checking input images
        if self.epoch == 5: #
          self._trainsamples.append({'pred_dpmap':pred_dpmap, 'pred_mask': pred_mask,'dpmap': dpmap, 'mask': mask, 'bgfg': bgfg, 'bg': bg})

        # calculate average training loss
        train_data_size = len(self.trainloader)*self.trainloader.batch_size
        self.train_loss += loss.item() #*trainLoader.batch_size

        self.depth_ssim_index += depth_ssim.item()*self.trainloader.batch_size

        self.iou_score += (1-maskloss.item())*self.trainloader.batch_size
        # rmse_score += (rmse_v.item())*trainLoader.batch_size


        # Update PBAR-TQDM
        pbar.set_description(f'TrainLoss={loss.item():0.3f}; IoU={1-dpmaploss.item():0.3f}')
        
        
    # Calculate Avg Model Params
    train_loss = self.train_loss/len(self.trainloader)
    depth_ssim_index = self.depth_ssim_index/train_data_size
    # rmse = rmse_score/train_data_size
    iou = self.iou_score/train_data_size
    
    # Append the same
    self.train_losses.append(train_loss)
    # self.dice_scores.append(dice_score)
    self.d_ssim_index.append(depth_ssim_index)
    self.iou_scores.append(iou)
    
    
    # # writing model attr to tensorboard
    # self.tb.save_value('IoU Score', 'iou score', self.train_iter, iou_score)
    # self.tb.save_value('Train Loss', 'train loss', self.train_iter, train_loss) 
    

  def test(self):
    self.model.eval()
    pbar = notebook.tqdm(enumerate(self.testloader))
    with torch.no_grad(): 
      for batch_idx, data in enumerate(pbar):
        bg, bgfg, mask, dpmap = data[1]['bg'].to(self.device), data[1]['bgfg'].to(self.device), data[1]['mask'].to(self.device), data[1]['dpmap'].to(self.device)

        # concatenate two input images
        input_image = data[1]['input'].to(self.device)

        # Model Predictions
        pred_mask, pred_dpmap = self.model(input_image) 
        
        # calculate loss using criterion
        maskloss = self.mask_criterion(pred_mask, mask) # Output, Target
        dpmaploss = self.depth_criterion(pred_dpmap, dpmap)

        depth_ssim_index = ssim(pred_dpmap, dpmap)
        depth_ssim = torch.clamp((1 - ssim(pred_dpmap, dpmap)) * 0.5, 0, 1)

        
        loss =  (depth_ssim *0.8 + dpmaploss *0.2 + maskloss)


        # calculate average test loss
        test_data_size = len(self.testloader)*self.testloader.batch_size
        self.valid_loss += loss.item()
        self.depth_ssim_index += depth_ssim_index.item()*self.testloader.batch_size
        # self.dice_score += (1-maskloss.item())*self.testloader.batch_size
        self.iou_score += (1-maskloss.item())*self.testloader.batch_size

        # Update PBAR-TQDM
        pbar.set_description(f'ValidationLoss={loss.item():0.3f}; IoU={1-maskloss.item():0.3f}')


        # For checking prediction results
        # if self.epoch: #
        self._testsamples.append({self.epoch:{'pred_dpmap':pred_dpmap, 'pred_mask': pred_mask,'dpmap': dpmap, 'mask': mask, 'bgfg': bgfg, 'bg': bg}})


      #  current lr
      clr = [param_group['lr'] for param_group in self.optimizer.param_groups]
      print('Learning Rate:', clr)

      # update loss and loss-coefficients
      self.valid_loss = self.valid_loss/len(self.testloader)
      self.depth_ssim_index = self.depth_ssim_index/test_data_size
      # dice_coeff = self.dice_score/test_data_size
      self.iou_coeff = self.iou_score/test_data_size

      # append the same 
      self.test_losses.append(self.valid_loss_min)
      self.test_d_ssim_index.append(self.depth_ssim_index)
      # self.m_ssim_index.append(mask_ssim_index)
      # self.dice_scores.append(dice_coeff)
      self.test_iou_scores.append(self.iou_coeff)
      self.learning_rate.append(clr)


      # update learning rate
      self.scheduler.step(self.valid_loss)



  # print training/validation statistics
    print('IoU_Score={:.3f} '.format(self.iou_coeff))


    self.end_epoch_time = time.time()
    print(f'Epoch {self.epoch} took {self.end_epoch_time-self.epoch_start} seconds.')

    # save model if validation loss has decreased
    if self.valid_loss <= self.valid_loss_min:
      print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.valid_loss_min, self.valid_loss))
      torch.save(self.model.state_dict(), '/content/drive/My Drive/Saved S15 Models/image_segmentation.pt')
      self.valid_loss_min = self.valid_loss
     