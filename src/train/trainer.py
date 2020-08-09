import torch
import time
import torchvision
from tqdm import notebook
import torch.nn.functional as F
from tensorboardcolab import TensorBoardColab
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD


# Initialize TensorBoardColab Object
tb = TensorBoardColab()

# Model Config Class
class ModelConfig:
  def __init__(self, **kwargs):
    for k, v in kwargs:
      setattr(self, k, v)


unet_model = Unet(2,2).to(device)

# Model Configs
model_config = ModelConfig(
    cuda = True if torch.cuda.is_available() else False,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed = 9091,
    model = unet_model, 
    trainloader = trainLoader,
    testloader = testLoader, 
    lr = 0.01,
    mask_criterion = DiceLoss(),
    depth_criterion = IoULoss(),
    optimizer = Adam(unet_model.parameters(),lr=0.01),
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True),
    epochs = 40,
    save_model = False,
    batch_size = 48)


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
    self.train_acc = []
    self.test_losses = []
    self.test_acc = []
    self.learning_rate = []

    # loss coeffs 
    self.dice_scores = []
    self.iou_scores = []
    self.ssim_indices = []
    self.d_ssim_index = []
    self.m_ssim_index = []


    # Sample Container
    self._trainsamples = []
    self._testsamples = []


    # train_loss, valid_loss, dice_score, iou_score, mask_ssim_index
    self.train_loss = 0.0
    self.valid_loss = 0.0
    self.dice_score = 0.0
    self.iou_score = 0.0
    self.depth_ssim_index = 0.0


    # pass time
    self.pass_time = []

    # TensorboardColab
    self.tb = TensorboardColab()
    self.train_iter = 0
    self.test_iter = 0

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
        
        # concatenate two input images
        _input = torch.cat((bg, bgfg), dim=1)
      
        # clear the gradients of all optimized variables
        self.optimizer.zero_grad()

        # Run the model on this input batch
        output = self.model(_input) # 96x96x2
        
        # predictions
        pred_mask = output[:, :1, :, :] # N:C:H:W = 48,1,96,96
        pred_dpmap = output[:, 1:, :, :] # 48,1,96,96

        # calculate loss using criterion
        maskloss = self.mask_criterion(output[:, :1, :, :], mask) # Output, Target
        dpmaploss = self.depth_criterion(output[:, 1:, :, :], dpmap)

        depth_ssim_index = ssim(output[:, 1:, :, :], dpmap) # ssim loss function
        depth_ssim = torch.clamp((1 - ssim(output[:, 1:, :, :], dpmap)) * 0.5, 0, 1)

        
        loss =  2*((depth_ssim *0.9 + dpmaploss *0.1) + maskloss)

        # for backpropagation
        loss.backward()

        # parameter update
        self.optimizer.step()

        # For checking input images
        if batch_idx == 1:
          for i in range(10):
           self._trainsamples.append({'pred_dpmap': pred_dpmap[i], 'pred_mask': pred_mask[i] ,'dpmap': dpmap[i], 'mask': mask[i], 'bgfg': bgfg[i], 'bg': bg[i]})

        # calculate average training loss
        train_data_size = len(self.trainloader)*self.trainloader.batch_size
        self.train_loss += loss.item()*self.trainloader.batch_size
        self.depth_ssim_index += depth_ssim_index.item()*self.trainloader.batch_size
        self.dice_score += (1-maskloss.item())*self.trainloader.batch_size
        self.iou_score += (1-dpmaploss.item())*self.trainloader.batch_size

        # Update PBAR-TQDM
        pbar.set_description(f'SSIM={depth_ssim_index.item():0.2f}; IoU={1-dpmaploss.item():0.3f}')
        
        
    # Calculate Avg Model Params
    train_loss = self.train_loss/train_data_size
    depth_ssim_index = self.depth_ssim_index/train_data_size
    dice_score = self.dice_score/train_data_size
    iou_score = self.iou_score/train_data_size

    # Append the same
    self.train_losses.append(train_loss)
    self.dice_scores.append(dice_score)
    self.ssim_indices.append(depth_ssim_index)
    self.iou_scores.append(iou_score)
    
    
    
    # writing model attr to tensorboard
    self.tb.save_value('IoU Score', 'iou score', self.train_iter, iou_score)
    self.tb.save_value('Train Loss', 'train loss', self.train_iter, train_loss) 
    

  def test(self):
    self.model.eval()
    pbar = notebook.tqdm(enumerate(self.testloader))
    with torch.no_grad(): 
      for batch_idx, data in enumerate(pbar):
        bg, bgfg, mask, dpmap = data[1]['bg'].to(self.device), data[1]['bgfg'].to(self.device), data[1]['mask'].to(self.device), data[1]['dpmap'].to(self.device)

        # concatenate two input images
        _input = torch.cat((bg, bgfg), dim=1)

        # Model Predictions
        output = unet_model(_input) 
        
        # predictions
        pred_mask = output[:, :1, :, :] # N:C:H:W
        pred_dpmap = output[:, 1:, :, :]

        
        # calculate loss using criterion
        maskloss = self.mask_criterion(output[:, :1, :, :], mask) # Output, Target
        dpmaploss = self.depth_criterion(output[:, 1:, :, :], dpmap)

        mask_ssim_index = ssim(output[:, :1, :, :], mask)
        mask_ssim = torch.clamp((1 - ssim(output[:, :1, :, :], mask)) * 0.5, 0, 1)

        depth_ssim_index = ssim(output[:, 1:, :, :], dpmap)
        depth_ssim = torch.clamp((1 - ssim(output[:, 1:, :, :], dpmap)) * 0.5, 0, 1)

        
        loss =  2*((depth_ssim *0.8 + dpmaploss *0.2) + (mask_ssim * 0.7+ maskloss *0.3))

      
        # calculate average test loss
        test_data_size = len(self.testloader)*self.testloader.batch_size
        self.valid_loss += loss.item()*self.testloader.batch_size
        self.depth_ssim_index += depth_ssim_index.item()*self.testloader.batch_size
        self.dice_score += (1-maskloss.item())*self.testloader.batch_size
        self.iou_score += (1-dpmaploss.item())*self.testloader.batch_size

        # Update PBAR-TQDM
        pbar.set_description(f'M_SSIM={mask_ssim_index.item():0.2f}; D_SSIM={depth_ssim_index.item():0.2f}; IoU={1-dpmaploss.item():0.3f}')


        # For checking prediction results
        if batch_idx == len(self.testloader)-1:
          for i in range(10):
            self._testsamples.append({'pred_dpmap': pred_dpmap[i], 'pred_mask': pred_mask[i] ,'dpmap': dpmap[i], 'mask': mask[i], 'bgfg': bgfg[i], 'bg': bg[i]})

      #  current lr
      clr = [param_group['lr'] for param_group in self.optimizer.param_groups]
      print('Learning Rate:', clr)

      # update learning rate
      self.scheduler.step(self.valid_loss)

      # update loss and loss-coefficients
      valid_loss = self.valid_loss/test_data_size
      depth_ssim_index = self.depth_ssim_index/test_data_size
      # mask_ssim_index = self.mask_ssim_index/test_data_size

      dice_coeff = self.dice_score/test_data_size
      iou_coeff = self.iou_score/test_data_size

      # append the same 
      self.test_losses.append(self.valid_loss_min)
      self.d_ssim_index.append(depth_ssim_index)
      self.m_ssim_index.append(mask_ssim_index)
      self.dice_scores.append(dice_coeff)
      self.iou_scores.append(iou_coeff)
      self.learning_rate.append(clr)


  # print training/validation statistics
    print('Test Stats:- Dice_Score={:.3f} ; IoU_Score={:.3f} '.format( dice_coeff, iou_coeff))


    self.end_epoch_time = time.time()
    print(f'Epoch {self.epoch} took {self.end_epoch_time-self.epoch_start} seconds.')

    # save model if validation loss has decreased
    if self.valid_loss <= self.valid_loss_min:
      print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
      torch.save(unet_model.state_dict(), '/content/drive/My Drive/Saved S15 Models/image_segmentation.pt')
      valid_loss_min = valid_loss
      # writing test stats to tensorboard
      self.tb.save_value('IoU Score', 'IoU', self.test_iter, iou_coeff)
      self.tb.save_value('Test Loss', 'test loss', self.test_iter, valid_loss) 
      

def evaluate(configs):
  
  t1 = Trainer(configs)
  for epoch in range(t1.epoch+1):
    t1.train()
    t1.test()
    if t1.scheduler and isinstance(t1.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      t1.scheduler.step(t1.test_losses[-1])
    # self.tb.flush_line()

  # if t1.save_model:
  #     torch.save(t1.model.state_dict(),"UNET_ImageSegmentation.pt")

