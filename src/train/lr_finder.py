# LR FINDER 
import copy
import tqdm
from torch.optim import Adam



LR_List = []
Acc_List = []
Loss_List = []
def lr_rangetest(device, 
                model,
                trainloader, 
                criterion,  
                minlr, 
                maxlr, 
                epochs,
                weight_decay=0.05,
                pltTest=True):
    """
    Args:-
    1. Device: Assigned Device
    2. TrainLoader: Dataloader for train dataset
    3. criterion: Wrapped Loss function
    4. minlr: Minimum Learning Rate 
    5. maxlr: Maximum Learning Rate
    6. epochs: Number of Epochs you want your model to run
    """
    lr = minlr
    testModel = copy.deepcopy(model)
    for e in range(1,epochs+1):
        optimizer = Adam(testModel.parameters(), lr = lr)
        lr_step = 0.1*(maxlr-minlr)/epochs
        train_loss = 0
        testModel.train()
        pbar = notebook.tqdm(enumerate(trainloader))
        # correct, processed = 0, 0
        # print('*'*40)
        print(f'EPOCH:- {e}')
        print(f'Learning Rate:- {optimizer.param_groups[0]["lr"]}')

        for batch_idx, data in enumerate(pbar):
            lr = lr + lr_step
            bg, bgfg, mask = data[1]['bg'].to(device), data[1]['bgfg'].to(device), data[1]['mask'].to(device)

            # concatenate two input images
            _input = torch.cat((bg, bgfg), dim=1)

            optimizer.zero_grad()

            # Model Predictions
            pred_mask = unet_model(_input)
            loss = mask_criterion(pred_mask, mask)
            loss.backward()
            optimizer.step()

            # Update TQDM BAR
            pbar.set_description(f'Train Loss={loss.item():0.3f}')

            # calculate average training loss
            train_data_size = len(trainLoader)*trainLoader.batch_size
            train_loss += loss.item()*trainLoader.batch_size
             
        # Update PBAR-TQDM
        pbar.set_description(f'Train Loss={loss.item():0.3f}')
        
        # append train loss and learning rate
        train_loss = train_loss/train_data_size
        print('Train Loss:', round(train_loss,4))
        print('-'*40)

        Loss_List.append(train_loss)
        LR_List.append(optimizer.param_groups[0]['lr'])
        
    if pltTest:
        with plt.style.context('fivethirtyeight'):
            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
            plt.rcParams["figure.figsize"] = (10,6)
            plt.subplot(2,1,2)
            plt.plot(LR_List, Loss_List)
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.show()


lr_rangetest(device, unet_model, testLoader, mask_criterion, 0.0001, 0.1, 100, weight_decay=0.05, pltTest=True)