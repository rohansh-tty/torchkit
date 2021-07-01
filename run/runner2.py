import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch
from collections import defaultdict

def train(model, config, scheduler):
    train_loss = []
    train_acc = []

    model.train()
    pbar = tqdm(config.trainloader)
    correct = 0
    processed = 0
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(config.device), target.to(config.device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        if config.L1Lambda:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss +=  1e-5 * l1

        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        # lr changes
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Train set: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_loss, train_acc


def test(model, config):
    model.eval()
    test_loss_value = 0
    correct = 0
    test_misc_images = []
    count = 0
    with torch.no_grad():
        for data, target in config.testloader:
            count += 1
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            # sum up batch loss
            test_loss_value += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            result = pred.eq(target.view_as(pred))

    if config.misclassified:
        if count >40  and count < 70:
            for i in range(0, config.testloader.batch_size):
                if not result[i]:
                    test_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
        

    test_loss_value /= len(config.testloader.dataset)

    print('\nTest set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_value, correct, len(config.testloader.dataset),
        100. * correct / len(config.testloader.dataset)))

    test_acc = 100. * correct / len(config.testloader.dataset)

    return test_loss_value, test_acc, test_misc_images


def run(model, config):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    model_results = defaultdict()
    misclassified=None

    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=0.2,
                                                steps_per_epoch=len(config.trainloader),
                                                epochs=config.EPOCHS) 

    lr_list = []

    print('='*10+'RUNNING THE MODEL'+'='*10)
    for epoch in range(config.EPOCHS):
        print('EPOCH {} | LR {}: '.format(epoch+1, scheduler.get_last_lr()))
        lr_list.append(scheduler.get_last_lr())
        train_epoch_loss, train_epoch_acc = train(model, config, scheduler)
        test_loss_val, test_acc_val, test_misc_images = test(model, config)

        train_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        train_acc.append(sum(train_epoch_acc)/len(train_epoch_acc))
        test_loss.append(test_loss_val)
        test_acc.append(test_acc_val)

    torch.save(model.state_dict(), f"{config.name}.pth")
    misclassified = test_misc_images
    model_results['TrainLoss'] = train_loss
    model_results['TestLoss'] = test_loss
    model_results['TrainAcc'] = train_acc
    model_results['TestAcc'] = test_acc
    model_results['LR'] = lr_list

    return model_results, test_misc_images