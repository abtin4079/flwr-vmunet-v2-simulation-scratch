import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def training_process(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    local_epochs, 
                    step):
    '''
    train model for one epoch
    '''

    for _ in range(local_epochs):
    # switch to train mode
        model.train() 
    
        loss_list = []

        for iter, data in enumerate(train_loader):
            step += iter
            optimizer.zero_grad()
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            out = model(images)
            loss = criterion(out, targets)

            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())

            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            if (iter) % 10 == 0:
                log_info = f'train: epoch {local_epochs}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                print(log_info)
        scheduler.step() 
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"Epoch {local_epochs + 1}/{local_epochs} completed. Learning rate updated to {now_lr}")

            # Write the updated learning rate to a text file
    with open('learning_rate.txt', 'w') as f:
        f.write(f"{now_lr}\n")

    return step 