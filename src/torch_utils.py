import platform
import random

import numpy as np
import torch

from utils import is_env_notebook

if is_env_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def torch_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def check_gpu():
    if platform.system() == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def fit(model, optimizer, criterion, epochs, loader_dict, device, history, save_model_weights=True):
    best_val_loss = np.inf
    for epoch in tqdm(range(epochs)):
        epoch_loss = {'train': 0.0, 'val': 0.0}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in loader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs,  = model(inputs)
                    loss = criterion(labels, outputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # lossの計算
                    epoch_loss[phase] += loss.item() * inputs.size(0)

            # 1 epochでの損失
            epoch_loss[phase] = epoch_loss[phase] / len(loader_dict[phase].dataset)

        history.append(epoch_loss)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f"train loss: {epoch_loss['train']}")
        print(f"val loss: {epoch_loss['val']}")
        print('-'*30)

        if save_model_weights:
            if epoch_loss['val'] < best_val_loss:
                print(f"val loss improved from {best_val_loss} to {epoch_loss['val']}")
                print('-'*30)
                best_val_loss = epoch_loss['val']
                torch.save(model.state_dict(), 'model_weight.pth')
                # TODO: self.paramsを変更しないと上手く動かない
                # model_script = torch.jit.script(net)
                # model_script.save('model_script.pth')

    return history
