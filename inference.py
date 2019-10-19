import train
import os

import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt


def inference():
    os.makedirs('./results', exist_ok=True)
    model: nn.Module = train.get_model().cpu()
    sigmoid: nn.Sigmoid = nn.Sigmoid()
    model.load_state_dict(th.load('model.pth', map_location='cpu'))
    model.eval()

    dataloader, dataset = train.get_data_loader('validation/00-damage')
    with th.no_grad():
        for i in range(4):
            plt.figure(figsize=(32, 32))
            for j in range(25):
                x, y, path = dataset[i * 25 + j]
                y_hat = sigmoid(model(x.unsqueeze(0)))

                plt.subplot(5, 5, j+1)
                plt.imshow(plt.imread(path))
                plt.title(fr'$\hat{{y}} = {y_hat.item() * 100:2.2f}$ {path}')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'results/results_{i:2d}.jpeg')
            plt.close()


if __name__ == '__main__':
    inference()
