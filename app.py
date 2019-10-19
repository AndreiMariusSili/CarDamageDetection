import pathlib as pth
import sys

import torchvision as tv
import torch.nn as nn
import torch as th
from PIL import Image

import train


def main(path: str):
    path = pth.Path('./data/validation/00-damage/') / path
    model: nn.Module = train.get_model().cpu()
    sigmoid: nn.Sigmoid = nn.Sigmoid()
    model.load_state_dict(th.load('model.pth', map_location='cpu'))
    model.eval()

    transforms = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor()
    ])

    img = Image.open(path)
    x = transforms(img)
    y = sigmoid(model(x.unsqueeze(0)))

    return y.item()


if __name__ == '__main__':
    pred = main(sys.argv[1])
    print(f'{pred * 100:2.2f}%')
