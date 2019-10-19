import torch.utils.data as thd
import torchvision as tv
import torch as th
import glob

from PIL import Image


class CarDamageDataset(thd.Dataset):
    def __init__(self):
        self.files = list(sorted(glob.glob('./data/*/*/*.JPEG')))

        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_path = self.files[item]
        x = Image.open(file_path)
        x = self.transforms(x)

        if 'damage' in file_path:
            y = th.tensor(1, dtype=th.float)
        else:
            y = th.tensor(0, dtype=th.float)

        return x, y
