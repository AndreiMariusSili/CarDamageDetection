
import torchvision.models as models
import torch.nn as nn

res_net = models.resnet50(pretrained=True)
res_net.fc = nn.Linear(512 * 4, 1)

