import torchvision.models as models
import torch.utils.data as thd
import torch.optim as optim
import torch.nn as nn
import torch as th

import dataset as ds


def get_model():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    res_net = models.resnet50(pretrained=True)
    res_net.fc = nn.Linear(512 * 4, 1)

    return res_net.to(device)


def get_data_loader(subset: str):
    dataset = ds.CarDamageDataset(subset)

    return thd.DataLoader(dataset, 64, shuffle=True, num_workers=12, pin_memory=False), dataset


def main():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    data_loader, dataset = get_data_loader('training/*')
    model = get_model()
    sigmoid = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1):
        epoch_acc = 0
        for i, (x, y, _) in enumerate(data_loader):
            b = x.shape[0]
            y = y.reshape(b, 1)
            x, y = x.to(device), y.to(device)

            e_hat = model(x)
            loss = criterion(e_hat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with th.no_grad():
                # noinspection PyUnresolvedReferences
                acc = (sigmoid(e_hat).round() == y).sum().to(th.float)
                epoch_acc += acc
                print(f'[Epoch: {epoch:03d}][Batch: {i:03d}][BCE LOSS: {loss.item():.4f}][Accuracy@1: {acc / b:.4f}]')
                th.save(model.state_dict(), 'model.pth')
        print(f'[Epoch Accuracy@1: {epoch_acc / len(dataset):.4f}]')
    th.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
