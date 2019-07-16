from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


my_transform = transforms.Compose([
    transforms.ToTensor()
])


def default_loader(img_path):
    return Image.open(img_path).convert('RGB')


class MyDataSet(Dataset):
    def __init__(self, img_path, txt_path, img_transform=my_transform, loader=default_loader):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [os.path.join(img_path, line.split()[0]) for line in lines]
            self.label_list =[line.split()[1] for line in lines]
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = int(self.label_list[index])
        img = self.loader(img_path)
        img = img.resize((512, 512), Image.ANTIALIAS)
        # img = np.array(self.loader(img_path))
        # to tensor
        # img = transforms.ToTensor()(img)
        label = torch.tensor(label)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10*6*6, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        output = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        output = F.relu(F.max_pool2d(self.conv_drop(self.conv2(output)), kernel_size=2))
        output = F.relu(F.max_pool2d(self.conv3(output), kernel_size=2))
        output = F.relu(F.max_pool2d(self.conv4(output), kernel_size=2))
        output = output.view(-1, 10*6*6)  # flat the dimension
        output = F.relu(self.fc1(output))
        output = F.dropout(output, training=self.training)  # dropout
        output = self.fc2(output)
        return F.log_softmax(output, dim=1)


def train(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_id, (X, Y) in enumerate(data_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = F.nll_loss(pred, Y)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_id+1) * len(X), len(data_loader.dataset),
            100.0 * (batch_id+1) / len(data_loader), loss.item()
        ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += F.nll_loss(output, Y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(Y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    train_img_path = '.\Train'
    train_txt_path = '.\Train\Label.TXT'
    train_data = MyDataSet(train_img_path, train_txt_path)
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

    test_img_path = '.\Test'
    test_txt_path = '.\Test\Label.txt'
    test_data = MyDataSet(test_img_path, test_txt_path)
    test_loader = DataLoader(test_data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

    for epoch in range(1, 101):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
