from facenet_pytorch import InceptionResnetV1
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import dataset as DS
from torch.utils.data import DataLoader, Dataset
import config
import matplotlib.pyplot as plt
import seaborn as sns
class model_architecture(nn.Module):
    def __init__(self):
        super(model_architecture, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.Linear(512, 30))
    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x
def train(model, optimizer, train_data, criterion, device, epoch):
    loss = 0
    for index, (x, y) in enumerate(tqdm(train_data)):
        x = x.to(device)
        y = y.to(device)
        predict_ = model(x)
        loss_ = criterion(predict_[y!=-1], y[y!=-1])
        loss += loss_
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    return loss

def train_fn():
    train_dataset = DS.Dataset_(root='data/train/', csv_file="training/training.csv", train=True, transform=config.train_transforms)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = model_architecture()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # scheduler=optim.lr_scheduler.ExponentialLR()
    loss_arr = []
    for epoch in range(10):
        loss = train(model, optimizer, loader, criterion, device, epoch)
        loss_arr.append(loss)
        print(f'loss in epoch {epoch} is equal to {loss}')

    print(f'testing the method')
    with torch.no_grad():
        # visulize(loss_arr)
        test(model, device)

def visulize(loss_arr):
    losses_float = [float(loss.cpu().detach().numpy()) for loss in loss_arr]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')
    plt.savefig('train.png')
def test(model, device):
    train_dataset = DS.Dataset_(root='data/test/', csv_file="test/test.csv", train=False,
                                transform=None)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for _, (x, y) in enumerate(loader):
            x = x.to(device)
            y_ = model(x)
            plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')
            plt.plot(y_[0][0::2].detach().cpu().numpy(), y_[0][1::2].detach().cpu().numpy(), "go")
            plt.show()

if __name__=='__main__':
    train_fn()







