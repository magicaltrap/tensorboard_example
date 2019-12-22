import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from collections import OrderedDict

from RunManager import RunBuilder, RunManager


# 1) Prepare the data

train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, 
                                              transform=transforms.Compose([transforms.ToTensor()]))


# 2) Build the model

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # For the first conv layer, we have 1 color channel that should be convolved 
        # by 6 filters of size 5x5 to produce 6 output channels.
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        # Now, the second conv layer has 12 filters, and instead of convolving a single 
        # input channel, there are 6 input channels coming from the previous layer.
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
        
    def forward(self, t):
        
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 12 * 4 * 4)
        t = F.relu(self.fc1(t))
        
        t = F.relu(self.fc2(t))
        
        t = self.out(t)
        
        return t


# 3) Train the model
# 4) Calculate the loss, the gradient, and update the weights

params = OrderedDict(
    lr = [0.01, 0.005, 0.001],
    batch_size = [100, 1000],
    num_workers = [0, 1]
    #shuffle = [True, False]
)

m = RunManager()

for run in RunBuilder.get_runs(params):
    
    network = Network()
    loader = DataLoader(train_set, batch_size=run.batch_size, num_workers=run.num_workers)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(3):
        m.begin_epoch()
        
        for batch in loader:
            
            images, labels = batch
            preds = network(images) #Pass batch into network
            loss = F.cross_entropy(preds, labels) #Calculate loss
            optimizer.zero_grad() # Zero gradients
            loss.backward() # Calculate gradients
            optimizer.step() # Update weights
            
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
            
        m.end_epoch()
    m.end_run()
m.save('results')