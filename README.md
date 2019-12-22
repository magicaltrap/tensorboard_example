# About
This is an implementation of [Deeplizard's TensorBoard with PyTorch - Visualize Deep Learning Metrics](https://youtu.be/pSexXMdruFM) to showcase TensorBoard. With TensorBoard, we can track, analyze and visualize our training models. It is especially useful for searching a good set of hyperparameters.

# Hyperparameter Search

Finding good hyperparameters is always a bit of a tedious task in machine learning. Let's say we have three different sets of hyperparameters we want to use for training our model:

```python
params = OrderedDict(
    lr = [0.01, 0.005, 0.001],
    batch_size = [100, 1000],
    num_workers = [0, 1]
)
```

We have 3 different learning rates [0.01, 0.005, 0.001], two batch sizes [100, 1000] and different numbers of num_workers (denotes the number of processes that generate batches in parallel). Is the set `(lr=0.01, batch_size=100, num_workers=0)` good? Or is for example the set `(lr=0.005, batch_size=1000, num_workers=1)` better? Taking the cartesian product of these sets equals to 12 different combination of hyperparameters. We could conduct 12 separate training sessions and compare their performances afterwards. Or we could include all 12 possible sets of hyperparameters in a single training loop and compare them in TensorBoard afterwards. Let's do the later.

# Setup Before Training

1. [Install TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) so we can use TensorBoard's SummaryWriter function:
```python
from torch.utils.tensorboard import SummaryWriter
```
2. `RunManager.py` helps us in multiple ways.

**to be continued**



- useful to outsource our training loop. 
- Runmanager.py has two classes: RunBuilder and RunManager. Useful for saving all kinds of information for the SummaryWriter (save informaton about loss, accuracy, details of our parameter etc). Also possible to save results in json or csv format. Saves data in our Summary writer

We initialie RunManager before starting training. 
```python
- m = RunManager()

for run in RunBuilder.get_runs(params):
    
    network = Network()
    loader = DataLoader(train_set, batch_size=run.batch_size, num_workers=run.num_workers)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(3):
        m.begin_epoch()
        
        ...
 ```
        
 # Training: Step-by-Step
 





