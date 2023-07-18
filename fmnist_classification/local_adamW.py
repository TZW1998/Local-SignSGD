import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import argparse
from tensorboardX import SummaryWriter
import os
import shutil
from copy import deepcopy

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command line arguments
parser = argparse.ArgumentParser(description='FMNIST Classification')
parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=0.01, help='Local Learning rate for sgd')
parser.add_argument('--num_nodes', type=int, default=1, help=' Number of computing nodes')
parser.add_argument('--local_steps', type=int, default=50, help='Number of local steps')
parser.add_argument('--global_lr', type=float, default=1, help='Global Learning rate')
parser.add_argument('--global_beta1', type=float, default=0.9, help='Global beta1 for Adam')
parser.add_argument('--global_beta2', type=float, default=0.999, help='Global beta2 for Adam')
parser.add_argument('--global_weight_decay', type=float, default=0, help='Global Weight decay')
parser.add_argument('--total_steps', type=int, default=5000, help='Total number of gradient steps')
parser.add_argument('--exp_index', type=int, default=0, help='Experiment index')
parser.add_argument('--log_step', type=int, default=50, help='log steps')
args = parser.parse_args()

# Load FMNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())

# Create data loaders
batch_size = args.batchsize
global_lr = args.global_lr
global_betas = (args.global_beta1, args.global_beta2)
global_weight_decay = args.global_weight_decay


local_batch_size = batch_size // args.num_nodes
train_loader = DataLoader(train_dataset, batch_size = local_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)

# Create model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()


# Training loop
gradient_steps = 0
running_loss = 0.0
total = 0

log_dir = 'logs/fmnist_classification/localadamw_exp{}_bz{}_lr{}_glr{}_betas({},{})_wd{}_nn{}_ls{}'.format(args.exp_index, batch_size, args.lr, global_lr, global_betas[0], global_betas[1], global_weight_decay, args.num_nodes, args.local_steps)
if os.path.exists(log_dir):
    clean_dir(log_dir)
logger = SummaryWriter(logdir= log_dir)
print('Start training with Local-AdamW algorithm, batch size {}, local learning rate {}, global learning rate {}, global betas ({},{}), global weight decay {}, num nodes {}, local steps {}.'.format(batch_size, args.lr, global_lr, global_betas[0], global_betas[1], global_weight_decay, args.num_nodes, args.local_steps))


global_weight = deepcopy(model.state_dict()) # be careful if batchnorm is used
global_state = {"momentum" : {n:torch.zeros_like(p) for n,p in  global_weight.items()},
                "stepsize_scaling" : {n:torch.zeros_like(p) for n,p in  global_weight.items()}
                }
while gradient_steps < args.total_steps:
    # Local SGD
    aggregated_pseudo_gradients = {n: torch.zeros_like(p) for n, p in global_weight.items()}
    for node in range(args.num_nodes):
        model.load_state_dict(global_weight)
        local_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
        num_local_steps = 0
        while num_local_steps < args.local_steps:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Gradient accumulation if local_batch_size is larger than 1024
                if local_batch_size < 1024:
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward 
                    loss.backward()

                    # Compute running loss
                    total += labels.size(0)
                    running_loss += loss.item() * labels.size(0)

                else:
                    if (local_batch_size % 1024) == 0:
                        num_accumulation_steps = (local_batch_size // 1024)  
                    else: 
                        num_accumulation_steps = (local_batch_size // 1024)   + 1
            
                    for now_steps in range(num_accumulation_steps):
                        if now_steps == (num_accumulation_steps - 1):
                            input_var = images[(now_steps * 1024):].to(device)
                            target_var = labels[(now_steps * 1024):].to(device)
                        else:
                            input_var = images[(now_steps * 1024): ((now_steps + 1) * 1024)].to(device)
                            target_var = labels[(now_steps * 1024): ((now_steps + 1) * 1024)].to(device)
                        
                        output = model(input_var)
                        loss = criterion(output, target_var) * len(input_var) / local_batch_size
                        loss.backward()

                        # Compute running loss
                        total += target_var.size(0)
                        running_loss += loss.item() * target_var.size(0)

                local_optimizer.step()
                local_optimizer.zero_grad()
                num_local_steps += 1

                if num_local_steps == args.local_steps:
                    break
        
        local_weight = model.state_dict()
        for n, p in aggregated_pseudo_gradients.items():
            aggregated_pseudo_gradients[n] += global_weight[n] - local_weight[n]
        
    gradient_steps += args.local_steps
    # update global_weight and global_state, for different algorithm, you only need to update here
    for n, p in global_weight.items():
        # update momentum
        global_state["momentum"][n].mul_(global_betas[0]).add_(aggregated_pseudo_gradients[n] / args.num_nodes,alpha= 1 - global_betas[0])

        # update stepsize_scaling        
        global_state["stepsize_scaling"][n].mul_(global_betas[1]).add_(torch.pow(aggregated_pseudo_gradients[n] / args.num_nodes, 2), alpha= 1 - global_betas[1])

        # weight_deacy
        d_p = global_state["momentum"][n] / (global_state["stepsize_scaling"][n].sqrt() + 1e-8)  + global_weight_decay * p

        # update global weight
        global_weight[n].add_(d_p, alpha=-global_lr)
    

    # Print loss and accuracy every communication round
    if (args.log_step is None) or (gradient_steps % args.log_step == 0):
        model.load_state_dict(global_weight)
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # Print test accuracy
        print(''.format())
        model.train()

        print('Gradient Steps: {}, running Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(gradient_steps, running_loss / total,
                                                                            (test_correct / test_total) * 100))
        logger.add_scalar('running_train_loss', running_loss / total, gradient_steps)
        logger.add_scalar('test_accuracy', (test_correct / test_total) * 100, gradient_steps)



