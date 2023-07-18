import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional
import argparse
from tensorboardX import SummaryWriter
import os
import shutil

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

class SignSGD(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.99),
                 weight_decay=0, *,
                 differentiable=False):
        
        momentum = betas[0]
        momentum_interp = betas[1]

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum_interp < 0.0:
            raise ValueError("Invalid momentum_interp value: {}".format(momentum_interp))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, momentum_interp=momentum_interp,
                        weight_decay=weight_decay, differentiable=differentiable)

        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            error_residuals_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self._update_params(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                momentum_interp=group['momentum_interp'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def _update_params(self, params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            momentum_interp: float,
            lr: float,
            ):

        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if momentum > 1e-8:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    grad = buf.mul(momentum).add(d_p, alpha=1-momentum)
                    buf.mul_(momentum_interp).add_(d_p, alpha=1-momentum_interp)
                    d_p = grad
            
            # decouple sign and weight decay
            d_p.sign_()
            if weight_decay != 0:
                d_p.add_(param, alpha=weight_decay)

            # if noise_scale > 1e-8:
            #     d_p.add_(torch.randn_like(d_p), alpha = noise_scale)

            
            param.add_(d_p, alpha=-lr)

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
parser.add_argument('--batchsize', type=int, default=1024, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for Adam')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--total_steps', type=int, default=5000, help='Total number of gradient steps')
parser.add_argument('--exp_index', type=int, default=0, help='Experiment index')
parser.add_argument('--log_step', type=int, default=50, help='log steps')
args = parser.parse_args()

# Load FMNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())

# Create data loaders
batch_size = args.batchsize
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)

# Create model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SignSGD(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

# Training loop
gradient_steps = 0
running_loss = 0.0
total = 0

log_dir = 'logs/fmnist_classification/signsgd_exp{}_bz{}_lr{}_b1{}_b2{}_wd{}'.format(args.exp_index, args.batchsize, args.lr, args.beta1, args.beta2, args.weight_decay)
if os.path.exists(log_dir):
    clean_dir(log_dir)
logger = SummaryWriter(logdir= log_dir)
print('Start training with SignSGD algorithm, batch size {}, learning rate {}, betas ({},{}), weight decay {}.'.format(args.batchsize, args.lr, args.beta1, args.beta2, args.weight_decay))

while gradient_steps < args.total_steps:
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Gradient accumulation if batch_size is larger than 1024
        if batch_size < 1024:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward 
            loss.backward()

            # Compute running loss
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

        else:
            if (batch_size % 1024) == 0:
                num_accumulation_steps = (batch_size // 1024)  
            else: 
                num_accumulation_steps = (batch_size // 1024)   + 1
     
            for now_steps in range(num_accumulation_steps):
                if now_steps == (num_accumulation_steps - 1):
                    input_var = images[(now_steps * 1024):].to(device)
                    target_var = labels[(now_steps * 1024):].to(device)
                else:
                    input_var = images[(now_steps * 1024): ((now_steps + 1) * 1024)].to(device)
                    target_var = labels[(now_steps * 1024): ((now_steps + 1) * 1024)].to(device)
                
                output = model(input_var)
                loss = criterion(output, target_var) * len(input_var) / batch_size
                loss.backward()

                # Compute running loss
                total += target_var.size(0)
                running_loss += loss.item() * target_var.size(0)

        optimizer.step()
        optimizer.zero_grad()
        gradient_steps += 1

        # Print loss and accuracy every log_step gradient steps
        if (gradient_steps % args.log_step == 0) or (gradient_steps == args.total_steps):
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


        if gradient_steps == args.total_steps:
            break

