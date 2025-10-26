import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time       # MODIFIED - included to measure per-epoch elapsed time

"""
Train an MNIST model
Author: Tai Wan Kim
Date: November, 2025

Code cloned from: https://github.com/pytorch/examples/tree/main/mnist
Modifed for containerization and performance modeling.
Modified segments are marked MODIFIED below.
"""

VERBOSE = False   # MODIFIED - added flag to turn off verbose output

# ===== MODIFIED =====
# Replace torch.accelerator (not a public PyTorch API and not available in some environments)
def get_device(no_accel):
    if not no_accel and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
# ===== MODIFIED =====

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    # ===== MODIFIED =====
    # Modified to return per-epoch train loss and accuracy
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        running_loss += loss.item() * bs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += bs

        if VERBOSE and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        if args.dry_run:
            break
        
    avg_loss = running_loss / total
    acc_pct = 100.0 * correct / total
    return avg_loss, acc_pct
    # ===== MODIFIED =====

def test(model, device, test_loader):
    
    # ===== MODIFIED =====
    # Modified to return per-epoch validation loss and accuracy
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    acc_pct = 100.0 * correct / total
    return avg_loss, acc_pct
    # ===== MODIFIED =====

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ===== MODIFIED =====
    # Replace torch.accelerator (not a public PyTorch API and not available in some environments)
    device = get_device(args.no_accel)

    # Modified to disable shuffling in the testset
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs  = {'batch_size': args.test_batch_size, 'shuffle': False}
    
    if device.type == "cuda":
        common_loader = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    else:  # cpu
        common_loader = {'num_workers': 2}
    
    train_kwargs.update(common_loader)
    test_kwargs.update(common_loader)
    # ===== MODIFIED =====
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ===== MODIFIED =====
    # Modified to measure per-epoch and total elapsed time
    # Print out per-epoch summary, and final summary at the end
    total_start = time.perf_counter()
    last_train, last_val = (None, None), (None, None)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = test(model, device, test_loader)
        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        last_train, last_val = (train_loss, train_acc), (val_loss, val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
              f"epoch_time={epoch_time:.2f}s")

    total_time = time.perf_counter() - total_start

    final_train_loss, final_train_acc = last_train
    final_val_loss,   final_val_acc   = last_val

    print("\nFinal results:")
    print(f"  train_loss={final_train_loss:.4f} train_acc={final_train_acc:.2f}%")
    print(f"  val_loss={final_val_loss:.4f} val_acc={final_val_acc:.2f}%")
    print(f"  total_time={total_time:.2f}s for {args.epochs} epochs")
    # ===== MODIFIED =====
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
