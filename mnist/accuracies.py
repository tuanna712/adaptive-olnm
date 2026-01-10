import torch, time
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from olnm import OLNM, FTestDetector, MovingAverageDetector

# --- Hyperparameters ---
INPUT_SIZE = 28 * 28 
NUM_CLASSES = 10
BATCH_SIZE = 500
EPOCHS = 5 
SGD_LR = 0.1
ADAM_LR = 0.005
OLNM_LR = 0.05
C = 1789  # Tuning parameter for OLNM

# --- MNIST Dataset Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definition ---
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# --- Training and Evaluation Function ---
def _evaluate(model, criterion, test_loader):
    model.eval() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _test_acc, _test_err, _test_loss, total_test = 0, 0, 0, 0
    with torch.no_grad(): 
        for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                _test_acc += (predicted == labels).sum().item()
                _test_err += (predicted != labels).sum().item()
                _test_loss += criterion(outputs, labels).item()

    test_loss = _test_loss / total_test
    test_err = 100 * _test_err / total_test
    test_acc = 100 * _test_acc / total_test

    return test_loss, test_err, test_acc

def train_and_evaluate(model, optimizer, train_loader, test_loader,  EPOCHS, early_stop=False, threshold=90.0):
    criterion = nn.CrossEntropyLoss() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training with {optimizer.__class__.__name__} using Cross Entropy Loss ---")
    train_losses, test_losses, train_errs, test_errs, train_accs, test_accs, run_times = [], [], [], [], [], [], []
    model.to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        total_train, _train_err, _train_acc, running_loss, run_time = 0, 0, 0, 0.0, 0
        _start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            def closure(grad=True):
                output = model(data)
                loss = criterion(output, target)

                if grad:
                    optimizer.zero_grad()
                    loss.backward()
                return loss

            if isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
                loss = closure() # compute loss and gradients
                optimizer.step() # update weights
            else:
                loss = optimizer.step(closure)

            running_loss += loss.item()
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            total_train += target.size(0)
            _train_err += (predicted != target).sum().item()
            _train_acc += (predicted == target).sum().item()

            run_time = time.time() - _start
            
            epoch_train_loss = running_loss / total_train
            epoch_train_acc = 100 * _train_acc / total_train
            epoch_train_err = 100 * _train_err / total_train

            test_loss, test_err, test_acc = _evaluate(model, criterion, test_loader)

            train_losses.append(epoch_train_loss)
            train_errs.append(epoch_train_err)
            train_accs.append(epoch_train_acc)
            test_losses.append(test_loss)
            test_errs.append(test_err)
            test_accs.append(test_acc)
            run_times.append(run_time)

        if epoch % 1 == 0:
            print(f'E [{epoch+1}/{EPOCHS}]. train_loss_acc: {running_loss / len(train_loader):.4f}, {epoch_train_acc:.2f}%, '
                    f'test_acc: {test_acc:.2f}%, run_time: {run_time}')
        if early_stop and epoch_train_acc >= threshold:
            print(f"Early stopping at epoch {epoch+1} with train error {epoch_train_err:.2f}%")
            break
    return train_losses, test_losses, train_errs, test_errs, train_accs, test_accs, run_times

# --- Model and Optimizer Instantiation ---
# 1. SGD
torch.manual_seed(42)
model_sgd = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=SGD_LR)

# 2. Adam
torch.manual_seed(42)
model_adam = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=ADAM_LR)

# 3. ORIGINAL_OLNM
torch.manual_seed(42)
from olnm import OLNM
model_olnm = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
optimizer_olnm = OLNM(model_olnm.parameters(),
                              lr=OLNM_LR,
                              c=C,
                              batch_size=BATCH_SIZE)
# 4. ADAPTIVE MA
torch.manual_seed(42)
model_ma_olnm = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
detector_ma = MovingAverageDetector(window_size=68, c=1.5)
optimizer_ma_olnm = OLNM(
    model_ma_olnm.parameters(), 
    lr=OLNM_LR, 
    c=C, 
    batch_size=BATCH_SIZE,
    adaptive=True,
    error_detector=detector_ma
)

# 5. ADAPTIVE FT
torch.manual_seed(42)
model_ft_olnm = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
detector_ft = FTestDetector(alpha=0.5)
optimizer_ft_olnm = OLNM(
    model_ft_olnm.parameters(), 
    lr=OLNM_LR, 
    c=C, 
    batch_size=BATCH_SIZE,
    adaptive=True,
    error_detector=detector_ft
)

# --- Run Training and Collect History ---
train_losses_sgd, test_losses_sgd, train_errs_sgd, test_errs_sgd, train_accs_sgd, test_accs_sgd, run_times_sgd = train_and_evaluate(model_sgd, optimizer_sgd, 
                              train_loader, test_loader,  EPOCHS)

train_losses_adam, test_losses_adam, train_errs_adam, test_errs_adam, train_accs_adam, test_accs_adam, run_times_adam = train_and_evaluate(model_adam, optimizer_adam, 
                               train_loader, test_loader,  EPOCHS)

train_losses_olnm, test_losses_olnm, train_errs_olnm, test_errs_olnm, train_accs_olnm, test_accs_olnm, run_times_olnm = train_and_evaluate(model_olnm, optimizer_olnm, 
                                  train_loader, test_loader,  EPOCHS)

train_losses_ma_olnm, test_losses_ma_olnm, train_errs_ma_olnm, test_errs_ma_olnm, train_accs_ma_olnm, test_accs_ma_olnm, run_times_ma_olnm = train_and_evaluate(model_ma_olnm, optimizer_ma_olnm, 
                                  train_loader, test_loader,  EPOCHS)

train_losses_ft_olnm, test_losses_ft_olnm, train_errs_ft_olnm, test_errs_ft_olnm, train_accs_ft_olnm, test_accs_ft_olnm, run_times_ft_olnm = train_and_evaluate(model_ft_olnm, optimizer_ft_olnm, 
                                  train_loader, test_loader,  EPOCHS)

# --- Comparison Plot ---
import numpy as np  
def smooth_history(history, window_size=5):
    return np.convolve(history, np.ones(window_size)/window_size, mode='valid')

def plot_history(history, 
                 y_label, y_limit=None, 
                 loc='upper right',
                 filename='figure-name',
                 show=False, save=True,
                 ):
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.family'] = 'Helvetica'

        plt.plot(smooth_history(history['sgd']), alpha=0.8, color='blue',
                linewidth=2.5, 
                linestyle='dotted',
                label='Default SGD (LR=0.1)', 
                markevery=50, marker='.')
        plt.plot(smooth_history(history['adam']), alpha=0.8, color='red',
                linewidth=2.5, 
                linestyle='dashdot',
                label='Adam (LR=0.005)',
                markevery=50, marker='v')
        plt.plot(smooth_history(history['olnm']), alpha=0.8, color='orange',
                linewidth=2.5, 
                linestyle='dashed',
                label='Default OLNM (LR=0.05)', 
                markevery=50, marker='8')
        plt.plot(smooth_history(history['ft_olnm']), alpha=0.8, color='green',
                linewidth=2.5, 
                linestyle='-',
                label='Adaptive OLNM (LR=0.05)',    
                markevery=50, marker='*')

        plt.xlabel('Iteration', fontsize=16)
        plt.xticks(fontsize=16)

        plt.ylabel(y_label, fontsize=16)
        plt.yticks(fontsize=16)
        if y_limit:
            plt.ylim(y_limit)

        plt.legend(loc=loc, fontsize=16)
        plt.grid(True, alpha=0.5)
        if save:
            plt.savefig(f"imgs/{filename}.png")
        if show:
            plt.show()

# Train Accuracies
history = {'sgd': train_accs_sgd,
        'adam': train_accs_adam,
        'olnm': train_accs_olnm,
        'ft_olnm': train_accs_ft_olnm
        }
plot_history(history, y_label='Train Accuracy', 
             loc='lower right', y_limit=(85, 94.5),
             filename='figure-3a',
             )

# Train Losses
history = {'sgd': train_losses_sgd,
        'adam': train_losses_adam,
        'olnm': train_losses_olnm,
        'ft_olnm': train_losses_ft_olnm
        }
plot_history(history, y_label='Train LCE Loss', 
             y_limit=(0.0004, 0.0012),
             filename='figure-3b',
             )

# Train Errors
history = {'sgd': train_errs_sgd,
        'adam': train_errs_adam,
        'olnm': train_errs_olnm,
        'ft_olnm': train_errs_ft_olnm
        }
plot_history(history, y_label='Train LCE Error', 
             y_limit=(5, 12), 
             filename='figure-3c',
             )

# Test Accuracies
history = {'sgd': test_accs_sgd,
        'adam': test_accs_adam,
        'olnm': test_accs_olnm,
        'ft_olnm': test_accs_ft_olnm
        }
plot_history(history, y_label='Test Accuracy', loc='lower right', 
             y_limit=(90, 93),
             filename='figure-3d',
             )

# Test Losses
history = {'sgd': test_losses_sgd,
        'adam': test_losses_adam,
        'olnm': test_losses_olnm,
        'ft_olnm': test_losses_ft_olnm
        }
plot_history(history, y_label='Test LCE Loss', 
             filename='figure-3e',
             y_limit=(.0005, .0007))

# Test Errors
history = {'sgd': test_errs_sgd,
        'adam': test_errs_adam,
        'olnm': test_errs_olnm,
        'ft_olnm': test_errs_ft_olnm
        }
plot_history(history, y_label='Test LCE Error', 
             filename='figure-3f',
             y_limit=(7, 12))