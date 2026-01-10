import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from olnm import OLNM, MovingAverageDetector, FTestDetector

# --- Hyperparameters ---
INPUT_SIZE = 28 * 28 
NUM_CLASSES = 10
BATCH_SIZE = 500
EPOCHS = 5 
SGD_LR = 2.0
ADAM_LR = 0.005
OLNM_LR = 1.0
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

# --- MSE Loss Function ---
criterion = nn.MSELoss()

# --- Training and Evaluation Function ---
def train_and_evaluate(optimizer_name, model, optimizer):
    print(f"--- Training with {optimizer_name} using MSE Loss ---")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            def closure(grad=True):
                logits = model(data)
                output_probs = torch.softmax(logits, dim=1)
                target_one_hot = nn.functional.one_hot(target, num_classes=NUM_CLASSES).float()
                loss = criterion(output_probs, target_one_hot)

                if grad:
                    optimizer.zero_grad()
                    loss.backward()
                return loss

            if isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
                loss = closure() # compute loss and gradients
                optimizer.step() # update weights
            else:
                loss = optimizer.step(closure)

            loss_history.append(loss.item())
            
            if (batch_idx + 1) % 200 == 0:
                print(f"Optimizer: {optimizer_name}, Epoch [{epoch+1}/{EPOCHS}], "
                      f"Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f"Final Accuracy for {optimizer_name}: {100 * correct / total:.2f}%\n")
    return loss_history


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
model_olnm = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
optimizer_olnm = OLNM(model_olnm.parameters(),
                              lr=OLNM_LR,
                              c=C,
                              batch_size=BATCH_SIZE)

# 4. Adaptive OLNM - Moving Average
torch.manual_seed(42)
model_olnm_ma = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
MA_Detector = MovingAverageDetector(window_size=68, c=1.5)
optimizer_olnm_ma = OLNM(
    model_olnm_ma.parameters(), 
    lr=OLNM_LR, 
    c=C, 
    batch_size=BATCH_SIZE,
    adaptive=True,
    error_detector=MA_Detector
)

# 5. Adaptive OLNM - F-Test
torch.manual_seed(42)
model_olnm_ft = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
FT_Detector = FTestDetector(alpha=0.5)
optimizer_olnm_ft = OLNM(
    model_olnm_ft.parameters(), 
    lr=OLNM_LR, 
    c=C, 
    batch_size=BATCH_SIZE,
    adaptive=True,
    error_detector=FT_Detector
)

# --- Run Training and Collect History ---
loss_sgd = train_and_evaluate("SGD", model_sgd, optimizer_sgd)
loss_adam = train_and_evaluate("Adam", model_adam, optimizer_adam)
loss_olnm = train_and_evaluate("OLNM", model_olnm, optimizer_olnm)
loss_olnm_ma = train_and_evaluate("Adaptive OLNM (MA)", model_olnm_ma, optimizer_olnm_ma)
loss_olnm_ft = train_and_evaluate("Adaptive OLNM (FT)", model_olnm_ft, optimizer_olnm_ft)

# --- Comparison Plot ---
import numpy as np
def smooth_history(history, window_size=200):
    return np.convolve(history, np.ones(window_size)/window_size, mode='valid')
plt.close('all')
plt.figure(figsize=(8, 6))
plt.plot(smooth_history(loss_sgd), alpha=0.8, color='blue',
        linewidth=2.5, 
        linestyle='dotted',
        label='Default SGD (LR=2.0)', 
        markevery=50, marker='.')
plt.plot(smooth_history(loss_adam), alpha=0.8, color='red',
        linewidth=2.5, 
        linestyle='dashdot',
        label='Adam (LR=0.005)',
        markevery=50, marker='v')
plt.plot(smooth_history(loss_olnm), alpha=0.8, color='orange',
        linewidth=2.5, 
        linestyle='dashed',
        label=f'Default OLNM (LR=1.0)', 
        markevery=50, marker='8')
plt.plot(smooth_history(loss_olnm_ft), alpha=0.8, color='green',
        linewidth=2.5, 
        linestyle='-',
        label=f'FT Adaptive OLNM (LR=1.0)',    
        markevery=50, marker='*')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('MSE Loss', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0.011, 0.015)
plt.legend(loc='upper right', fontsize=16)
plt.grid(True, alpha=0.5)
plt.savefig("imgs/figure-2a.png")
# plt.show()