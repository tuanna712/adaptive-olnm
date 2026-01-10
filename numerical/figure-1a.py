import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
np.random.seed(42)
n, d = 10000, 100
A = np.random.randn(n, d)
true_x = np.random.randn(d)
y = A @ true_x + 0.1 * np.random.randn(n)
x_star, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

def compute_x_star(A_batch, y_batch):
    x_star, _, _, _ = np.linalg.lstsq(A_batch, y_batch, rcond=None)
    return x_star

batch_sizes = np.arange(200, 8000, step=100)
avg_diffs = []
avg_diffs_w_true = []

for b in batch_sizes:
    k = n // b
    diffs = []
    diffs_w_true = []

    perm = np.random.permutation(n)
    A_shuf, y_shuf = A[perm], y[perm]
    
    x_prev = compute_x_star(A_shuf[:b], y_shuf[:b])
    diffs_w_true.append(np.linalg.norm(x_prev - x_star))

    for t in range(1, k):
        A_batch = A_shuf[t*b:(t+1)*b]
        y_batch = y_shuf[t*b:(t+1)*b]
        x_curr = compute_x_star(A_batch, y_batch)
        
        diffs.append(np.linalg.norm(x_curr - x_prev))
        x_prev = x_curr
    
    avg_diffs.append(np.mean(diffs))
    avg_diffs_w_true.append(np.mean(diffs_w_true))


plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Helvetica'

plt.loglog(batch_sizes, avg_diffs_w_true, color='blue', 
           linewidth=2.5, 
           linestyle='dotted',
           label="Distance to true minimizer",
            markevery=1e-1,marker='*')
           
plt.loglog(batch_sizes, avg_diffs, color='green', 
           linewidth=2.5, 
           linestyle='dashed',
           label="Distance between consecutive minimizers",
            markevery=1e-1,marker='v')
           
plt.loglog(batch_sizes, 2/np.sqrt(batch_sizes), color='red', 
           linewidth=2.5, 
           linestyle='dashdot',
           label=r"$O(1/\sqrt{b})$",
            markevery=1e-1,marker='8')
           
plt.xlabel("Batch-size", fontsize=16)
plt.xticks(fontsize=16)

plt.ylabel("Distance (norm)", fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, which='both', ls='--', alpha=0.7)
plt.legend(loc='lower left', fontsize=16)

plt.savefig("imgs/figure-1a.png")
# plt.show()