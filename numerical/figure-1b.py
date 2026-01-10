import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from olnm_np import SGD_Optimizer, OLNM_Optimizer, get_experiment_data

N = 10000       
d = 100         
num_steps = 500

def find_condition_numbers(N, d, L, kappa):
    A_full, _ = get_experiment_data(N, d, L, kappa, seed=42)
    mList = np.linspace( d, N, num=10, dtype=int)
    cond_numbers = []
    for m in mList:
        nReps = 20
        cond = 0.
        for _ in range(nReps):
            indices = np.random.choice(N, m, replace=False)
            A_mini = A_full[indices, :]
            cov_matrix = (A_mini.T @ A_mini)
            cond += np.linalg.cond(cov_matrix)
        cond /= nReps
        cond_numbers.append(cond)
    return mList, cond_numbers


def run(N, d, m, L, kappa, c=1900):
    A_full, _ = get_experiment_data(N, d, L, kappa, seed=42)
    x_star = np.random.randn(d) * 0.1
    b_full = A_full @ x_star + np.random.normal(0, 1e-3, N)

    olnm = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='root_decay', c=c)
    # olnm = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='root_scaled')

    eta_sgd = 0.003
    sgd = SGD_Optimizer(d=d, m=m, step_size=eta_sgd)


    olnm_error_history = [np.linalg.norm(olnm.x - x_star)]
    sgd_error_history = [np.linalg.norm(sgd.x - x_star)]

    for t in range(num_steps):
        # Sample a minibatch
        indices = np.random.choice(N, m, replace=True)
        A_t = A_full[indices, :]
        b_t = b_full[indices]
        
        # Run SGD
        x_sgd = sgd.step(A_t, b_t)
        sgd_error = np.linalg.norm(x_sgd - x_star)
        sgd_error_history.append(sgd_error)

        # Run OLNM
        x_olnm = olnm.step(A_t, b_t)
        olnm_error = np.linalg.norm(x_olnm - x_star)
        olnm_error_history.append(olnm_error)

    return olnm_error_history, sgd_error_history

bs = [200, 500, 1000, 2000]

olnm_m200, sgd_m200 = run(N=N, d=d, m=bs[0], L=500, kappa=500)
olnm_m500, sgd_m500 = run(N=N, d=d, m=bs[1], L=500, kappa=500)
olnm_m1000, sgd_m1000 = run(N=N, d=d, m=bs[2], L=500, kappa=500)
olnm_m2000, sgd_m2000 = run(N=N, d=d, m=bs[3], L=500, kappa=500)

colors = ['#DD0303', '#FA812F', '#E9A319', '#71C436', '#002947', '#432323']

plt.figure(figsize=(8, 6))
# OLNM (solid lines)
plt.loglog(np.arange(len(olnm_m200)) + 1, olnm_m200, label='OLNM w/ b=200', linewidth=2.5, 
    linestyle=':', color=colors[5], alpha=0.7,
    markevery=1e-1,marker='.')
plt.loglog(np.arange(len(olnm_m500)) + 1, olnm_m500, label='OLNM w/ b=500', linewidth=2.5, 
    linestyle='-.', color=colors[2], alpha=0.7,
    markevery=1e-1,marker='v')
plt.loglog(np.arange(len(olnm_m1000)) + 1, olnm_m1000, label='OLNM w/ b=1000', linewidth=2.5, 
    linestyle='--', color=colors[1], alpha=0.7,
    markevery=1e-1,marker='s')
plt.loglog(np.arange(len(olnm_m2000)) + 1, olnm_m2000, label='OLNM w/ b=2000', linewidth=2.5, 
    linestyle='-', color=colors[0], alpha=0.7,
    markevery=1e-1,marker='*')

# SGD (dashed lines)
plt.loglog(np.arange(len(sgd_m200)) + 1, sgd_m200, label='Default SGD', linewidth=2.5, 
    linestyle=(0, (3, 1, 1, 1, 1, 1)), color='blue', alpha=0.8)

plt.rcParams['font.family'] = 'Helvetica'
plt.xlabel('Mini-Batch Step ($t$)', fontsize=16)
plt.ylabel('Tracking Error', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='lower left', fontsize=16)
plt.savefig("imgs/figure-1b.png")

# plt.show()