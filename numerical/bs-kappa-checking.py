import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from olnm_np import SGD_Optimizer, OLNM_Optimizer, get_experiment_data

def analyze_batch_size_impact(N, d, L, kappa, nsteps=100):
    # Setup Data
    A_full, _ = get_experiment_data(N, d, L, kappa, seed=42)
    x_star = np.random.randn(d) * 0.1
    b_full = A_full @ x_star + np.random.normal(0, 1e-3, N)
    # Batch sizes
    m_list = np.linspace(3*d, N/2, num=15, dtype=int)
    
    avg_cond_numbers, final_errors_olnm, final_errors_sgd = [], [], []

    print(f"{'Batch Size':>10} | {'Avg Condition #':>15} | {'Final SGD Err':>15}")
    print("-" * 50)

    for m in m_list:
        # --- Condition Number ---
        nReps = 20
        total_cond = 0.
        for _ in range(nReps):
            indices = np.random.choice(N, m, replace=False)
            A_mini = A_full[indices, :]
            # Covariance matrix: A^T * A
            cov_matrix = (A_mini.T @ A_mini)
            total_cond += np.linalg.cond(cov_matrix)
        
        avg_cond = total_cond / nReps
        avg_cond_numbers.append(avg_cond)

        # --- Training ---
        olnm = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='root_scaled')
        sgd = SGD_Optimizer(d=d, m=m, step_size=1 / L * 0.5)

        for t in range(nsteps):
            indices = np.random.choice(N, m, replace=True)
            A_t, b_t = A_full[indices, :], b_full[indices]
            
            x_sgd = sgd.step(A_t, b_t)
            x_olnm = olnm.step(A_t, b_t)

        # Final error
        final_errors_sgd.append(np.linalg.norm(x_sgd - x_star))
        final_errors_olnm.append(np.linalg.norm(x_olnm - x_star))
        
        print(f"{m:10d} | {avg_cond:15.2f} | {final_errors_sgd[-1]:15.6f}")
    return m_list, avg_cond_numbers, final_errors_olnm, final_errors_sgd

def plot_results(m_list, avg_cond_numbers, d):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Batch Size (m)', fontsize=16)
    ax1.set_ylabel('Condition Number', color=color, fontsize=16)
    ax1.plot(m_list, avg_cond_numbers, marker='o', linewidth=2.5, linestyle='-', alpha=0.7, color=color, label='Condition Number')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('$1/\sqrt{m}$', color=color, fontsize=16)
    ref_line = [1/np.sqrt(m) for m in m_list]
    ax2.plot(m_list, ref_line,  linewidth=2.5, linestyle='--', alpha=0.7, color='red', label='$1/\sqrt{m}$')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)

    plt.title(f'Batch Size vs. Matrix Conditioning and Convergence (d={d})')
    plt.legend(loc='upper right', fontsize=16)
    fig.tight_layout()

    plt.savefig("imgs/figure-x1.png")
    # plt.show()

N, d, nsteps = 10000, 100, 500
kappa, L = 500, 500

m_list, avg_cond_numbers, final_errors_olnm, final_errors_sgd = analyze_batch_size_impact(N, d, L, kappa, nsteps=nsteps)
plot_results(m_list, avg_cond_numbers, d)

def run(N, d, m, L, kappa, Epochs=10, c=1):
    A_full, _ = get_experiment_data(N, d, L, kappa, seed=42)
    x_star = np.random.randn(d) * 0.1
    b_full = A_full @ x_star + np.random.normal(0, 1e-3, N)

    olnm_1 = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='root_scaled', c=1)
    olnm_2 = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='constant', c=30)
    olnm_3 = OLNM_Optimizer(d=d, L=L, kappa=kappa, m=m, method='root_decay', c=500)

    eta_sgd = 1 / L * 0.5
    sgd = SGD_Optimizer(d=d, m=m, step_size=eta_sgd)


    olnm_1_error_history = [np.linalg.norm(olnm_1.x - x_star)]
    olnm_2_error_history = [np.linalg.norm(olnm_2.x - x_star)]
    olnm_3_error_history = [np.linalg.norm(olnm_3.x - x_star)]
    sgd_error_history = [np.linalg.norm(sgd.x - x_star)]

    for epoch in range(Epochs):
        if epoch % 200 == 0:
            print(f"Epoch {epoch + 1}/{Epochs}")
        perm = np.random.permutation(N)
        olnm_1_errors, olnm_2_errors, olnm_3_errors, sgd_errors = [], [], [], []

        for i in range(0, N, m):
            batch_indices = perm[i:i + m]

            A_t = A_full[batch_indices, :]
            b_t = b_full[batch_indices]
        
            # Run SGD
            x_sgd = sgd.step(A_t, b_t)
            sgd_error = np.linalg.norm(x_sgd - x_star)
            sgd_errors.append(sgd_error)

            # Run OLNM Root Scaled
            x_olnm = olnm_1.step(A_t, b_t)
            olnm_1_error = np.linalg.norm(x_olnm - x_star)
            olnm_1_errors.append(olnm_1_error)

            # Run OLNM Constant
            x_olnm = olnm_2.step(A_t, b_t)
            olnm_2_error = np.linalg.norm(x_olnm - x_star)
            olnm_2_errors.append(olnm_2_error)

            # Run OLNM Root Decay
            x_olnm = olnm_3.step(A_t, b_t)
            olnm_3_error = np.linalg.norm(x_olnm - x_star)
            olnm_3_errors.append(olnm_3_error)

        olnm_1_error_history.append(np.mean(olnm_1_errors))
        olnm_2_error_history.append(np.mean(olnm_2_errors))
        olnm_3_error_history.append(np.mean(olnm_3_errors))
        sgd_error_history.append(np.mean(sgd_errors))

    return np.mean(olnm_1_error_history), np.mean(olnm_2_error_history), np.mean(olnm_3_error_history), np.mean(sgd_error_history)

N, d, Epochs = 10000, 100, 1000     
m_list = np.linspace(3*d, N/2, num=10, dtype=int)

olnm_1_error_history, olnm_2_error_history, olnm_3_error_history, sgd_error_history = [], [], [], []

for m in m_list:
    print("="*5, f"Running with batch size ={m}", "="*45)
    olnm_1_err, olnm_2_err, olnm_3_err, sgd_err = run(N=N, d=d, m=m, L=500, kappa=500, Epochs=Epochs, c=30)
    olnm_1_error_history.append(olnm_1_err)
    olnm_2_error_history.append(olnm_2_err)
    olnm_3_error_history.append(olnm_3_err)
    sgd_error_history.append(sgd_err)

plt.figure(figsize=(8,6))
plt.plot(m_list, olnm_1_error_history, marker='o', label='OLNM Root Scaled', linewidth=2.5)
plt.plot(m_list, olnm_2_error_history, marker='s', label='OLNM Constant', linewidth=2.5)
plt.plot(m_list, olnm_3_error_history, marker='^', label='OLNM Root Decay', linewidth=2.5)
plt.xlabel('Batch Size (m)', fontsize=16)
plt.ylabel('Final Training Error', fontsize=16)
plt.title('Final Training Error vs. Batch Size', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.savefig("imgs/figure-x2.png")
# plt.show()

diff_1 = np.array(olnm_1_error_history) - np.array(olnm_2_error_history)
diff_2 = np.array(olnm_2_error_history) - np.array(olnm_2_error_history)
diff_3 = np.array(olnm_3_error_history) - np.array(olnm_2_error_history)

plt.figure(figsize=(8,6))
plt.plot(m_list, diff_1, marker='o', label='T = c1 * sqrt(b)', linewidth=2.5)
plt.plot(m_list, diff_2, marker='s', label='T = c2', linewidth=2.5)
plt.plot(m_list, diff_3, marker='^', label='T = c3 / sqrt(b)', linewidth=2.5)
plt.xlabel('Batch Size (m)', fontsize=16)
plt.ylabel('Training Error Difference', fontsize=16)
plt.title('Training Error Difference varies with Batch Size', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.savefig("imgs/figure-x3.png")
# plt.show()