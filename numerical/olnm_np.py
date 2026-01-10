import numpy as np
from math import ceil

class SGD_Optimizer:
    def __init__(self, d, m, step_size=1e-4):
        self.step_size = step_size
        self.x = np.zeros(d)  
        self.t = 0            

    def _grad(self, x, A_t, b_t, m):
        residual = A_t @ x - b_t
        gradient = (1 / m) * (A_t.T @ residual)
        return gradient

    def step(self, A_t, b_t):
        self.t += 1
        m = A_t.shape[0]  # minibatch size

        nabla_f_t_x = self._grad(self.x, A_t, b_t, m)
        x_next = self.x - self.step_size * nabla_f_t_x

        self.x = x_next
        
        return self.x
    
class OLNM_Optimizer:
    def __init__(self, d, L, kappa, m, method='default', c=1.0):
        self.L = L
        self.kappa = kappa
        
        if method == 'root_scaled':
            self.BigT = ceil(c * np.sqrt(m))
        elif method == 'default':
            self.BigT = ceil((2 + np.sqrt(2)) * np.sqrt(kappa))
        elif method == 'root_decay':
            self.BigT = ceil(c / np.sqrt(m))
        elif method == 'constant':
            self.BigT = c

        # print(f'Method: {method}; Using BigT = {self.BigT}')
        
        self.x = np.zeros(d)
        self.y = np.zeros(d)
        self.z = np.zeros(d)
        self.a = 1.0
        self.t = 0

    def _grad(self, x, A_t, b_t, m):
        residual = A_t @ x - b_t
        gradient = (1 / m) * (A_t.T @ residual) # m for minibatch size
        return gradient

    def step(self, A_t, b_t):
        self.t += 1
        m = A_t.shape[0]

        # z_t+1
        nabla_f_t_y_t = self._grad(self.y, A_t, b_t, m)
        z_next = self.y - (1 / self.L) * nabla_f_t_y_t

        z_current = self.z
        a_current = self.a

        if (self.t % self.BigT) != 0: 
            a_next = (1 + np.sqrt(1 + 4 * a_current**2)) / 2
            y_next = z_next + ((a_current - 1) / a_next) * (z_next - z_current)
            x_next = self.x
            
        else:
            a_next = 1.0
            y_next = z_next
            x_next = z_next
            
        # update states
        self.z = z_next
        self.y = y_next
        self.a = a_next
        self.x = x_next
        
        return self.x

def get_experiment_data(n, d, L, kappa, seed=None):
    import numpy as np
    from scipy.linalg import qr

    if seed is not None:
        np.random.seed(seed)

    mu = L / kappa

    # eigenvalues for (1/n) A^T A
    eigvals = np.linspace(mu, L, d)

    # random orthonormal basis Q (d x d)
    Z = np.random.randn(d, d)
    Q, _ = qr(Z)

    # G = Q diag(eigvals) Q^T
    G = Q @ np.diag(eigvals) @ Q.T

    # (1/n) A^T A = G => A = sqrt(n) * U * sqrt(diag(eigvals)) * Q^T
    Z2 = np.random.randn(n, d)
    U, _ = qr(Z2, mode='economic')

    sqrtG = Q @ np.diag(np.sqrt(eigvals))
    A = np.sqrt(n) * (U @ sqrtG.T)

    Gram = (A.T @ A) / n
    min_eig, max_eig = np.linalg.eigvalsh(Gram)[[0, -1]]

    return A, mu

