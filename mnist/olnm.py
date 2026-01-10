import math
import torch
from torch.optim.optimizer import Optimizer, required

class OLNM(Optimizer):
    def __init__(self, params, 
                 lr=required, 
                 c=1, 
                 batch_size=required, 
                 adaptive=False, 
                 error_detector=None):
        
        if adaptive and error_detector is None:
            raise ValueError("If 'adaptive' is True, 'error_detector' must be provided.")

        defaults = dict(lr=lr, 
                        batch_size=batch_size, 
                        T=math.ceil(c / math.sqrt(batch_size)), # ROOT_DECAY
                        c=c,
                        adaptive=adaptive,
                        error_detector=error_detector)
        
        super(OLNM, self).__init__(params, defaults)
        
        for group in self.param_groups:
            # Common initialization
            group['t'] = 0
            group['a'] = 1.0
            
            # Adaptive specific initialization
            if group['adaptive']:
                group['prev_loss'] = 0.0 # Initialize high
                group['tracking_errors'] = []

            for p in group['params']:
                state = self.state[p]
                state['y'] = p.data.clone().detach()
                state['z'] = p.data.clone().detach()

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss. Required if adaptive=True.
        """
        group = self.param_groups[0]
        if group['adaptive'] and closure is None:
            raise RuntimeError("Closure is required for OLNM when adaptive=True")

        # 1. Prepare: Swap current params (x) with extrapolation (y) for gradient calculation
        x_backup = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
                if 'y' in self.state[p]:
                    state = self.state[p]
                    x_backup[p] = p.data.clone().detach()
                    p.data.copy_(state['y'])
        
        # 2. Compute Gradient at y_t
        loss = closure()
        
        # 3. Update parameters
        self._update(x_backup, closure)
        
        return loss

    @torch.no_grad()
    def _update(self, x_backup, closure):
        group = self.param_groups[0]
        t = group['t']
        T = group['T']
        a = group['a']
        lr = group['lr']
        adaptive = group['adaptive']

        z_next_list = {}
        
        # --- Step A: Calculate z_{t+1} (Gradient Descent step on y_t) ---
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    if p in x_backup:
                        p.data.copy_(x_backup[p]) # Restore if no grad
                    continue
                
                # z_{k+1} = y_k - lr * grad(y_k)
                z_next = torch.add(self.state[p]['y'], p.grad.data, alpha=-lr)
                z_next_list[p] = z_next
                
                # Restore p.data to original x before logic determines next step
                p.data.copy_(x_backup[p])

        # --- Step B: Determine if we need to Restart (Outer Loop) ---
        should_restart = False
        
        # Condition 1: Fixed Schedule (End of period T)
        if (t + 1) % T == 0:
            should_restart = True
            if adaptive: 
                 print(f"*** Reached maximum inner loop T = {T}")

        # Condition 2: Adaptive Error Detection (Only if not already restarting)
        elif adaptive:
            # To check adaptive error, we must tentatively compute y_next and check loss
            # Calculate tentative variables
            a_next = (1 + math.sqrt(1 + 4 * a**2)) / 2
            beta = (a - 1) / a_next
            
            # Apply tentative update to y state to measure loss
            for p, z_next in z_next_list.items():
                y_next = torch.add(z_next, z_next - self.state[p]['z'], alpha=beta)
                self.state[p]['y'].copy_(y_next)
                # Note: We don't update z state yet, waiting for decision

            # Check loss at tentative y_next
            # We assume closure handles the model forward pass using the updated p.data (which acts as y)
            # We must load y_next into p.data for the closure
            for p in z_next_list:
                p.data.copy_(self.state[p]['y'])
            
            check_loss = closure(grad=False) # In inference mode ideally, or just forward
            
            # Restore p.data to x_backup for consistency after check
            for p in z_next_list:
                p.data.copy_(x_backup[p])

            # Error Detection Logic
            current_loss_val = check_loss.item() if isinstance(check_loss, torch.Tensor) else check_loss
            prev_loss = group['prev_loss']
            
            # Handle first step initialization of prev_loss
            if prev_loss == float('inf'):
                error_t = 0.0
            else:
                error_t = abs(current_loss_val - prev_loss)
            
            group['prev_loss'] = current_loss_val
            
            old_errors = group['tracking_errors']
            updated_errors = old_errors + [error_t]
            
            is_change_detected = group['error_detector'].detect_change(old_errors, updated_errors)
            group['tracking_errors'] = updated_errors

            if is_change_detected:
                print(f"Error change detected. Restarting at t = {t + 1}")
                should_restart = True
            else:
                # If no restart, we keep the tentative y_next we calculated
                # And update z state formally
                for p, z_next in z_next_list.items():
                    self.state[p]['z'].copy_(z_next)
                group['a'] = a_next
                group['t'] += 1

        # --- Step C: Execute Update ---
        
        if should_restart:
            # Outer Loop Update (Reset)
            self._outer_loop_update(z_next_list)
        elif not adaptive:
            # Standard Inner Loop Update (Non-Adaptive)
            # (Adaptive path handled update inside Condition 2 block to avoid re-calc)
            a_next = (1 + math.sqrt(1 + 4 * a**2)) / 2
            beta = (a - 1) / a_next
            
            for group in self.param_groups:
                for p in group['params']:
                    if p not in z_next_list: continue
                    z_next = z_next_list[p]
                    y_next = torch.add(z_next, z_next - self.state[p]['z'], alpha=beta)
                    
                    self.state[p]['z'].copy_(z_next)
                    self.state[p]['y'].copy_(y_next)
            
            group['a'] = a_next
            group['t'] += 1

    def _outer_loop_update(self, z_next_list):
        """Resets the sequence (Restart)"""
        for group in self.param_groups:
            group['t'] = 0
            group['a'] = 1.0
            if group['adaptive']:
                group['prev_loss'] = 0.0
                group['tracking_errors'] = []
            
            for p in group['params']:
                if p not in z_next_list:
                    continue
                z_next = z_next_list[p]
                
                # Restart: y, z, and actual weights (x) all become z_next
                self.state[p]['y'].copy_(z_next)
                self.state[p]['z'].copy_(z_next)
                p.data.copy_(z_next)

import numpy as np
from typing import List
from scipy.stats import f
from abc import ABC, abstractmethod
class ErrorChangeDetector(ABC):
    @abstractmethod
    def detect_change(self, old_error_list: List[float], 
                      updated_error_list: List[float]
                      ) -> bool:
        pass

    def _get_current_error_and_history(self, old_error_list: List[float], 
                                       updated_error_list: List[float]
                                       ):
        if not updated_error_list:
            raise ValueError("Updated_Error_List cannot be empty.")
        if len(updated_error_list) != len(old_error_list) + 1:
            raise ValueError("Updated_Error_List must be one element longer than Old_Error_List.")
        
        err_t = updated_error_list[-1]
        return err_t, old_error_list
    
class FTestDetector(ErrorChangeDetector):
    def __init__(self, alpha: float = 0.05):
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha

    def detect_change(self, old_error_list: List[float], 
                      updated_error_list: List[float]) -> bool:
        # F-test requires at least two samples in each group to compute variance.
        if len(old_error_list) < 2 or len(updated_error_list) < 2:
            return False

        var1 = np.var(np.array(old_error_list), ddof=1)
        var2 = np.var(np.array(updated_error_list), ddof=1)

        # Avoid division by zero if variance is zero
        if var1 == 0:
            return var2 > 0

        F = var2 / var1
        dof1 = len(updated_error_list) - 1
        dof2 = len(old_error_list) - 1
        
        f_critical = f.ppf(1 - self.alpha, dof1, dof2)

        return F > f_critical
    
class MovingAverageDetector(ErrorChangeDetector):
    """
    Detects change based on a simple moving average of recent errors.
    """
    def __init__(self, window_size: int = 10, c: float = 2.0):
        self.window_size = window_size
        self.c = c

    def detect_change(self, old_error_list: List[float], 
                      updated_error_list: List[float]) -> bool:
        err_t, history = self._get_current_error_and_history(old_error_list, updated_error_list)
        
        if len(history) < self.window_size:
            return False
            
        recent_history = history[-self.window_size:]
        moving_average = np.mean(recent_history)
        
        return err_t > self.c * moving_average