'''
GWO is inspired by the social hierarchy and hunting behavior of grey wolves. 
It uses the positions of the alpha, beta, and delta wolves to guide the search process.
'''

import numpy as np
from .optimizer import Optimizer
from .tasks import OptimizationTaskPool, rastrigin
import matplotlib.pyplot as plt
import torch

class GreyWolfOptimizer(Optimizer):
    def __init__(self, task, num_wolves=30):
        super().__init__(task)
        self.solver_name = 'GreyWolfOptimizer'
        self.num_wolves = num_wolves
        self.wolves = None
        self.alpha = None
        self.beta = None
        self.delta = None

    def _convert_to_numpy(self, tensor):
        """Helper method to convert tensor to numpy value."""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.detach().numpy()
            if isinstance(tensor, np.ndarray):
                tensor = float(tensor)
        return tensor

    def initialize_wolves(self, lower_bound, upper_bound):
        dimension = len(lower_bound)
        self.wolves = np.random.uniform(lower_bound, upper_bound, (self.num_wolves, dimension))
        self.alpha = np.copy(self.wolves[0])
        self.beta = np.copy(self.wolves[1])
        self.delta = np.copy(self.wolves[2])

    def update_positions(self, lower_bound, upper_bound):
        a = 2 - 2 * (self.current_iteration / self.num_iterations)  # Decrease linearly from 2 to 0

        for i in range(self.num_wolves):
            for j in range(len(lower_bound)):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha[j] - self.wolves[i, j])
                X1 = self.alpha[j] - A1 * D_alpha

                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta[j] - self.wolves[i, j])
                X2 = self.beta[j] - A2 * D_beta

                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta[j] - self.wolves[i, j])
                X3 = self.delta[j] - A3 * D_delta

                self.wolves[i, j] = (X1 + X2 + X3) / 3.0

            self.wolves[i] = np.clip(self.wolves[i], lower_bound, upper_bound)

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])

        self.initialize_wolves(lower_bound, upper_bound)
        
        self.num_iterations = self.budget // self.num_wolves
        
        # Convert alpha to tensor for initial fitness evaluation
        alpha_tensor = torch.from_numpy(self.alpha).float()
        if torch.cuda.is_available():
            alpha_tensor = alpha_tensor.cuda()
        self.best_solution = np.copy(self.alpha)
        self.best_objective_function = self._convert_to_numpy(self.fitness_function(alpha_tensor))

        for self.current_iteration in range(self.num_iterations):
            # Convert wolves to tensor for fitness evaluation
            wolves_tensor = torch.from_numpy(self.wolves).float()
            if torch.cuda.is_available():
                wolves_tensor = wolves_tensor.cuda()
            
            # Evaluate fitness for all wolves
            scores = np.array([self._convert_to_numpy(self.fitness_function(w)) for w in wolves_tensor])
            indices = np.argsort(scores)

            self.alpha = np.copy(self.wolves[indices[0]])
            self.beta = np.copy(self.wolves[indices[1]])
            self.delta = np.copy(self.wolves[indices[2]])

            if scores[indices[0]] < self.best_objective_function:
                self.best_solution = np.copy(self.alpha)
                self.best_objective_function = scores[indices[0]]

            self.update_positions(lower_bound, upper_bound)
            self.objective_function_history.append(self.best_objective_function)

if __name__ == "__main__":
    task_pool = OptimizationTaskPool()

    # Add tasks to the pool
    task_pool.add_task('Rastrigin', rastrigin, [(-5.12, 5.12)] * 5, 4000)
    
    gwo_solver = GreyWolfOptimizer(task_pool.get_task('Rastrigin'))
    gwo_solver.minimize()
    
    gwo_solver.print_results()
    gwo_solver.make_plot()
    plt.legend()
    plt.show()
