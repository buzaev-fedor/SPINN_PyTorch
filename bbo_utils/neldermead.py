'''

The Nelder-Mead Simplex Algorithm is a well-known optimization method used 
for finding the minimum of an objective function in a multidimensional space. 
It is a derivative-free optimization algorithm that uses the concept of a 
simplex, which is a polytope with n+1 vertices in n-dimensional space.

'''


import numpy as np
import torch
from .optimizer import Optimizer
from .tasks import OptimizationTaskPool, rastrigin, sphere
import matplotlib.pyplot as plt

class NelderMead(Optimizer):
    def __init__(self, task, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        super().__init__(task)
        self.solver_name = 'NelderMead'
        self.alpha = alpha  # Reflection coefficient
        self.gamma = gamma  # Expansion coefficient
        self.rho = rho      # Contraction coefficient
        self.sigma = sigma  # Shrink coefficient

    def _convert_to_numpy(self, tensor):
        """Helper method to convert tensor to numpy value."""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.detach().numpy()
            if isinstance(tensor, np.ndarray):
                tensor = float(tensor)
        return tensor

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])
        dimension = len(self.bounds)

        # Initialize the simplex
        simplex = [np.random.uniform(lower_bound, upper_bound, dimension) for _ in range(dimension + 1)]
        simplex_scores = []
        
        # Convert simplex to tensor for fitness evaluation
        simplex_tensor = torch.from_numpy(np.array(simplex)).float()
        if torch.cuda.is_available():
            simplex_tensor = simplex_tensor.cuda()
        
        # Evaluate fitness for all vertices
        for vertex in simplex_tensor:
            score = self.fitness_function(vertex)
            simplex_scores.append(self._convert_to_numpy(score))

        def order_simplex():
            nonlocal simplex, simplex_scores
            scores_array = np.array(simplex_scores)
            indices = np.argsort(scores_array)
            simplex = [simplex[i] for i in indices]
            simplex_scores = [simplex_scores[i] for i in indices]

        def centroid(vertices):
            return np.mean(vertices[:-1], axis=0)

        order_simplex()
        self.best_solution = simplex[0]
        self.best_objective_function = simplex_scores[0]
        self.objective_function_history.append(self.best_objective_function)

        for _ in range(self.budget):
            order_simplex()
            centroid_point = centroid(simplex)
            xr = centroid_point + self.alpha * (centroid_point - simplex[-1])
            xr = np.clip(xr, lower_bound, upper_bound)
            
            # Convert to tensor for fitness evaluation
            xr_tensor = torch.from_numpy(xr).float()
            if torch.cuda.is_available():
                xr_tensor = xr_tensor.cuda()
            fxr = self._convert_to_numpy(self.fitness_function(xr_tensor))

            if simplex_scores[0] <= fxr < simplex_scores[-2]:
                simplex[-1] = xr
                simplex_scores[-1] = fxr
            elif fxr < simplex_scores[0]:
                xe = centroid_point + self.gamma * (xr - centroid_point)
                xe = np.clip(xe, lower_bound, upper_bound)
                
                # Convert to tensor for fitness evaluation
                xe_tensor = torch.from_numpy(xe).float()
                if torch.cuda.is_available():
                    xe_tensor = xe_tensor.cuda()
                fxe = self._convert_to_numpy(self.fitness_function(xe_tensor))
                
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_scores[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_scores[-1] = fxr
            else:
                xc = centroid_point + self.rho * (simplex[-1] - centroid_point)
                xc = np.clip(xc, lower_bound, upper_bound)
                
                # Convert to tensor for fitness evaluation
                xc_tensor = torch.from_numpy(xc).float()
                if torch.cuda.is_available():
                    xc_tensor = xc_tensor.cuda()
                fxc = self._convert_to_numpy(self.fitness_function(xc_tensor))
                
                if fxc < simplex_scores[-1]:
                    simplex[-1] = xc
                    simplex_scores[-1] = fxc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower_bound, upper_bound)
                        
                        # Convert to tensor for fitness evaluation
                        vertex_tensor = torch.from_numpy(simplex[i]).float()
                        if torch.cuda.is_available():
                            vertex_tensor = vertex_tensor.cuda()
                        score = self._convert_to_numpy(self.fitness_function(vertex_tensor))
                        simplex_scores[i] = score

            self.best_solution = simplex[0]
            self.best_objective_function = simplex_scores[0]
            self.objective_function_history.append(self.best_objective_function)

if __name__ == "__main__":
    task_pool = OptimizationTaskPool()

    # Add tasks to the pool
    task_pool.add_task('Sphere', sphere, [(-5.12, 5.12)] * 5, 100)
    task_pool.add_task('Sphere', sphere, [(-5.12, 5.12)] * 5, 100)
    nm_solver = NelderMead(task_pool.get_task('Sphere'))
    nm_solver.minimize()
    
    nm_solver = NelderMead(task_pool.get_task('Sphere'))
    nm_solver.minimize()    
    
    nm_solver.print_results()
    nm_solver.make_plot()
    plt.legend()
    plt.show()
