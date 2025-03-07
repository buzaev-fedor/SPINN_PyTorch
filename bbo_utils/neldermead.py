'''

The Nelder-Mead Simplex Algorithm is a well-known optimization method used 
for finding the minimum of an objective function in a multidimensional space. 
It is a derivative-free optimization algorithm that uses the concept of a 
simplex, which is a polytope with n+1 vertices in n-dimensional space.

'''


import numpy as np
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

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])
        dimension = len(self.bounds)

        # Initialize the simplex
        simplex = [np.random.uniform(lower_bound, upper_bound, dimension) for _ in range(dimension + 1)]
        simplex_scores = [self.fitness_function(vertex) for vertex in simplex]

        def order_simplex():
            nonlocal simplex, simplex_scores
            indices = np.argsort(simplex_scores)
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
            fxr = self.fitness_function(xr)

            if simplex_scores[0] <= fxr < simplex_scores[-2]:
                simplex[-1] = xr
                simplex_scores[-1] = fxr
            elif fxr < simplex_scores[0]:
                xe = centroid_point + self.gamma * (xr - centroid_point)
                xe = np.clip(xe, lower_bound, upper_bound)
                fxe = self.fitness_function(xe)
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_scores[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_scores[-1] = fxr
            else:
                xc = centroid_point + self.rho * (simplex[-1] - centroid_point)
                xc = np.clip(xc, lower_bound, upper_bound)
                fxc = self.fitness_function(xc)
                if fxc < simplex_scores[-1]:
                    simplex[-1] = xc
                    simplex_scores[-1] = fxc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower_bound, upper_bound)
                        simplex_scores[i] = self.fitness_function(simplex[i])

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
