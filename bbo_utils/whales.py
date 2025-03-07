import numpy as np
from .optimizer import Optimizer
from .tasks import OptimizationTaskPool, rastrigin
import matplotlib.pyplot as plt
import random
from .lshade import LShadeAlgorithm


class WhaleOptimization(Optimizer):
    def __init__(self, task, population_size=30):
        super().__init__(task)
        self.solver_name = 'WhaleOptimization'
        self.population_size = population_size

    def initialize_population(self, lower_bound, upper_bound):
        """Initialize the population within bounds."""
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, len(lower_bound)))
        self.fitness = np.array([self.fitness_function(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx]
        self.best_objective_function = self.fitness[self.best_idx]

    def update_position(self, agent, best_agent, random_agent, a, A, C, p, lower_bound, upper_bound):
        """Update the position of a whale."""
        if p < 0.5:
            if abs(A) < 1:
                D = np.abs(C * best_agent - agent)
                return np.clip(best_agent - A * D, lower_bound, upper_bound)
            else:
                D = np.abs(C * random_agent - agent)
                return np.clip(random_agent - A * D, lower_bound, upper_bound)
        else:
            D = np.abs(best_agent - agent)
            return np.clip(D * np.exp(-2 * np.abs(D)) * np.cos(2 * np.pi * np.abs(D)) + best_agent, lower_bound, upper_bound)

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])

        self.initialize_population(lower_bound, upper_bound)

        max_iterations = self.budget // self.population_size

        for t in range(max_iterations):
            a = 2 - t * (2 / max_iterations)
            for i in range(self.population_size):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                p = np.random.rand()

                random_agent = self.population[np.random.randint(0, self.population_size)]
                self.population[i] = self.update_position(
                    self.population[i], self.best_solution, random_agent, a, A, C, p, lower_bound, upper_bound
                )

                fitness = self.fitness_function(self.population[i])
                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness

                    if fitness < self.best_objective_function:
                        self.best_solution = self.population[i]
                        self.best_objective_function = fitness

            self.objective_function_history.append(self.best_objective_function)



if __name__ == "__main__":
    task_pool = OptimizationTaskPool()

    # Add tasks to the pool
    task_pool.add_task('Rastrigin', rastrigin, [(-5.12, 5.12)] * 5, 10000)

    lshade_solver = LShadeAlgorithm(task_pool.get_task('Rastrigin'), population_size=32)
    lshade_solver.minimize()

    lshade_solver.print_results()
    lshade_solver.make_plot()

    woa_solver = WhaleOptimization(task_pool.get_task('Rastrigin'), population_size=30)
    woa_solver.minimize()
    woa_solver.print_results()
    woa_solver.make_plot()

    plt.legend()
    plt.show()
