'''
 L-SHADE algorithm (Success-History based Adaptive Differential Evolution with Linear population size reduction), 
 which is an extension of the JADE algorithm. L-SHADE is known for its efficient and effective 
 performance on various optimization problems. It includes mechanisms for adapting control 
 parameters and reducing the population size over generations.
'''

import numpy as np
from .optimizer import Optimizer
from .tasks import OptimizationTaskPool, rastrigin
import matplotlib.pyplot as plt
import random

class LShadeAlgorithm(Optimizer):
    def __init__(self, task, population_size=100, c=0.1, p=0.05):
        super().__init__(task)
        self.solver_name = 'LShadeAlgorithm'
        self.initial_population_size = population_size
        self.population_size = population_size
        self.budget = self.budget
        self.c = c  # Learning rate for adapting parameters
        self.p = p  # Proportion of top individuals to use in mutation
        self.archive = []

    def initialize_population(self, lower_bound, upper_bound):
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, len(lower_bound)))
        self.fitness = np.array([self.fitness_function(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx]
        self.best_objective_function = self.fitness[self.best_idx]
        self.mean_cr = 0.5
        self.mean_f = 0.5

    def mutate(self, idx, lower_bound, upper_bound):
        r1, r2 = np.random.choice(self.population_size, 2, replace=False)
        if random.random() < 0.5 and len(self.archive) > 0:
            r3 = random.randint(0, len(self.archive) - 1)
            x_r3 = self.archive[r3]
        else:
            r3 = np.random.choice(self.population_size, 1)[0]
            x_r3 = self.population[r3]
        
        while r1 == idx or r2 == idx or r3 == idx:
            r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
        
        if random.random() < self.p:
            num_top_individuals = max(1, int(self.p * self.population_size))
            idxs = np.argsort(self.fitness)[:num_top_individuals]
            x_best_p = self.population[np.random.choice(idxs)]
        else:
            x_best_p = self.best_solution
        
        f = np.random.normal(self.mean_f, 0.1)
        while f <= 0:
            f = np.random.normal(self.mean_f, 0.1)
        
        mutant = self.population[idx] + f * (x_best_p - self.population[idx]) + f * (self.population[r1] - self.population[r2])
        mutant = np.clip(mutant, lower_bound, upper_bound)
        return mutant, f

    def crossover(self, target, mutant, cr):
        crossover_mask = np.random.rand(len(target)) < cr
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_archive(self, target):
        if len(self.archive) < self.initial_population_size:
            self.archive.append(target)
        else:
            self.archive[random.randint(0, len(self.archive) - 1)] = target

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])
        max_iterations = self.budget // self.initial_population_size
        reduction_factor = (self.initial_population_size - 4) / (max_iterations - 1)

        self.initialize_population(lower_bound, upper_bound)

        evaluations = 0
        for generation in range(max_iterations):
            new_population = np.copy(self.population)
            cr_values = []
            f_values = []
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                mutant, f = self.mutate(i, lower_bound, upper_bound)
                cr = np.clip(np.random.normal(self.mean_cr, 0.1), 0, 1)
                trial = self.crossover(self.population[i], mutant, cr)
                trial_fitness = self.fitness_function(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.update_archive(self.population[i])
                    cr_values.append(cr)
                    f_values.append(f)
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_objective_function:
                        self.best_solution = trial
                        self.best_objective_function = trial_fitness

            self.population = new_population
            self.objective_function_history.append(self.best_objective_function)
            
            if cr_values:
                self.mean_cr = (1 - self.c) * self.mean_cr + self.c * np.mean(cr_values)
            if f_values:
                f_values = np.array(f_values)
                self.mean_f = (1 - self.c) * self.mean_f + self.c * (np.sum(f_values ** 2) / np.sum(f_values))
            
            # Reduce population size
            self.population_size = int(self.initial_population_size - generation * reduction_factor)
            if self.population_size < 4:
                self.population_size = 4

            # Select the best individuals to form the new population
            best_indices = np.argsort(self.fitness)[:self.population_size]
            self.population = self.population[best_indices]
            self.fitness = self.fitness[best_indices]

            if evaluations >= self.budget:
                break

        self.best_solution = self.population[np.argmin(self.fitness)]
        self.best_objective_function = np.min(self.fitness)
        self.objective_function_history.append(self.best_objective_function)

if __name__ == "__main__":
    task_pool = OptimizationTaskPool()

    # Add tasks to the pool
    task_pool.add_task('Rastrigin', rastrigin, [(-5.12, 5.12)] * 5, 10000)
    
    lshade_solver = LShadeAlgorithm(task_pool.get_task('Rastrigin'), population_size=32)
    lshade_solver.run_minimize()
    
    lshade_solver.print_results()
    lshade_solver.make_plot()
    plt.legend()
    plt.show()
