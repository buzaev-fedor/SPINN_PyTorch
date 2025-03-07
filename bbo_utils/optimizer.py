import numpy as np
import matplotlib.pyplot as plt
import time 
import torch

class Optimizer:
    def __init__(self, task):
        self.objective_function_history = []
        self.best_solution = None
        self.best_objective_function = np.inf
        self.task_name = task['name']
        
        # Оборачиваем fitness_function для автоматического преобразования тензоров
        original_func = task['func']
        def wrapped_func(x):
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x = x.cpu()
                x = x.detach().numpy()
            return original_func(x)
        self.fitness_function = wrapped_func
        
        self.bounds = task['bounds']
        self.budget = task['budget']
        self.solver_name = 'Undefined'
        self.elapsed_time = 'Undefined'
    
    def run_minimize(self):
        start_time = time.time()
        self.minimize()
        end_time = time.time()
        self.elapsed_time = end_time - start_time

    def minimize(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def print_results(self):
        print(f'Task: {self.task_name}')
        print('Best objective function:', self.best_objective_function)
        print('Found solution:', self.best_solution)
        print(f'Elapsed time: {self.elapsed_time} sec')
    
    def make_plot(self):
        plt.semilogy(self.objective_function_history, label=self.solver_name + ': '+self.task_name)
        
    def run_multiple_times(self, n_runs=10):
        self.all_histories = []
        self.best_objective_functions = []
        start_time = time.time()
        for _ in range(n_runs):
            self.objective_function_history = []
            self.best_solution = None
            self.best_objective_function = np.inf
            self.run_minimize()
            self.best_objective_functions.append(self.best_objective_function)
        end_time = time.time()
        self.elapsed_time = (end_time - start_time)/n_runs

        self.best_mean = np.mean(self.best_objective_functions)
        self.best_std = np.std(self.best_objective_functions)