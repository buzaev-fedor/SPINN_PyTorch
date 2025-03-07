import numpy as np
from tabulate import tabulate
# Define the benchmark functions
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def rosenbrock(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])

def sphere(x):
    return sum([xi ** 2 for xi in x])

def quadratic(x):
    return np.sum(np.square(x))

# Define the class for managing the pool of tasks
class OptimizationTaskPool:
    def __init__(self):
        self.tasks = {}
        
    def add_task(self, name, func, bounds, budget):
        self.tasks[name] = {'name':name,'func': func, 'bounds': bounds, 'dim': len(bounds), 'budget':budget }
        
    def get_task(self, name):
        return self.tasks.get(name)
    
    def list_tasks(self):
        return list(self.tasks.keys())
    
    def print_tasks(self):
        table_data = []
        for name, task in self.tasks.items():
            func_name = task['func'].__name__
            bounds_str = str(task['bounds'])
            dim_str = str(task['dim'])
            table_data.append([name, func_name, dim_str])
        
        headers = ["Task Name", "Function", "dim"]
        print(tabulate(table_data, headers, tablefmt="pretty"))

if __name__ == "__main__":
    # Initialize the task pool
    task_pool = OptimizationTaskPool()

    # Define dimensions
    dim = 2

    # Add tasks to the pool
    task_pool.add_task('Rastrigin', rastrigin, [(-5.12, 5.12)] * dim)
    task_pool.add_task('Rosenbrock', rosenbrock, [(-5, 10)] * dim)
    task_pool.add_task('Sphere', sphere, [(-5.12, 5.12)] * dim)
    task_pool.add_task('Quadratic', quadratic, [(-10, 10)] * dim)

    # List the tasks
    #print("Available tasks:", task_pool.list_tasks())
    task_pool.print_tasks()
    
    