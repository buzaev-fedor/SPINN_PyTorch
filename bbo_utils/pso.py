import numpy as np
from .optimizer import Optimizer
from .tasks import OptimizationTaskPool, rastrigin
import matplotlib.pyplot as plt
import random
import torch

class ParticleSwarmOptimization(Optimizer):
    def __init__(self, task, num_particles=30, num_iterations=100, w=0.5, c1=1.0, c2=1.0):
        super().__init__(task)
        self.solver_name = 'ParticleSwarmOptimization'
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive (particle) weight
        self.c2 = c2  # Social (swarm) weight
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def _convert_to_numpy(self, tensor):
        """Helper method to convert tensor to numpy value."""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.detach().numpy()
            if isinstance(tensor, np.ndarray):
                tensor = float(tensor)
        return tensor

    def initialize_particles(self, lower_bound, upper_bound):
        self.particles = np.random.uniform(lower_bound, upper_bound, (self.num_particles, len(lower_bound)))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, len(lower_bound)))
        self.personal_best_positions = np.copy(self.particles)
        
        # Convert particles to tensor for fitness evaluation
        particles_tensor = torch.from_numpy(self.particles).float()
        if torch.cuda.is_available():
            particles_tensor = particles_tensor.cuda()
        
        # Evaluate fitness for all particles
        self.personal_best_scores = np.array([
            self._convert_to_numpy(self.fitness_function(p)) for p in particles_tensor
        ])

        # Initialize global best
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx]
        self.global_best_score = self.personal_best_scores[best_idx]

    def update_particles(self, lower_bound, upper_bound):
        for i in range(self.num_particles):
            r1 = np.random.rand(len(lower_bound))
            r2 = np.random.rand(len(lower_bound))

            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity
            self.particles[i] = self.particles[i] + self.velocities[i]

            # Ensure particles stay within bounds
            self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            # Convert particle to tensor for fitness evaluation
            particle_tensor = torch.from_numpy(self.particles[i]).float()
            if torch.cuda.is_available():
                particle_tensor = particle_tensor.cuda()
            
            current_score = self._convert_to_numpy(self.fitness_function(particle_tensor))
            if current_score < self.personal_best_scores[i]:
                self.personal_best_positions[i] = self.particles[i]
                self.personal_best_scores[i] = current_score

            if current_score < self.global_best_score:
                self.global_best_position = self.particles[i]
                self.global_best_score = current_score

    def minimize(self):
        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])

        self.initialize_particles(lower_bound, upper_bound)
        
        self.num_iterations = self.budget // self.num_particles
        for _ in range(self.num_iterations):
            self.update_particles(lower_bound, upper_bound)
            self.objective_function_history.append(self.global_best_score)

        self.best_solution = self.global_best_position
        self.best_objective_function = self.global_best_score

if __name__ == "__main__":
    task_pool = OptimizationTaskPool()

    # Add tasks to the pool
    task_pool.add_task('Rastrigin', rastrigin, [(-5.12, 5.12)] * 5, 10000)
    
    pso_solver = ParticleSwarmOptimization(task_pool.get_task('Rastrigin'))
    pso_solver.run_minimize()
    
    pso_solver.print_results()
    pso_solver.make_plot()
    plt.legend()
    plt.show()
