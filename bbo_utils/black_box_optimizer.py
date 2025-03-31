import os
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

# Import BBO algorithms
from .jade import JadeAlgorithm
from .lshade import LShadeAlgorithm
from .neldermead import NelderMead
from .pso import ParticleSwarmOptimization
from .grey_wolf_optimizer import GreyWolfOptimizer
from .whales import WhaleOptimization

class BlackBoxOptimizer:
    ALGORITHMS = {
        'jade': JadeAlgorithm,
        'lshade': LShadeAlgorithm,
        'nelder_mead': NelderMead,
        'pso': ParticleSwarmOptimization,
        'grey_wolf': GreyWolfOptimizer,
        'whales': WhaleOptimization
    }

    def _get_algorithm_description(self, algorithm: str) -> str:
        """Возвращает описание алгоритма оптимизации."""
        descriptions = {
            'jade': "JADE (Adaptive Differential Evolution) - Адаптивный алгоритм дифференциальной эволюции. "
                   "Автоматически адаптирует параметры мутации и скрещивания. "
                   "Эффективен для непрерывной оптимизации и хорошо масштабируется.",
            
            'lshade': "L-SHADE (Linear Success-History based Adaptive DE) - Улучшенная версия SHADE алгоритма. "
                     "Использует линейное уменьшение размера популяции и историю успешных решений. "
                     "Особенно эффективен для задач большой размерности.",
            
            'nelder_mead': "Nelder-Mead (Симплекс-метод) - Безградиентный метод оптимизации. "
                          "Использует симплекс для поиска минимума функции. "
                          "Хорошо работает для гладких функций небольшой размерности.",
            
            'pso': "Particle Swarm Optimization - Метод роя частиц. "
                  "Имитирует социальное поведение птиц или рыб. "
                  "Эффективен для многомодальных функций и параллельных вычислений.",
            
            'grey_wolf': "Grey Wolf Optimizer - Алгоритм, имитирующий иерархию и охотничье поведение серых волков. "
                        "Хорошо балансирует между глобальным и локальным поиском. "
                        "Эффективен для сложных многомерных задач.",
            
            'whales': "Whale Optimization Algorithm - Алгоритм, основанный на охотничьем поведении горбатых китов. "
                     "Использует стратегию пузырьковой сети и преследования добычи. "
                     "Хорошо подходит для мультимодальных функций."
        }
        return descriptions.get(algorithm, "Описание алгоритма отсутствует")

    def __init__(self, 
                 n_trials: int = 150,
                 timeout: Optional[int] = None,
                 study_name: str = "spinn_optimization",
                 logger: Optional['ResultLogger'] = None,
                 algorithm: str = 'jade',
                 algorithm_params: Optional[Dict] = None,
                 verbose: bool = True):
        """
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (None for no timeout)
            study_name: Name of the study
            logger: Logger for saving results
            algorithm: Algorithm name ('jade', 'lshade', 'nelder_mead', 'pso', 'grey_wolf', 'whales')
            algorithm_params: Additional parameters for the algorithm
            verbose: Whether to print detailed logs
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.logger = logger
        self.best_trial = None
        self.verbose = verbose
        self.current_trial = 0
        self.best_error_so_far = float('inf')
        self.training_history = []  # Добавляем список для хранения истории обучения
        
        # Validate algorithm choice
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHMS.keys())}")
        
        self.algorithm_name = algorithm
        self.algorithm_params = algorithm_params or {}
        
        if self.verbose:
            print("\n" + "="*60)
            print(f"Initializing {self.algorithm_name.upper()} optimizer")
            print("="*60)
            print("Algorithm description:")
            print(self._get_algorithm_description(algorithm))
            print("-"*60)
            print(f"Configuration:")
            print(f"  Number of trials: {self.n_trials}")
            print(f"  Timeout: {self.timeout if self.timeout else 'None'}")
            print(f"  Algorithm parameters: {self.algorithm_params}")
            print("-"*60)

    def create_model(self, params: Dict) -> Tuple[nn.Module, Dict]:
        """Creates a model with the given parameters.
        This method should be implemented by the user to create a model based on parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def train_model(self, model: nn.Module, optimizer_params: Dict, train_data: Tuple, test_data: Tuple,
                   device: torch.device, n_epochs: int) -> float:
        """Trains the model and returns the validation error.
        This method should be implemented by the user to train the model.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def objective_function(self, params: Dict) -> float:
        """Objective function for optimization."""
        # Add training parameters to the optimization parameters
        params.update(self.train_params)
        
        self.current_trial += 1
        
        if self.verbose:
            print("\n" + "-"*60)
            print(f"Trial {self.current_trial}/{self.n_trials}")
            print("-"*60)
            print("Current parameters:")
            print(f"  Architecture:")
            print(f"    Layers: {params['n_layers']}")
            print(f"    Layer sizes: {[params[f'layer_{i}_size'] for i in range(params['n_layers'])]}")
            print(f"    Activation: {params['activation']}")
            print(f"  Optimizers:")
            print(f"    AdamW lr: {params['lr_adamw']:.2e}")
            print(f"  Scheduler: {params['scheduler_type']}")
            if params['scheduler_type'] != 'none':
                print("    Parameters:")
                if params['scheduler_type'] == 'reduce_on_plateau':
                    print(f"      Factor: {params['scheduler_factor']:.2f}")
                    print(f"      Patience: {params['scheduler_patience']}")
                    print(f"      Min lr: {params['scheduler_min_lr']:.2e}")
                elif params['scheduler_type'] == 'cosine_annealing':
                    print(f"      T_max: {params['scheduler_T_max']}")
                    print(f"      Eta min: {params['scheduler_eta_min']:.2e}")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create model and get optimizer parameters
        model, optimizer_params = self.create_model(params)
        model = model.to(device)
        
        # Generate data
        train_data = self.generate_train_data(params)
        train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                     [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                     for t in train_data]
        
        test_data = self.generate_test_data(params)
        test_data = [t.to(device) for t in test_data]
        
        # Train model with specified epochs
        n_epochs = params['EPOCHS']
        
        if self.verbose:
            print(f"\nTraining for {n_epochs} epochs...")
        
        error = self.train_model(
            model=model,
            optimizer_params=optimizer_params,
            train_data=train_data,
            test_data=test_data,
            device=device,
            n_epochs=n_epochs
        )
        
        if error < self.best_error_so_far:
            self.best_error_so_far = error
            if self.verbose:
                print("\n" + "*"*60)
                print(f"New best error found: {error:.2e}")
                print("*"*60)
        
        if self.verbose:
            print(f"\nTrial {self.current_trial} completed:")
            print(f"  Current error: {error:.2e}")
            print(f"  Best error so far: {self.best_error_so_far:.2e}")
        
        # Сохраняем результаты trial
        if self.logger is not None:
            self.logger.log_trial(
                trial_number=self.current_trial,
                optimizer_name=self.algorithm_name,
                error=error,
                params=params,
                training_history=self.training_history
            )
        
        return error

    def generate_train_data(self, params: Dict) -> Tuple:
        """Generate training data based on parameters.
        This method should be implemented by the user to generate training data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def generate_test_data(self, params: Dict) -> Tuple:
        """Generate test data based on parameters.
        This method should be implemented by the user to generate test data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def optimize(self, params: Dict) -> Dict[str, Any]:
        """Runs the optimization process."""
        if self.verbose:
            print("\n" + "="*60)
            print("Starting optimization process")
            print("="*60)
            print("Parameter bounds:")
            param_bounds = {
                'n_layers': (2, 5),
                'layer_size': (16, 128),
                'lr_adamw': (1e-4, 1e-2),
                'scheduler_factor': (0.1, 0.5),
                'scheduler_patience': (5, 20),
                'scheduler_min_lr': (1e-6, 1e-4),
                'scheduler_T_max': (50, 200),
                'scheduler_eta_min': (1e-6, 1e-4),
            }
            for name, bound in param_bounds.items():
                print(f"  {name:<20}: {bound}")
            print("-"*60)
        
        # Store training parameters
        self.train_params = params
        
        # Get parameter names and bounds
        param_names = self._get_parameter_names()
        bounds = [param_bounds[p] for p in param_names]
        
        # Create objective function
        def objective(x):
            return self.objective_function(self._decode_parameters(x))
        
        # Initialize and run the optimizer based on algorithm type
        if self.verbose:
            print(f"\nInitializing {self.algorithm_name.upper()} optimizer...")
        
        optimizer_class = self.ALGORITHMS[self.algorithm_name]
        
        # Create task dictionary
        task = {'name': self.study_name, 'func': objective, 'bounds': bounds, 'dim': len(bounds), 'budget': self.n_trials}
        
        try:
            optimizer = optimizer_class(task, **self.algorithm_params)
            
            if self.verbose:
                print("\nStarting optimization...")
                start_time = time.time()
            
            # Run optimization with or without timeout
            if self.timeout is not None:
                optimizer.run_minimize_with_timeout(self.timeout)
            else:
                optimizer.run_minimize()
            
            if self.verbose:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("\n" + "="*60)
                print("Optimization completed successfully!")
                print("="*60)
                print(f"Total time: {elapsed_time:.2f} seconds")
                print(f"Best objective value: {optimizer.best_objective_function:.2e}")
        
        except Exception as e:
            if self.verbose:
                print("\n" + "="*60)
                print("Optimization failed!")
                print(f"Error: {str(e)}")
                print("="*60)
            raise
        
        # Store results
        self.best_trial = {
            'value': optimizer.best_objective_function,
            'params': self._decode_parameters(optimizer.best_solution)
        }
        
        if self.verbose:
            print("\nBest parameters found:")
            print("-"*60)
            print("Architecture:")
            print(f"  Number of layers: {self.best_trial['params']['n_layers']}")
            print(f"  Layer sizes: {[self.best_trial['params'][f'layer_{i}_size'] for i in range(self.best_trial['params']['n_layers'])]}")
            print(f"  Activation: {self.best_trial['params']['activation']}")
            print("\nOptimizers:")
            print(f"  AdamW learning rate: {self.best_trial['params']['lr_adamw']:.2e}")
            print(f"\nScheduler type: {self.best_trial['params']['scheduler_type']}")
            if self.best_trial['params']['scheduler_type'] != 'none':
                print("  Parameters:")
                if self.best_trial['params']['scheduler_type'] == 'reduce_on_plateau':
                    print(f"    Factor: {self.best_trial['params']['scheduler_factor']:.2f}")
                    print(f"    Patience: {self.best_trial['params']['scheduler_patience']}")
                    print(f"    Min lr: {self.best_trial['params']['scheduler_min_lr']:.2e}")
                elif self.best_trial['params']['scheduler_type'] == 'cosine_annealing':
                    print(f"    T_max: {self.best_trial['params']['scheduler_T_max']}")
                    print(f"    Eta min: {self.best_trial['params']['scheduler_eta_min']:.2e}")
            print("-"*60)
        
        # Log results if logger is available
        if self.logger is not None:
            if self.verbose:
                print("\nSaving results to logger...")
            
            self.logger.log_optimizer_info(
                algorithm=self.algorithm_name,
                description=self._get_algorithm_description(self.algorithm_name),
                params=self.algorithm_params
            )
            
            
            # Выводим топ-10 лучших конфигураций
            self.logger.print_top_configurations(10)
            
            if self.verbose:
                print("Results saved successfully")
        
        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Returns the results of optimization.
        Override this method to customize the results format.
        """
        return {
            'best_error': self.best_trial['value'],
            'params': self.best_trial['params'],
            'optimization_history': getattr(self, 'optimization_history', []),
            'elapsed_time': getattr(self, 'elapsed_time', None)
        }

    def _get_parameter_names(self) -> List[str]:
        """Returns the list of parameter names for optimization."""
        return [
            'n_layers',
            'layer_size',
            'lr_adamw',
            'scheduler_factor',
            'scheduler_patience',
            'scheduler_min_lr',
            'scheduler_T_max',
            'scheduler_eta_min'
        ]

    def _decode_parameters(self, x: np.ndarray) -> Dict:
        """Decodes the optimizer's solution vector into parameter dictionary."""
        # Check input dimension
        expected_dim = len(self._get_parameter_names())
        if len(x) != expected_dim:
            raise ValueError(f"Expected {expected_dim} parameters, but got {len(x)}")
        
        try:
            params = {}
            
            # Basic parameters
            params['n_layers'] = max(2, min(5, int(round(x[0]))))
            for i in range(params['n_layers']):
                params[f'layer_{i}_size'] = max(16, min(128, int(round(x[1]))))
            
            params['activation'] = np.random.choice(['tanh', 'relu', 'gelu', 'silu'])
            params['lr_adamw'] = max(1e-4, min(1e-2, x[2]))
            
            # Scheduler parameters
            params['scheduler_type'] = np.random.choice(['reduce_on_plateau', 'cosine_annealing', 'none'])
            if params['scheduler_type'] == 'reduce_on_plateau':
                params['scheduler_factor'] = max(0.1, min(0.5, x[3]))
                params['scheduler_patience'] = max(5, min(20, int(round(x[4]))))
                params['scheduler_min_lr'] = max(1e-6, min(1e-4, x[5]))
            elif params['scheduler_type'] == 'cosine_annealing':
                params['scheduler_T_max'] = max(50, min(200, int(round(x[6]))))
                params['scheduler_eta_min'] = max(1e-6, min(1e-4, x[7]))
            
            return params
            
        except Exception as e:
            raise ValueError(f"Error decoding parameters: {str(e)}") from e 