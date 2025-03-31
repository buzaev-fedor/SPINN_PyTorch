import os
import time
import json
# import optuna
import numpy as np
from copy import deepcopy
from pathlib import Path
import csv
from datetime import datetime
import sys
import pandas as pd 

import matplotlib.pyplot as plt
from tqdm import trange
from typing import Sequence, List, Dict, Any, Optional, Tuple
from functools import partial
import torch.nn as nn
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Add sklearn for Gaussian Process implementation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

# Add parent directory to Python path to import bbo_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BBO algorithms and utilities
from bbo_utils import (JadeAlgorithm, LShadeAlgorithm, NelderMead, 
                      ParticleSwarmOptimization, GreyWolfOptimizer, 
                      WhaleOptimization, BlackBoxOptimizer, ResultLogger)

class SPINNArchitecture:
    def __init__(self, n_layers: int, features: List[int], activation: str):
        self.n_layers = n_layers
        self.features = features
        self.activation = activation
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_layers': self.n_layers,
            'features': self.features,
            'activation': self.activation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SPINNArchitecture':
        return cls(
            n_layers=data['n_layers'],
            features=data['features'],
            activation=data['activation']
        )

class SPINN(nn.Module):
    ACTIVATIONS = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanhshrink': nn.Tanhshrink,
        'Hardtanh': nn.Hardtanh,
    }
    
    def __init__(self, architecture: SPINNArchitecture):
        super().__init__()
        self.features = architecture.features
        self.activation_name = architecture.activation
        self.activation = self.ACTIVATIONS[architecture.activation]()
        
        # Создаем слои для каждого входа (t, x, y)
        self.networks = nn.ModuleList([
            self._build_network() for _ in range(3)
        ])
        
        # Добавляем слои для объединения выходов
        self.combine_layer1 = nn.Linear(self.features[-1] * 2, self.features[-1])
        self.combine_layer2 = nn.Linear(self.features[-1] * 2, self.features[-1])
        self.final_layer = nn.Linear(self.features[-1], 1)
    
    def _build_network(self):
        layers = []
        layers.append(nn.Linear(1, self.features[0]))
        layers.append(self.activation)
        
        for i in range(len(self.features) - 2):
            layers.append(nn.Linear(self.features[i], self.features[i + 1]))
            layers.append(self.activation)
            
        layers.append(nn.Linear(self.features[-2], self.features[-1]))
        layers.append(self.activation)
        return nn.Sequential(*layers)
    
    def _ensure_2d(self, x):
        if x.dim() == 1:
            return x.unsqueeze(1)
        return x
    
    def forward(self, t, x, y):
        # Преобразуем входы в 2D тензоры [batch_size, 1]
        t = self._ensure_2d(t)
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        
        # Пропускаем через отдельные сети
        t_features = self.networks[0](t)
        x_features = self.networks[1](x)
        y_features = self.networks[2](y)
        
        # Объединяем признаки
        combined = torch.cat([t_features, x_features], dim=1)
        combined = self.activation(self.combine_layer1(combined))
        
        combined = torch.cat([combined, y_features], dim=1)
        combined = self.activation(self.combine_layer2(combined))
        
        # Финальный слой
        output = self.final_layer(combined)
        return output.squeeze(-1)

# Функция для вычисления производных второго порядка
def compute_second_derivative(u, x):
    """Вычисляет вторую производную du/dx."""
    # Убеждаемся, что x требует градиентов
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # Вычисляем первую производную
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if du_dx is None:
        return torch.zeros_like(x)
    
    # Вычисляем вторую производную
    d2u_dx2 = torch.autograd.grad(
        du_dx, x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if d2u_dx2 is None:
        return torch.zeros_like(x)
    
    return d2u_dx2

class SPINN_Loss:
    def __init__(self, model):
        self.model = model

    def residual_loss(self, t, x, y, source_term):
        # Убеждаемся, что входные тензоры требуют градиентов
        if not t.requires_grad:
            t.requires_grad_(True)
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Получаем выход модели и убеждаемся, что он требует градиентов
        u = self.model(t, x, y)
        if not u.requires_grad:
            u.requires_grad_(True)
        
        # Вычисляем производные
        utt = compute_second_derivative(u, t)
        uxx = compute_second_derivative(u, x)
        uyy = compute_second_derivative(u, y)
        
        # Вычисляем невязку
        residual = utt - uxx - uyy + u**2 - source_term
        return torch.mean(residual**2)

    def initial_loss(self, t, x, y, u_true):
        u_pred = self.model(t, x, y)
        return torch.mean((u_pred - u_true)**2)

    def boundary_loss(self, t, x, y, u_true):
        loss = 0.
        for i in range(len(t)):
            u_pred = self.model(t[i], x[i], y[i])
            loss += torch.mean((u_pred - u_true[i])**2)
        return loss / len(t)

    def __call__(self, *train_data):
        tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
        
        loss_residual = self.residual_loss(tc, xc, yc, uc)
        loss_initial = self.initial_loss(ti, xi, yi, ui)
        loss_boundary = self.boundary_loss(tb, xb, yb, ub)
        
        return loss_residual + loss_initial + loss_boundary

# Функция шага оптимизации
def update_model(model, optimizer, train_data):
    optimizer.zero_grad()
    loss = spinn_loss_klein_gordon3d(model, *train_data)
    loss.backward()
    optimizer.step()
    return loss.item()


    # Точное решение уравнения Кляйна-Гордона
def _klein_gordon3d_exact_u(t, x, y):
    return (x + y) * torch.cos(2*t) + (x * y) * torch.sin(2*t)

# Источниковый член уравнения Кляйна-Гордона
def _klein_gordon3d_source_term(t, x, y):
    u = _klein_gordon3d_exact_u(t, x, y)
    return u**2 - 4*u

# Генератор тренировочных данных
def spinn_train_generator_klein_gordon3d(nc, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    # Точки коллокации
    tc = torch.rand(nc) * 10.0
    xc = torch.rand(nc) * 2.0 - 1.0
    yc = torch.rand(nc) * 2.0 - 1.0
    uc = _klein_gordon3d_source_term(tc, xc, yc)
    
    # Начальные точки
    ti = torch.zeros(nc)
    xi = torch.rand(nc) * 2.0 - 1.0
    yi = torch.rand(nc) * 2.0 - 1.0
    ui = _klein_gordon3d_exact_u(ti, xi, yi)
    
    # Граничные точки
    tb = [tc] * 4
    xb = [torch.full_like(tc, -1.0),
          torch.full_like(tc, 1.0),
          xc,
          xc]
    yb = [yc,
          yc,
          torch.full_like(tc, -1.0),
          torch.full_like(tc, 1.0)]
    
    ub = [_klein_gordon3d_exact_u(tb[i], xb[i], yb[i]) for i in range(4)]
    
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub

# Генератор тестовых данных
def spinn_test_generator_klein_gordon3d(nc_test):
    t = torch.linspace(0, 10, nc_test)
    x = torch.linspace(-1, 1, nc_test)
    y = torch.linspace(-1, 1, nc_test)
    
    tm, xm, ym = torch.meshgrid(t, x, y, indexing='ij')
    u_gt = _klein_gordon3d_exact_u(tm, xm, ym)
    
    return t, x, y, u_gt, tm, xm, ym


def relative_l2(u, u_gt):
    return torch.norm(u - u_gt) / torch.norm(u_gt)

def plot_klein_gordon3d(t, x, y, u, logger: Optional[ResultLogger] = None, name: str = "solution"):
    # Преобразуем тензоры PyTorch в numpy массивы для визуализации
    t = t.detach().cpu().numpy().flatten()
    x = x.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()
    u = u.detach().cpu().numpy().flatten()
    
    # Нормализуем значения для цветовой карты
    u_norm = (u - u.min()) / (u.max() - u.min())
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Создаем scatter plot с нормализованными цветами
    scatter = ax.scatter(t, x, y, c=u_norm, s=1, cmap='seismic', vmin=0, vmax=1)
    
    # Настраиваем внешний вид
    ax.set_title('U(t, x, y)', fontsize=20)
    ax.set_xlabel('t', fontsize=18, labelpad=10)
    ax.set_ylabel('x', fontsize=18, labelpad=10)
    ax.set_zlabel('y', fontsize=18, labelpad=10)
    
    # Добавляем colorbar с реальными значениями
    cbar = plt.colorbar(scatter)
    cbar.set_label('u(t,x,y)', fontsize=16)
    
    if logger is not None:
        logger.save_plot(fig, name)
    else:
        plt.savefig('klein_gordon3d.png', dpi=300, bbox_inches='tight')
        plt.close()

# Наследуем от базового класса BlackBoxOptimizer для решения нашей конкретной задачи Klein-Gordon
class KleinGordonOptimizer(BlackBoxOptimizer):
    def create_model(self, params: Dict) -> Tuple[SPINN, Dict]:
        """Creates a model with the given parameters."""
        # Extract architecture parameters
        n_layers = int(params['n_layers'])
        features = [int(params[f'layer_{i}_size']) for i in range(n_layers)]
        activation = params['activation']
        
        # Extract optimizer parameters
        lr_adamw = float(params['lr_adamw'])
        
        # Create architecture and model
        architecture = SPINNArchitecture(n_layers, features, activation)
        model = SPINN(architecture)
        
        optimizer_config = {
            'lr_adamw': lr_adamw,
            'scheduler_type': params.get('scheduler_type', 'none'),
            'scheduler_params': {}
        }
        
        if optimizer_config['scheduler_type'] == 'reduce_on_plateau':
            optimizer_config['scheduler_params'].update({
                'factor': float(params['scheduler_factor']),
                'patience': int(params['scheduler_patience']),
                'min_lr': float(params['scheduler_min_lr'])
            })
        elif optimizer_config['scheduler_type'] == 'cosine_annealing':
            optimizer_config['scheduler_params'].update({
                'T_max': int(params['scheduler_T_max']),
                'eta_min': float(params['scheduler_eta_min'])
            })
        
        return model, optimizer_config
    
    def generate_train_data(self, params: Dict) -> Tuple:
        """Generate training data for Klein-Gordon equation."""
        return spinn_train_generator_klein_gordon3d(params['NC'], seed=params['SEED'])
    
    def generate_test_data(self, params: Dict) -> Tuple:
        """Generate test data for Klein-Gordon equation."""
        return spinn_test_generator_klein_gordon3d(params['NC_TEST'])

    def train_model(self, model: SPINN, optimizer_params: Dict, train_data: Tuple, test_data: Tuple,
                   device: torch.device, n_epochs: int) -> float:
        """Trains the model and returns the validation error."""
        criterion = SPINN_Loss(model)
        
        # Setup optimizer
        optimizer = optim.AdamW(model.parameters(), lr=optimizer_params['lr_adamw'])
        
        # Setup scheduler if specified
        scheduler = None
        if optimizer_params['scheduler_type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=optimizer_params['scheduler_params']['factor'],
                patience=optimizer_params['scheduler_params']['patience'],
                min_lr=optimizer_params['scheduler_params']['min_lr']
            )
        elif optimizer_params['scheduler_type'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=optimizer_params['scheduler_params']['T_max'],
                eta_min=optimizer_params['scheduler_params']['eta_min']
            )
        
        best_error = float('inf')
        t, x, y, u_gt, tm, xm, ym = test_data
        
        # Очищаем историю обучения для нового trial
        self.training_history = []
        
        for epoch in range(1, n_epochs + 1):
            # AdamW step
            optimizer.zero_grad()
            tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
            
            # Ensure tensors require gradients
            if not tc.requires_grad:
                tc.requires_grad_(True)
            if not xc.requires_grad:
                xc.requires_grad_(True)
            if not yc.requires_grad:
                yc.requires_grad_(True)
            
            loss_residual = criterion.residual_loss(tc, xc, yc, uc)
            loss_initial = criterion.initial_loss(ti, xi, yi, ui)
            loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
            loss = loss_residual + loss_initial + loss_boundary
            loss.backward()
            optimizer.step()
            
            # Compute validation error
            with torch.no_grad():
                model.eval()
                u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
                error = relative_l2(u, u_gt.reshape(-1))
                
                if error < best_error:
                    best_error = error
                
                model.train()
            
            # Сохраняем историю обучения
            self.training_history.append({
                'epoch': int(epoch),
                'total_loss': float(loss.item()),
                'residual_loss': float(loss_residual.item()),
                'initial_loss': float(loss_initial.item()),
                'boundary_loss': float(loss_boundary.item()),
                'error': float(error.item()),
                'learning_rate': float(optimizer.param_groups[0]['lr'])
            })
            
            # Update scheduler if using ReduceLROnPlateau
            if scheduler is not None and optimizer_params['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(error)
        
        # Update cosine annealing scheduler
        if scheduler is not None and optimizer_params['scheduler_type'] == 'cosine_annealing':
            scheduler.step()
        
        # Final validation error
        with torch.no_grad():
            model.eval()
            u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
            final_error = relative_l2(u, u_gt.reshape(-1))
        
        return min(best_error, final_error)
    
    def get_results(self) -> Dict[str, Any]:
        """Override to return SPINN-specific results structure."""
        return {
            'architecture': SPINNArchitecture(
                n_layers=self.best_trial['params']['n_layers'],
                features=[self.best_trial['params'][f'layer_{i}_size'] for i in range(self.best_trial['params']['n_layers'])],
                activation=self.best_trial['params']['activation']
            ),
            'lr_adamw': self.best_trial['params']['lr_adamw'],
            'best_error': self.best_trial['value'],
            'optimization_history': getattr(self, 'optimization_history', []),
            'elapsed_time': getattr(self, 'elapsed_time', None)
        }

class GaussianProcessOptimizer:
    """
    Gaussian Process-based Bayesian Optimization algorithm that models the objective function
    and suggests promising configurations to evaluate.
    """
    def __init__(self, param_bounds, initial_points=5, 
                 acquisition_function='EI', random_state=None):
        """
        Initialize the Gaussian Process optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names to (lower, upper) bounds
            initial_points: Number of random initial points to evaluate
            acquisition_function: Acquisition function ('EI', 'UCB', or 'PI')
            random_state: Random state for reproducibility
        """
        self.param_bounds = param_bounds
        self.initial_points = initial_points
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        
        # Set up GP kernel with Matérn kernel (smoothness parameter = 2.5)
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            random_state=random_state
        )
        
        # Initialize history
        self.X_observed = []  # Observed parameters
        self.y_observed = []  # Observed objective values
        self.best_params = None
        self.best_value = float('inf')
        
    def _normalize_params(self, params):
        """Normalize parameters to [0, 1] range for GP."""
        normalized = {}
        for name, value in params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                normalized[name] = (value - low) / (high - low)
        return normalized
    
    def _denormalize_params(self, normalized_params):
        """Convert normalized parameters back to original range."""
        params = {}
        for name, value in normalized_params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                params[name] = low + value * (high - low)
        return params
    
    def _dict_to_array(self, param_dict):
        """Convert parameter dictionary to array for GP model."""
        return np.array([param_dict[name] for name in sorted(self.param_bounds.keys())])
    
    def _array_to_dict(self, param_array):
        """Convert parameter array back to dictionary."""
        return {name: param_array[i] for i, name in enumerate(sorted(self.param_bounds.keys()))}
    
    def _acquisition(self, x, gp, best_f):
        """
        Acquisition function to determine next points to sample.
        
        Args:
            x: Parameters to evaluate
            gp: Gaussian Process model
            best_f: Best observed value so far
            
        Returns:
            Acquisition function value (higher is better)
        """
        x = x.reshape(1, -1)
        mean, std = gp.predict(x, return_std=True)
        
        if self.acquisition_function == 'UCB':
            # Upper Confidence Bound
            kappa = 2.576  # 99% confidence
            return mean + kappa * std
        
        elif self.acquisition_function == 'PI':
            # Probability of Improvement
            z = (mean - best_f) / (std + 1e-9)
            return norm.cdf(z)
        
        else:  # Default: Expected Improvement
            # Expected Improvement
            improvement = mean - best_f
            z = improvement / (std + 1e-9)
            return improvement * norm.cdf(z) + std * norm.pdf(z)
    
    def _next_sample(self):
        """Generate next sample point by maximizing acquisition function."""
        dim = len(self.param_bounds)
        
        # If we don't have enough samples yet, return random point
        if len(self.X_observed) < self.initial_points:
            return {name: np.random.uniform(low, high) 
                   for name, (low, high) in self.param_bounds.items()}
        
        # Fit GP model
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # Define bounds for optimization
        bounds = [(0, 1) for _ in range(dim)]
        
        # Use multiple random starts to avoid local optima
        best_x = None
        best_acq = -float('inf')
        
        for _ in range(10):
            x0 = np.random.uniform(0, 1, dim)
            result = minimize(
                lambda x: -self._acquisition(x, self.gp, min(self.y_observed)),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        # Convert to parameter dictionary
        return self._denormalize_params(self._array_to_dict(best_x))
    
    def suggest_params(self):
        """Suggest next set of parameters to evaluate."""
        return self._next_sample()
    
    def update(self, params, value):
        """
        Update the model with new observation.
        
        Args:
            params: Dictionary of parameters
            value: Observed objective value
        """
        # Normalize parameters
        norm_params = self._normalize_params(params)
        
        # Convert to array
        x = self._dict_to_array(norm_params)
        
        # Update observations
        self.X_observed.append(x)
        self.y_observed.append(value)
        
        # Update best value
        if value < self.best_value:
            self.best_value = value
            self.best_params = params

# Add GPBlackBoxOptimizer that integrates with the existing framework
class GPBlackBoxOptimizer(BlackBoxOptimizer):
    """
    Black Box Optimizer that uses Gaussian Process to guide the optimization.
    """
    
    def _get_algorithm_description(self, algorithm):
        """Get the description of the optimization algorithm."""
        if algorithm == 'gp':
            return "Gaussian Process-based Bayesian optimization"
        return super()._get_algorithm_description(algorithm)
    
    def _setup_algorithm(self):
        """Set up the optimization algorithm."""
        if self.algorithm == 'gp':
            # Extract algorithm parameters
            acq_function = self.algorithm_params.get('acquisition_function', 'EI')
            initial_points = self.algorithm_params.get('initial_points', 5)
            
            # Define parameter bounds for GP
            param_bounds = self._get_param_bounds()
            
            # Create GP optimizer
            self.gp_optimizer = GaussianProcessOptimizer(
                param_bounds=param_bounds,
                initial_points=initial_points,
                acquisition_function=acq_function,
                random_state=42  # Fixed for reproducibility
            )
        else:
            super()._setup_algorithm()
    
    def _get_param_bounds(self):
        """Get parameter bounds for the GP optimizer."""
        bounds = {
            'n_layers': (2, 5),
            'lr_adamw': (1e-5, 1e-2)
        }
        
        # Add layer size bounds
        for i in range(5):  # Up to 5 layers
            bounds[f'layer_{i}_size'] = (8, 64)
        
        return bounds
    
    def optimize(self, params: Dict) -> Dict[str, Any]:
        """Run optimization using GP optimizer."""
        if self.algorithm != 'gp':
            return super().optimize(params)
        
        self._setup_algorithm()
        start_time = time.time()
        
        # Track optimization history
        self.optimization_history = []
        
        for trial in range(self.n_trials):
            if self.timeout and time.time() - start_time > self.timeout:
                print(f"Timeout reached after {trial} trials")
                break
            
            print(f"\nTrial {trial+1}/{self.n_trials}")
            
            # Get next parameters to evaluate
            trial_params = self.gp_optimizer.suggest_params()
            
            # Convert numeric params to appropriate types
            processed_params = {}
            processed_params['n_layers'] = int(round(trial_params['n_layers']))
            processed_params['lr_adamw'] = float(trial_params['lr_adamw'])
            
            # Set layer sizes
            for i in range(processed_params['n_layers']):
                processed_params[f'layer_{i}_size'] = int(round(trial_params[f'layer_{i}_size']))
            
            # Set fixed parameters
            processed_params['activation'] = 'tanh'  # Default activation
            processed_params['scheduler_type'] = 'none'  # Default scheduler
            
            # Evaluate model with these parameters
            error = self._evaluate_trial(processed_params, params)
            
            # Update GP model
            self.gp_optimizer.update(trial_params, error)
            
            # Track history
            self.optimization_history.append({
                'trial': trial,
                'params': processed_params,
                'error': error
            })
            
            # Update best trial
            if error < self.best_value:
                self.best_value = error
                self.best_trial = {
                    'params': processed_params,
                    'value': error
                }
                
                # Log progress
                if self.verbose:
                    print(f"New best error: {error:.4e}")
                    print(f"Parameters: {processed_params}")
        
        self.elapsed_time = time.time() - start_time
        return self.get_results()

class NestedGPOptimizer(BlackBoxOptimizer):
    """
    Nested optimization approach: 
    1. Outer optimizer (JADE, LSHADE, etc.) optimizes GP hyperparameters
    2. GP model optimizes SPINN/PINN parameters
    """
    
    def __init__(self, n_trials, timeout, study_name, logger, algorithm, algorithm_params, verbose=False,
                 gp_trials=25):
        """
        Initialize the nested optimizer.
        
        Args:
            n_trials: Number of trials for the outer optimizer
            timeout: Timeout for the entire optimization
            study_name: Name of the study
            logger: Result logger
            algorithm: Outer optimization algorithm
            algorithm_params: Parameters for the outer algorithm
            verbose: Whether to print progress
            gp_trials: Number of trials for the GP optimizer (inner loop)
        """
        super().__init__(n_trials, timeout, study_name, logger, algorithm, algorithm_params, verbose)
        self.gp_trials = gp_trials
        self.best_gp_params = None
        self.best_pinn_params = None
        self.best_pinn_error = float('inf')
        # Сохраняем алгоритм отдельно для использования в _setup_outer_algorithm
        self.outer_algorithm = algorithm
        # Создаем экземпляр KleinGordonOptimizer для делегирования методов
        self.kg_optimizer = KleinGordonOptimizer(
            n_trials=1,  # Не имеет значения для наших целей
            timeout=None,
            study_name=study_name,
            logger=logger,
            algorithm="jade",  # Не имеет значения для наших целей
            algorithm_params={},
            verbose=verbose
        )
        
    def _create_optimization_task(self, dimensions, bounds):
        """
        Создает совместимую "задачу" для различных алгоритмов оптимизации.
        
        Args:
            dimensions: количество измерений пространства параметров
            bounds: ограничения для каждого измерения
            
        Returns:
            task: словарь, совместимый с интерфейсом Optimizer
        """
        # Функция для оптимизации - мы минимизируем ошибку PINN
        # Эта функция будет фактически реализована в методе safe_tell
        def dummy_func(x):
            return 0.0  # Заглушка
            
        # Создаем словарь с необходимыми ключами
        task = {
            'name': 'gp_hyperparameter_optimization',
            'func': dummy_func,
            'bounds': bounds,
            'budget': self.n_trials * 10,  # Бюджет вычислений
            'dimensions': dimensions,
            'lower_bound': np.array([b[0] for b in bounds]),
            'upper_bound': np.array([b[1] for b in bounds])
        }
        
        return task
    
    def create_model(self, params: Dict) -> Tuple[SPINN, Dict]:
        """Делегирует создание модели KleinGordonOptimizer."""
        return self.kg_optimizer.create_model(params)
    
    def generate_train_data(self, params: Dict) -> Tuple:
        """Делегирует генерацию тренировочных данных KleinGordonOptimizer."""
        return self.kg_optimizer.generate_train_data(params)
    
    def generate_test_data(self, params: Dict) -> Tuple:
        """Делегирует генерацию тестовых данных KleinGordonOptimizer."""
        return self.kg_optimizer.generate_test_data(params)
    
    def train_model(self, model: SPINN, optimizer_params: Dict, train_data: Tuple, test_data: Tuple,
                   device: torch.device, n_epochs: int) -> float:
        """Делегирует тренировку модели KleinGordonOptimizer и сохраняет историю обучения."""
        error = self.kg_optimizer.train_model(model, optimizer_params, train_data, test_data,
                                           device, n_epochs)
        
        # Получаем историю обучения из KleinGordonOptimizer
        if hasattr(self.kg_optimizer, 'training_history'):
            # Копируем историю обучения для текущего trial
            self.training_history = deepcopy(self.kg_optimizer.training_history)
            
            # Логируем каждую эпоху, если есть логгер
            if self.logger and len(self.training_history) > 0:
                # Логируем каждый 10-й шаг, чтобы не перегружать логи
                for i, entry in enumerate(self.training_history):
                    if i % 10 == 0 or i == len(self.training_history) - 1:
                        self.logger.log_training(
                            entry['epoch'],
                            {
                                'total': entry['total_loss'],
                                'residual': entry['residual_loss'],
                                'initial': entry['initial_loss'],
                                'boundary': entry['boundary_loss']
                            },
                            entry['error'],
                            'inner_trial'
                        )
        
        return error
        
    def _evaluate_trial(self, trial_params, problem_params):
        """
        Создает модель с заданными параметрами и оценивает ее производительность.
        
        Args:
            trial_params: Параметры для создания и тренировки модели
            problem_params: Общие параметры задачи (NC, SEED и т.д.)
            
        Returns:
            float: Ошибка модели
        """
        # Create model and optimizer configuration
        model, optimizer_config = self.create_model(trial_params)
        
        # Move model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Generate training and test data
        train_data = self.generate_train_data(problem_params)
        test_data = self.generate_test_data(problem_params)
        
        # Move data to device
        train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                    [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                    for t in train_data]
        
        test_data = list(test_data)
        test_data = [t.to(device) if isinstance(t, torch.Tensor) else t for t in test_data]
        
        # Train model for specified number of epochs
        error = self.train_model(
            model=model,
            optimizer_params=optimizer_config,
            train_data=train_data,
            test_data=test_data,
            device=device,
            n_epochs=problem_params.get('EPOCHS', 1000)
        )
        
        return error
        
    def _setup_outer_algorithm(self):
        """Set up the outer optimization algorithm."""
        gp_param_bounds = self._get_gp_param_bounds_list()
        dimensions = len(self._get_gp_param_bounds())
        
        # Создаем задачу для оптимизаторов
        task = self._create_optimization_task(dimensions, gp_param_bounds)
        
        if self.outer_algorithm == 'jade':
            # Создаем JadeAlgorithm с нашей задачей
            self.outer_optimizer = JadeAlgorithm(
                task=task,
                population_size=self.algorithm_params.get('population_size', 100),
                c=self.algorithm_params.get('c', 0.1),
                p=self.algorithm_params.get('p', 0.05)
            )
        elif self.outer_algorithm == 'lshade':
            # Создаем LShadeAlgorithm с нашей задачей
            self.outer_optimizer = LShadeAlgorithm(
                task=task,
                population_size=self.algorithm_params.get('population_size', 100)
            )
        elif self.outer_algorithm == 'nelder_mead':
            # Создаем копию параметров, исключая специальные параметры
            nm_params = {k: v for k, v in self.algorithm_params.items() 
                        if not k.startswith('_')}
            
            # Выводим отладочную информацию
            print(f"DEBUG: NelderMead initialization parameters: {nm_params}")
            
            # Используем нашу реализацию CustomNelderMead вместо исходной
            try:
                # Создаем экземпляр CustomNelderMead
                self.outer_optimizer = CustomNelderMead(**nm_params)
                print("Successfully created CustomNelderMead instance")
                
                # Инициализируем алгоритм
                dims = dimensions
                bounds = gp_param_bounds
                print(f"Initializing CustomNelderMead with dims={dims}, bounds={bounds}")
                self.outer_optimizer.initialize(dims, bounds)
                print("CustomNelderMead initialized successfully")
            except Exception as e:
                print(f"ERROR initializing CustomNelderMead: {str(e)}")
                # Временное решение - используем другой алгоритм
                print("Falling back to JADE algorithm as a temporary solution")
                self.outer_optimizer = JadeAlgorithm(
                    task=task,
                    population_size=self.algorithm_params.get('population_size', 50),
                    c=self.algorithm_params.get('c', 0.1),
                    p=self.algorithm_params.get('p', 0.05)
                )
        elif self.outer_algorithm == 'pso':
            # Создаем PSO с нашей задачей
            self.outer_optimizer = ParticleSwarmOptimization(
                task=task,
                num_particles=self.algorithm_params.get('num_particles', 100),
                w=self.algorithm_params.get('w', 0.5),
                c1=self.algorithm_params.get('c1', 1.0),
                c2=self.algorithm_params.get('c2', 1.0)
            )
        elif self.outer_algorithm == 'grey_wolf':
            # Создаем GreyWolfOptimizer с нашей задачей
            self.outer_optimizer = GreyWolfOptimizer(
                task=task,
                num_wolves=self.algorithm_params.get('num_wolves', 100)
            )
        elif self.outer_algorithm == 'whales':
            # Создаем WhaleOptimization с нашей задачей
            self.outer_optimizer = WhaleOptimization(
                task=task,
                population_size=self.algorithm_params.get('population_size', 100)
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.outer_algorithm}")
    
    def _get_gp_param_bounds(self):
        """Get parameter bounds for the GP hyperparameters."""
        # Define the parameters to optimize for the GP itself
        return {
            'constant_value': (0.1, 10.0),         # ConstantKernel value
            'length_scale': (0.01, 10.0),          # Matérn kernel length scale
            'nu': (0.5, 2.5),                      # Matérn kernel smoothness (0.5, 1.5, 2.5)
            'noise_level': (1e-10, 1.0),           # GP noise level
            'n_restarts_optimizer': (1, 20),       # Number of restarts for GP optimizer
            'initial_points': (3, 15)              # Number of initial random points
        }
    
    def _get_gp_param_bounds_list(self):
        """Convert parameter bounds dictionary to list of tuples for algorithms."""
        bounds_dict = self._get_gp_param_bounds()
        return [(low, high) for _, (low, high) in sorted(bounds_dict.items())]
    
    def _get_pinn_param_bounds(self):
        """Get parameter bounds for the PINN parameters."""
        bounds = {
            'n_layers': (2, 5),
            'lr_adamw': (1e-5, 1e-2)
        }
        
        # Add layer size bounds
        for i in range(5):  # Up to 5 layers
            bounds[f'layer_{i}_size'] = (8, 64)
        
        return bounds
    
    def _dict_to_array(self, param_dict, bounds_dict):
        """Convert parameter dictionary to array for the optimizer."""
        return np.array([param_dict[name] for name in sorted(bounds_dict.keys())])
    
    def _array_to_dict(self, param_array, bounds_dict):
        """Convert parameter array back to dictionary."""
        result = {}
        
        # Проверяем, не является ли param_array None
        if param_array is None:
            # Если None, генерируем случайные значения из допустимого диапазона
            print("WARNING: param_array is None, generating random parameters")
            for name, (low, high) in bounds_dict.items():
                result[name] = np.random.uniform(low, high)
            return result
            
        for i, name in enumerate(sorted(bounds_dict.keys())):
            # Преобразуем numpy.ndarray в обычный Python тип для безопасного использования
            if isinstance(param_array[i], np.ndarray):
                value = float(param_array[i])
            else:
                value = param_array[i]
            result[name] = value
        return result
    
    def _create_gp_from_params(self, params):
        """Create a GP with the given hyperparameters."""
        constant_kernel = ConstantKernel(constant_value=params['constant_value'])
        matern_kernel = Matern(
            length_scale=params['length_scale'],
            nu=params['nu']
        )
        kernel = constant_kernel * matern_kernel
        
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=params['noise_level'],
            n_restarts_optimizer=int(round(params['n_restarts_optimizer'])),
            random_state=42
        )
    
    def _run_gp_optimization(self, gp_params, problem_params):
        """Run GP optimization with the given GP hyperparameters."""
        # Create GP optimizer with the given parameters
        param_bounds = self._get_pinn_param_bounds()
        initial_points = int(round(gp_params['initial_points']))
        
        gp = self._create_gp_from_params(gp_params)
        
        # Create a GaussianProcessOptimizer configured with our parameters
        gp_optimizer = GaussianProcessOptimizer(
            param_bounds=param_bounds,
            initial_points=initial_points,
            acquisition_function='EI',  # Используем Expected Improvement по умолчанию
            random_state=42  # Fixed for reproducibility
        )
        # Заменяем GP-объект на созданный с нашими параметрами
        gp_optimizer.gp = gp
        
        # Track optimization history for this GP run
        history = []
        best_error = float('inf')
        best_params = None
        
        # Создаем прогресс-бар для внутренних итераций GP
        gp_pbar = trange(self.gp_trials, desc="GP Optimization Progress")
        
        # Печатаем детали параметров в более читаемом формате
        print("\n" + "-"*50)
        print("GP Hyperparameters:")
        print(f"  Constant kernel value:  {gp_params['constant_value']:.2f}")
        print(f"  Matern length scale:    {gp_params['length_scale']:.2f}")
        print(f"  Matern smoothness (nu): {gp_params['nu']:.2f}")
        print(f"  Noise level:            {gp_params['noise_level']:.2e}")
        print(f"  GP optimizer restarts:   {int(round(gp_params['n_restarts_optimizer']))}")
        print(f"  Initial random points:   {int(round(gp_params['initial_points']))}")
        print("-"*50 + "\n")
        
        # Run optimization for a fixed number of trials
        for trial in gp_pbar:
            # Get next parameters to evaluate
            trial_params = gp_optimizer.suggest_params()
            
            # Convert numeric params to appropriate types
            processed_params = {}
            processed_params['n_layers'] = int(round(trial_params['n_layers']))
            processed_params['lr_adamw'] = float(trial_params['lr_adamw'])
            
            # Set layer sizes
            for i in range(processed_params['n_layers']):
                processed_params[f'layer_{i}_size'] = int(round(trial_params[f'layer_{i}_size']))
            
            # Set fixed parameters
            processed_params['activation'] = 'tanh'  # Default activation
            processed_params['scheduler_type'] = 'none'  # Default scheduler
            
            # Evaluate model with these parameters
            error = self._evaluate_trial(processed_params, problem_params)
            
            # Update GP model
            gp_optimizer.update(trial_params, error)
            
            # Track history
            history.append({
                'trial': trial,
                'params': processed_params,
                'error': error
            })
            
            # Update best parameters
            if error < best_error:
                best_error = error
                best_params = processed_params
                
                # Обновляем прогресс-бар с информацией о лучшей ошибке
                gp_pbar.set_description(f"GP Opt [Best Error: {error:.4e}]")
                
                if self.verbose:
                    print(f"  [GP] New best error: {error:.4e}")
                    print(f"  [GP] Parameters: n_layers={processed_params['n_layers']}, "
                          f"lr={processed_params['lr_adamw']:.2e}")
                
                # Логируем каждую эпоху, если есть улучшение
                if self.logger:
                    self.logger.log_inner_gp_trial(
                        outer_trial_idx=getattr(self, 'current_outer_trial', 0),
                        inner_trial_idx=trial,
                        gp_params=gp_params,
                        pinn_params=processed_params,
                        error=error,
                        is_best=True
                    )
            else:
                # Логируем каждую эпоху в любом случае
                if self.logger and trial % 5 == 0:  # Логируем каждую 5-ю итерацию, чтобы не переполнять логи
                    self.logger.log_inner_gp_trial(
                        outer_trial_idx=getattr(self, 'current_outer_trial', 0),
                        inner_trial_idx=trial,
                        gp_params=gp_params,
                        pinn_params=processed_params,
                        error=error,
                        is_best=False
                    )
        
        return best_error, best_params, history
    
    def optimize(self, problem_params: Dict) -> Dict[str, Any]:
        """Run nested optimization process."""
        self._setup_outer_algorithm()
        start_time = time.time()
        
        # Track optimization history
        self.optimization_history = []
        
        # Создаем прогресс-бар для внешних итераций 
        outer_pbar = trange(self.n_trials, desc="Outer Optimization Progress")
        
        # Обеспечиваем совместимость с различными интерфейсами оптимизаторов
        def safe_ask(optimizer):
            """Безопасный вызов метода ask для получения следующих параметров."""
            try:
                # Проверяем, какие методы есть у оптимизатора для получения параметров
                if hasattr(optimizer, 'ask'):
                    # Стандартный интерфейс ask
                    return optimizer.ask()
                elif hasattr(optimizer, 'get_next_parameters'):
                    # Альтернативный метод
                    return optimizer.get_next_parameters()
                elif hasattr(optimizer, 'suggest'):
                    # Еще один альтернативный метод
                    return optimizer.suggest()
                elif hasattr(optimizer, 'initialize_population') and hasattr(optimizer, 'population'):
                    # Проверяем, есть ли метод initialize_population - характерно для JADE и подобных
                    # Если популяция еще не инициализирована, инициализируем её
                    if not hasattr(optimizer, 'population') or len(getattr(optimizer, 'population', [])) == 0:
                        # Получаем границы параметров
                        bounds_list = self._get_gp_param_bounds_list()
                        lower_bound = np.array([b[0] for b in bounds_list])
                        upper_bound = np.array([b[1] for b in bounds_list])
                        optimizer.initialize_population(lower_bound, upper_bound)
                    
                    # Возвращаем первого индивида из популяции или случайные параметры
                    if hasattr(optimizer, 'population') and len(optimizer.population) > 0:
                        return optimizer.population[0]
                
                # Если ничего не сработало, генерируем случайные параметры
                print("WARNING: No suitable method found for parameters generation, using random parameters")
                bounds_list = self._get_gp_param_bounds_list()
                return np.array([np.random.uniform(low, high) for (low, high) in bounds_list])
                
            except Exception as e:
                print(f"ERROR in safe_ask: {str(e)}")
                # В случае ошибки возвращаем случайные параметры
                bounds_list = self._get_gp_param_bounds_list()
                return np.array([np.random.uniform(low, high) for (low, high) in bounds_list])
        
        def safe_tell(optimizer, params, value):
            """Безопасный вызов метода tell для обновления оптимизатора."""
            try:
                # Конвертируем CUDA тензоры в CPU перед использованием
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                    
                # Если params - это тензор, тоже конвертируем его
                if torch.is_tensor(params):
                    params = params.detach().cpu().numpy()
                elif isinstance(params, (list, tuple)) and any(torch.is_tensor(p) for p in params):
                    params = [p.detach().cpu().numpy() if torch.is_tensor(p) else p for p in params]
                
                # Проверяем интерфейс обновления
                if hasattr(optimizer, 'tell'):
                    optimizer.tell(params, value)
                elif hasattr(optimizer, 'update'):
                    optimizer.update(params, value)
                elif hasattr(optimizer, 'report_result'):
                    optimizer.report_result(params, value)
                elif hasattr(optimizer, 'fitness_function'):
                    # Для JADE и других оптимизаторов устанавливаем лучшее найденное решение
                    # так как они обновляют состояние в методе minimize()
                    if not hasattr(optimizer, 'best_solution') or value < optimizer.best_objective_function:
                        optimizer.best_solution = params
                        optimizer.best_objective_function = value
                        # Добавляем историю для совместимости
                        optimizer.objective_function_history.append(value)
                else:
                    print("WARNING: No tell/update method found. Optimization might not work correctly.")
            except Exception as e:
                print(f"ERROR in tell: {str(e)}")
                # Добавляем более подробный вывод для отладки
                print(f"Type of params: {type(params)}")
                print(f"Type of value: {type(value)}")
        
        for trial in outer_pbar:
            self.current_outer_trial = trial  # Сохраняем текущую итерацию для логирования
            
            if self.timeout and time.time() - start_time > self.timeout:
                print(f"Timeout reached after {trial} trials")
                break
            
            print(f"\nOuter Trial {trial+1}/{self.n_trials}")
            
            # Get next set of GP hyperparameters from outer optimizer
            gp_param_array = safe_ask(self.outer_optimizer)
            gp_params = self._array_to_dict(gp_param_array, self._get_gp_param_bounds())
            
            if self.verbose:
                print(f"Testing GP parameters: {gp_params}")
            
            # Run full GP optimization with these hyperparameters
            print(f"Running GP optimization with {self.gp_trials} trials...")
            best_error, best_params, history = self._run_gp_optimization(gp_params, problem_params)
            
            # Tell the outer optimizer the result
            safe_tell(self.outer_optimizer, gp_param_array, best_error)
            
            # Обновляем прогресс-бар внешней оптимизации
            outer_pbar.set_description(f"Outer Opt [Best Error: {self.best_pinn_error if self.best_pinn_error != float('inf') else 'N/A'}]")
            
            # Track this outer trial
            self.optimization_history.append({
                'outer_trial': trial,
                'gp_params': gp_params,
                'best_pinn_error': best_error,
                'best_pinn_params': best_params,
                'inner_history': history
            })
            
            # Update best overall result
            if best_error < self.best_pinn_error:
                self.best_pinn_error = best_error
                self.best_pinn_params = best_params
                self.best_gp_params = gp_params
                
                print(f"\n{'-'*50}")
                print(f"NEW BEST PINN MODEL:")
                print(f"  Error: {best_error:.4e}")
                print(f"  Architecture: {best_params['n_layers']} layers, sizes: {[best_params[f'layer_{i}_size'] for i in range(best_params['n_layers'])]}")
                print(f"  Learning rate: {best_params['lr_adamw']:.2e}")
                print(f"  Activation: {best_params['activation']}")
                print(f"{'-'*50}\n")
                
                # Логируем лучший результат внешней оптимизации
                if self.logger:
                    self.logger.log_outer_trial(
                        trial_idx=trial,
                        gp_params=gp_params,
                        best_pinn_error=best_error,
                        best_pinn_params=best_params,
                        elapsed_time=time.time() - start_time
                    )
            else:
                # Логируем результат внешней оптимизации в любом случае
                if self.logger:
                    self.logger.log_outer_trial(
                        trial_idx=trial,
                        gp_params=gp_params,
                        best_pinn_error=best_error,
                        best_pinn_params=best_params,
                        elapsed_time=time.time() - start_time,
                        is_best=False
                    )
        
        # Store best trial information
        self.best_trial = {
            'params': self.best_pinn_params,
            'value': self.best_pinn_error
        }
        self.best_value = self.best_pinn_error
        
        self.elapsed_time = time.time() - start_time
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Override to return SPINN-specific results structure with GP info."""
        return {
            'architecture': SPINNArchitecture(
                n_layers=self.best_trial['params']['n_layers'],
                features=[self.best_trial['params'][f'layer_{i}_size'] for i in range(self.best_trial['params']['n_layers'])],
                activation=self.best_trial['params']['activation']
            ),
            'lr_adamw': self.best_trial['params']['lr_adamw'],
            'best_error': self.best_trial['value'],
            'best_gp_params': self.best_gp_params,
            'optimization_history': getattr(self, 'optimization_history', []),
            'elapsed_time': getattr(self, 'elapsed_time', None)
        }

# Custom GP optimizer that allows providing a pre-configured GP
class CustomGaussianProcessOptimizer:
    """
    Gaussian Process-based Bayesian Optimization algorithm with configurable GP.
    """
    def __init__(self, param_bounds, initial_points=5, gp=None, random_state=None):
        """
        Initialize the Gaussian Process optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names to (lower, upper) bounds
            initial_points: Number of random initial points to evaluate
            gp: Pre-configured Gaussian Process (if None, a default one will be created)
            random_state: Random state for reproducibility
        """
        self.param_bounds = param_bounds
        self.initial_points = initial_points
        self.random_state = random_state
        
        # Use provided GP or create default
        if gp is not None:
            self.gp = gp
        else:
            # Default GP configuration
            self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=10,
                random_state=random_state
            )
        
        # Initialize history
        self.X_observed = []  # Observed parameters
        self.y_observed = []  # Observed objective values
        self.best_params = None
        self.best_value = float('inf')
        
    def _normalize_params(self, params):
        """Normalize parameters to [0, 1] range for GP."""
        normalized = {}
        for name, value in params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                normalized[name] = (value - low) / (high - low)
        return normalized
    
    def _denormalize_params(self, normalized_params):
        """Convert normalized parameters back to original range."""
        params = {}
        for name, value in normalized_params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                params[name] = low + value * (high - low)
        return params
    
    def _dict_to_array(self, param_dict):
        """Convert parameter dictionary to array for GP model."""
        return np.array([param_dict[name] for name in sorted(self.param_bounds.keys())])
    
    def _array_to_dict(self, param_array):
        """Convert parameter array back to dictionary."""
        return {name: param_array[i] for i, name in enumerate(sorted(self.param_bounds.keys()))}
    
    def _acquisition(self, x, gp, best_f):
        """Expected Improvement acquisition function."""
        x = x.reshape(1, -1)
        mean, std = gp.predict(x, return_std=True)
        
        # Expected Improvement
        improvement = best_f - mean  # Note: we're minimizing, so flip the sign
        z = improvement / (std + 1e-9)
        return improvement * norm.cdf(z) + std * norm.pdf(z)
    
    def _next_sample(self):
        """Generate next sample point by maximizing acquisition function."""
        dim = len(self.param_bounds)
        
        # If we don't have enough samples yet, return random point
        if len(self.X_observed) < self.initial_points:
            return {name: np.random.uniform(low, high) 
                   for name, (low, high) in self.param_bounds.items()}
        
        # Fit GP model
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # Define bounds for optimization
        bounds = [(0, 1) for _ in range(dim)]
        
        # Use multiple random starts to avoid local optima
        best_x = None
        best_acq = -float('inf')
        
        for _ in range(10):
            x0 = np.random.uniform(0, 1, dim)
            result = minimize(
                lambda x: -self._acquisition(x, self.gp, min(self.y_observed)),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        # Convert to parameter dictionary
        return self._denormalize_params(self._array_to_dict(best_x))
    
    def suggest_params(self):
        """Suggest next set of parameters to evaluate."""
        return self._next_sample()
    
    def update(self, params, value):
        """
        Update the model with new observation.
        
        Args:
            params: Dictionary of parameters
            value: Observed objective value
        """
        # Normalize parameters
        norm_params = self._normalize_params(params)
        
        # Convert to array
        x = self._dict_to_array(norm_params)
        
        # Update observations
        self.X_observed.append(x)
        self.y_observed.append(value)
        
        # Update best value
        if value < self.best_value:
            self.best_value = value
            self.best_params = params

class CustomNelderMead:
    """
    Простая реализация алгоритма Нелдера-Мида, которая работает независимо от интерфейса оригинального класса.
    Реализует интерфейс ask/tell для совместимости с остальной кодовой базой.
    """
    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, **kwargs):
        self.alpha = alpha  # reflection
        self.gamma = gamma  # expansion
        self.rho = rho      # contraction
        self.sigma = sigma  # shrink
        
        self.dimensions = None
        self.bounds = None
        self.population = None
        self.values = None
        self.best_point = None
        self.best_value = float('inf')
        self.iteration = 0
        
    def initialize(self, dimensions, bounds):
        """Initialize the algorithm with dimensions and bounds."""
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Create initial simplex (n+1 points for n dimensions)
        self.population = []
        self.values = []
        
        # Start with random point within bounds
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.population.append(x0)
        
        # Add n more points by perturbing each dimension
        for i in range(dimensions):
            x = x0.copy()
            # Perturbation proportional to the bounds
            delta = 0.05 * (bounds[i][1] - bounds[i][0])
            x[i] += delta
            # Ensure we stay within bounds
            x[i] = min(max(x[i], bounds[i][0]), bounds[i][1])
            self.population.append(x)
        
        # Initialize values to infinity (will be updated in tell)
        self.values = [float('inf')] * (dimensions + 1)
    
    def ask(self):
        """Return the next point to evaluate."""
        if self.population is None:
            raise ValueError("Algorithm not initialized. Call initialize first.")
        
        if self.best_point is None:
            # First call, return the first point of the simplex
            return self.population[0]
        
        # If all points have values, compute centroid and new point
        if all(v < float('inf') for v in self.values):
            # Sort by function value
            sorted_indices = np.argsort(self.values)
            sorted_pop = [self.population[i] for i in sorted_indices]
            sorted_values = [self.values[i] for i in sorted_indices]
            
            # Get the worst point
            worst_point = sorted_pop[-1]
            
            # Compute centroid of all points except the worst
            centroid = np.mean(sorted_pop[:-1], axis=0)
            
            # Reflection: reflect the worst point through the centroid
            reflected_point = centroid + self.alpha * (centroid - worst_point)
            
            # Ensure we stay within bounds
            for i in range(self.dimensions):
                reflected_point[i] = min(max(reflected_point[i], self.bounds[i][0]), self.bounds[i][1])
            
            return reflected_point
        else:
            # If not all points have values, return the next one without a value
            for i, value in enumerate(self.values):
                if value == float('inf'):
                    return self.population[i]
            
            # Fallback to a random point
            return np.array([np.random.uniform(low, high) for low, high in self.bounds])
    
    def tell(self, point, value):
        """Update the algorithm with the evaluated point and value."""
        if self.population is None:
            raise ValueError("Algorithm not initialized. Call initialize first.")
        
        # Find if this point is already in the population
        found = False
        for i, p in enumerate(self.population):
            if np.allclose(p, point):
                self.values[i] = value
                found = True
                break
        
        if not found:
            # Replace the worst point
            if all(v < float('inf') for v in self.values):
                worst_idx = np.argmax(self.values)
                self.population[worst_idx] = point
                self.values[worst_idx] = value
            else:
                # For initialization: try to find an empty spot
                for i, v in enumerate(self.values):
                    if v == float('inf'):
                        self.population[i] = point
                        self.values[i] = value
                        break
        
        # Update best point
        min_idx = np.argmin(self.values)
        if self.values[min_idx] < self.best_value:
            self.best_value = self.values[min_idx]
            self.best_point = self.population[min_idx].copy()
        
        self.iteration += 1

def main_with_algorithm(algorithm, n_trials, timeout, algorithm_params, **kwargs):
    print("\n" + "="*50)
    print("Starting SPINN optimization for Klein-Gordon equation")
    print("="*50)
    print("\nConfiguration:")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"NC (collocation points): {kwargs['NC']}")
    print(f"NI (initial points): {kwargs['NI']}")
    print(f"NB (boundary points): {kwargs['NB']}")
    print(f"NC_TEST (test points): {kwargs['NC_TEST']}")
    print(f"Random seed: {kwargs['SEED']}")
    print(f"Total epochs: {kwargs['EPOCHS']}")
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout if timeout else 'None'}")
    
    # Определяем запасной алгоритм на случай ошибок с основным
    fallback_algorithm = kwargs.get('fallback_algorithm', 'jade')
    if algorithm == fallback_algorithm:
        fallback_algorithm = 'lshade'  # Предотвращаем бесконечную рекурсию
    
    # Create results logger with algorithm name in directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Для nested_gp добавляем внешний алгоритм в название папки
    if algorithm == 'nested_gp':
        outer_algorithm = kwargs.get('outer_algorithm', 'nelder_mead')
        folder_name = f"{algorithm}_{outer_algorithm}"
    else:
        folder_name = algorithm
        
    log_dir = f"/home/user/SPINN_PyTorch/results/klein_gordon3d/spinn_clear/{folder_name}_{timestamp}"
    logger = ResultLogger(log_dir)
    
    if algorithm == 'nested_gp':
        print(f"\nResults will be saved to: {logger.run_dir}")
        print(f"Directory structure: nested_gp_{outer_algorithm}_{timestamp}")
    else:
        print(f"\nResults will be saved to: {logger.run_dir}")
    
    # Добавляем методы логирования для nested_gp, если их нет в классе ResultLogger
    if algorithm == 'nested_gp':
        if not hasattr(logger, 'log_inner_gp_trial'):
            def log_inner_gp_trial(self, outer_trial_idx, inner_trial_idx, gp_params, pinn_params, error, is_best=False):
                """Логирование внутренней итерации GP."""
                log_data = {
                    'outer_trial': outer_trial_idx,
                    'inner_trial': inner_trial_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'error': float(error),
                    'is_best': is_best,
                    'gp_params': gp_params,
                    'pinn_params': pinn_params
                }
                
                # Сохраняем в CSV-файл
                inner_trials_file = self.run_dir / 'inner_gp_trials.csv'
                
                is_new_file = not inner_trials_file.exists()
                with open(inner_trials_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                    if is_new_file:
                        writer.writeheader()
                    
                    # Преобразуем вложенные словари в JSON строки
                    csv_row = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in log_data.items()}
                    writer.writerow(csv_row)
                
                # Сохраняем детали лучших итераций отдельно в JSON-файл
                if is_best:
                    best_inner_dir = self.run_dir / 'best_inner_trials'
                    best_inner_dir.mkdir(exist_ok=True)
                    
                    best_inner_file = best_inner_dir / f'outer_{outer_trial_idx}_inner_{inner_trial_idx}.json'
                    with open(best_inner_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
            
            setattr(ResultLogger, 'log_inner_gp_trial', log_inner_gp_trial)
        
        if not hasattr(logger, 'log_outer_trial'):
            def log_outer_trial(self, trial_idx, gp_params, best_pinn_error, best_pinn_params, elapsed_time, is_best=True):
                """Логирование внешней итерации оптимизации."""
                log_data = {
                    'trial': trial_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'best_error': float(best_pinn_error),
                    'is_best_so_far': is_best,
                    'gp_params': gp_params,
                    'best_pinn_params': best_pinn_params,
                    'elapsed_time': elapsed_time
                }
                
                # Сохраняем в CSV-файл
                outer_trials_file = self.run_dir / 'outer_trials.csv'
                
                is_new_file = not outer_trials_file.exists()
                with open(outer_trials_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                    if is_new_file:
                        writer.writeheader()
                    
                    # Преобразуем вложенные словари в JSON строки
                    csv_row = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in log_data.items()}
                    writer.writerow(csv_row)
                
                # Сохраняем детали лучших итераций отдельно в JSON-файл
                if is_best:
                    best_outer_dir = self.run_dir / 'best_outer_trials'
                    best_outer_dir.mkdir(exist_ok=True)
                    
                    best_outer_file = best_outer_dir / f'outer_trial_{trial_idx}.json'
                    with open(best_outer_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
            
            setattr(ResultLogger, 'log_outer_trial', log_outer_trial)
    
    # Save run parameters
    run_params = {
        **kwargs,
        'algorithm': algorithm,
        'n_trials': n_trials,
        'timeout': timeout
    }
    
    # Handle algorithm parameters differently for nested_gp
    if algorithm == 'nested_gp':
        outer_algorithm = kwargs.get('outer_algorithm', 'nelder_mead')
        run_params['outer_algorithm'] = outer_algorithm
        run_params['outer_algorithm_params'] = algorithm_params[outer_algorithm]
        run_params['gp_trials'] = kwargs.get('gp_trials', 25)
    else:
        run_params['algorithm_params'] = algorithm_params[algorithm]
    
    with open(logger.run_dir / "run_params.json", 'w') as f:
        json.dump(run_params, f, indent=2)
    print("Run parameters saved to run_params.json")
    
    # Set random seed for reproducibility
    np.random.seed(kwargs['SEED'])
    torch.manual_seed(kwargs['SEED'])
    
    # Create optimizer with selected algorithm
    if algorithm == 'gp':
        # For basic GP optimizer
        optimizer = GPBlackBoxOptimizer(
            n_trials=n_trials,
            timeout=timeout,
            study_name="spinn_optimization_klein_gordon3d",
            logger=logger,
            algorithm=algorithm,
            algorithm_params=algorithm_params[algorithm],
            verbose=True
        )
    elif algorithm == 'nested_gp':
        # For nested optimization (outer algorithm optimizes GP, which optimizes PINN)
        # Get outer algorithm from kwargs or use default
        outer_algorithm = kwargs.get('outer_algorithm', 'nelder_mead')
        gp_trials = kwargs.get('gp_trials', 25)
        
        # Print nested configuration details
        print(f"\nNested GP Configuration:")
        print(f"Outer Algorithm: {outer_algorithm.upper()}")
        print(f"GP Trials per outer iteration: {gp_trials}")
        
        try:
            optimizer = NestedGPOptimizer(
                n_trials=n_trials,
                timeout=timeout,
                study_name="spinn_nested_optimization_klein_gordon3d",
                logger=logger,
                algorithm=outer_algorithm,  # The algorithm used for outer optimization
                algorithm_params=algorithm_params[outer_algorithm],
                verbose=True,
                gp_trials=gp_trials  # Number of GP trials per outer iteration
            )
        except Exception as e:
            print(f"Error initializing NestedGPOptimizer with {outer_algorithm}: {str(e)}")
            print(f"Falling back to {fallback_algorithm} as outer algorithm")
            
            # Пытаемся использовать запасной алгоритм
            try:
                optimizer = NestedGPOptimizer(
                    n_trials=n_trials,
                    timeout=timeout,
                    study_name="spinn_nested_optimization_klein_gordon3d",
                    logger=logger,
                    algorithm=fallback_algorithm,
                    algorithm_params=algorithm_params[fallback_algorithm],
                    verbose=True,
                    gp_trials=gp_trials
                )
            except Exception as e2:
                print(f"Error initializing with fallback algorithm: {str(e2)}")
                print("Using standard KleinGordonOptimizer as final fallback")
                
                # Если и это не сработало, используем обычный оптимизатор
                optimizer = KleinGordonOptimizer(
                    n_trials=n_trials,
                    timeout=timeout,
                    study_name="spinn_optimization_klein_gordon3d",
                    logger=logger,
                    algorithm=fallback_algorithm,
                    algorithm_params=algorithm_params[fallback_algorithm],
                    verbose=True
                )
    else:
        optimizer = KleinGordonOptimizer(
            n_trials=n_trials,
            timeout=timeout,
            study_name="spinn_optimization_klein_gordon3d",
            logger=logger,
            algorithm=algorithm,
            algorithm_params=algorithm_params[algorithm],
            verbose=True
        )
    
    print("\nStarting optimization process...")
    results = optimizer.optimize(kwargs)
    
    print("\n" + "="*50)
    print("Optimization completed!")
    print("="*50)
    
    print("\nBest Architecture Found:")
    print(f"Number of layers: {results['architecture'].n_layers}")
    print(f"Features per layer: {results['architecture'].features}")
    print(f"Activation function: {results['architecture'].activation}")
    
    print(f"\nOptimizer Configuration:")
    print(f"AdamW learning rate: {results['lr_adamw']:.2e}")
    
    # If using nested GP, print the best GP parameters
    if algorithm == 'nested_gp' and 'best_gp_params' in results:
        print("\nBest GP Configuration:")
        for param, value in results['best_gp_params'].items():
            print(f"{param}: {value}")
    
    print(f"\nBest validation error: {results['best_error']:.2e}")
    if results.get('elapsed_time'):
        print(f"Total optimization time: {results['elapsed_time']:.2f} seconds")
    
    print("\nTraining best model with full epochs...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SPINN(results['architecture']).to(device)
    criterion = SPINN_Loss(model)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=results['lr_adamw'])
    
    train_data = spinn_train_generator_klein_gordon3d(kwargs['NC'], seed=kwargs['SEED'])
    train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                 [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                 for t in train_data]
    
    t, x, y, u_gt, tm, xm, ym = spinn_test_generator_klein_gordon3d(kwargs['NC_TEST'])
    t, x, y = t.to(device), x.to(device), y.to(device)
    u_gt = u_gt.to(device)
    tm, xm, ym = tm.to(device), xm.to(device), ym.to(device)
    
    pbar = trange(1, kwargs['EPOCHS'] + 1)
    best_error = float('inf')
    
    for e in pbar:
        # AdamW step
        optimizer.zero_grad()
        tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
        
        # Ensure tensors require gradients
        if not tc.requires_grad:
            tc.requires_grad_(True)
        if not xc.requires_grad:
            xc.requires_grad_(True)
        if not yc.requires_grad:
            yc.requires_grad_(True)
        
        loss_residual = criterion.residual_loss(tc, xc, yc, uc)
        loss_initial = criterion.initial_loss(ti, xi, yi, ui)
        loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
        loss = loss_residual + loss_initial + loss_boundary
        loss.backward()
        optimizer.step()
        
        # Log results at each step
        with torch.no_grad():
            model.eval()
            u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
            error = relative_l2(u, u_gt.reshape(-1))
            
            # Log results
            logger.log_training(
                e,
                {
                    'total': loss.item(),
                    'residual': loss_residual.item(),
                    'initial': loss_initial.item(),
                    'boundary': loss_boundary.item()
                },
                error.item(),
                'adamw'
            )
            
            # Update progress bar
            pbar.set_description(
                f'Loss: {loss.item():.2e} '
                f'(R: {loss_residual.item():.2e}, '
                f'I: {loss_initial.item():.2e}, '
                f'B: {loss_boundary.item():.2e}), '
                f'Error: {error.item():.2e}'
            )
            
            # Save plot and model only at LOG_ITER steps
            if e % kwargs['LOG_ITER'] == 0:
                if error < best_error:
                    best_error = error
                    u = u.reshape(tm.shape)
                    plot_klein_gordon3d(tm, xm, ym, u, logger, f"solution_epoch_{e}")
                    # Save best model
                    logger.save_model(model, f"best_model_epoch_{e}")
        
        model.train()
    
    # Save final model
    logger.save_model(model, "final_model")
    print(f'\nFinal training completed! Best error: {best_error:.2e}')
    print(f'Results saved in: {logger.run_dir}')

def main(NC, NI, NB, NC_TEST, SEED, EPOCHS):
    # Создаем параметры для алгоритмов
    algorithm_params = {
        'jade': {
            'population_size': args.population_size,
            'c': 0.1,
            'p': 0.05
        },
        'lshade': {
            'population_size': args.population_size
        },
        'nelder_mead': {
            'alpha': 1.0,
            'gamma': 2.0,
            'rho': 0.5,
            'sigma': 0.5
        },
        'pso': {
            'num_particles': args.population_size,
            'w': 0.5,
            'c1': 1.0,
            'c2': 1.0
        },
        'grey_wolf': {
            'num_wolves': args.population_size
        },
        'whales': {
            'population_size': args.population_size
        },
        'gp': {
            'acquisition_function': args.acquisition_function,
            'initial_points': args.initial_points
        },
        'nested_gp': {
            'outer_algorithm': args.outer_algorithm,
            'gp_trials': args.gp_trials
        }
    }
    
    # Формируем базовые параметры для оптимизации
    params = {
        'NC': NC,
        'NI': NI,
        'NB': NB,
        'NC_TEST': NC_TEST,
        'SEED': SEED,
        'EPOCHS': EPOCHS,
        'LOG_ITER': args.log_iter  # Сразу добавляем LOG_ITER
    }
    
    # Для nested_gp добавляем параметры внешнего алгоритма и количество GP итераций
    if args.algorithm == 'nested_gp':
        params['outer_algorithm'] = args.outer_algorithm
        params['gp_trials'] = args.gp_trials
        print(f"Setting nested parameters: outer_algorithm={args.outer_algorithm}, gp_trials={args.gp_trials}")
    
    # Запускаем main с выбранным алгоритмом
    main_with_algorithm(args.algorithm, args.n_trials, args.timeout, algorithm_params=algorithm_params, **params)

if __name__ == '__main__':
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='SPINN optimization for Klein-Gordon equation')
    
    # Добавляем аргументы
    parser.add_argument('--algorithm', type=str, 
                      choices=['jade', 'lshade', 'nelder_mead', 'pso', 'grey_wolf', 'whales', 'gp', 'nested_gp'],
                      default='nelder_mead', help='Optimization algorithm to use')
    parser.add_argument('--nc', type=int, default=1000, help='Number of collocation points')
    parser.add_argument('--ni', type=int, default=100, help='Number of initial points')
    parser.add_argument('--nb', type=int, default=100, help='Number of boundary points')
    parser.add_argument('--nc-test', type=int, default=50, help='Number of test points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--n-trials', type=int, default=150, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Optimization timeout in seconds')
    parser.add_argument('--population-size', type=int, default=100, 
                      help='Population size for population-based algorithms (JADE, L-SHADE, PSO, Grey Wolf, Whales)')
    parser.add_argument('--log-iter', type=int, default=500,
                      help='Iteration interval for saving plots and models')
    # Add GP-specific arguments
    parser.add_argument('--acquisition-function', type=str, choices=['EI', 'UCB', 'PI'], default='EI',
                      help='Acquisition function for Gaussian Process optimizer')
    parser.add_argument('--initial-points', type=int, default=5,
                      help='Number of initial random points for GP optimizer')
    
    # Add nested GP-specific arguments
    parser.add_argument('--outer-algorithm', type=str, 
                      choices=['jade', 'lshade', 'nelder_mead', 'pso', 'grey_wolf', 'whales'],
                      default='nelder_mead', help='Outer optimization algorithm for nested_gp')
    parser.add_argument('--gp-trials', type=int, default=25,
                      help='Number of GP trials per outer iteration in nested_gp')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Запускаем main с введенными параметрами
    main(
        NC=args.nc,
        NI=args.ni, 
        NB=args.nb, 
        NC_TEST=args.nc_test, 
        SEED=args.seed, 
        EPOCHS=args.epochs
    )
