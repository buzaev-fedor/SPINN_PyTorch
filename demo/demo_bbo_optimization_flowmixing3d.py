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
    
    # Create results logger with algorithm name in directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/user/SPINN_PyTorch/results/klein_gordon3d/spinn_clear/{algorithm}_{timestamp}"
    logger = ResultLogger(log_dir)
    print(f"\nResults will be saved to: {logger.run_dir}")
    
    # Save run parameters
    with open(logger.run_dir / "run_params.json", 'w') as f:
        json.dump({
            **kwargs,
            'algorithm': algorithm,
            'n_trials': n_trials,
            'timeout': timeout,
            'algorithm_params': algorithm_params[algorithm]
        }, f, indent=2)
    print("Run parameters saved to run_params.json")
    
    # Set random seed for reproducibility
    np.random.seed(kwargs['SEED'])
    torch.manual_seed(kwargs['SEED'])
    
    # Create optimizer with selected algorithm
    optimizer = KleinGordonOptimizer(
        n_trials=n_trials,
        timeout=timeout,
        study_name="spinn_optimization_klein_gordon3d",
        logger=logger,
        algorithm=algorithm,
        algorithm_params=algorithm_params[algorithm],
        verbose=True
    )
    
    # Log BBO algorithm information - REMOVING OR COMMENTING THIS SECTION
    # logger.log_bbo_info(
    #     algorithm=algorithm,
    #     description=optimizer._get_algorithm_description(algorithm),
    #     params=algorithm_params[algorithm]
    # )
    
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
        }
    }
    
    # Формируем параметры для оптимизации
    params = {
        'NC': NC,
        'NI': NI,
        'NB': NB,
        'NC_TEST': NC_TEST,
        'SEED': SEED,
        'EPOCHS': EPOCHS
    }
    
    # Запускаем main с выбранным алгоритмом
    main_with_algorithm(args.algorithm, args.n_trials, args.timeout, algorithm_params=algorithm_params, **params)

if __name__ == '__main__':
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='SPINN optimization for Klein-Gordon equation')
    
    # Добавляем аргументы
    parser.add_argument('--algorithm', type=str, choices=['jade', 'lshade', 'nelder_mead', 'pso', 'grey_wolf', 'whales'],
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
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Добавляем параметр LOG_ITER
    params = {
        'NC': args.nc, 
        'NI': args.ni, 
        'NB': args.nb, 
        'NC_TEST': args.nc_test, 
        'SEED': args.seed, 
        'EPOCHS': args.epochs,
        'LOG_ITER': args.log_iter
    }
    
    # Запускаем main с введенными параметрами
    main_with_algorithm(args.algorithm, args.n_trials, args.timeout, 
                      algorithm_params={
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
                          }
                      }, 
                      **params)
