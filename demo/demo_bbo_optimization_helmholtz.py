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
        'hardtanh': nn.Hardtanh,
    }
    
    def __init__(self, architecture: SPINNArchitecture, k: float = 1.0):
        super().__init__()
        self.features = architecture.features
        self.activation_name = architecture.activation
        self.activation = self.ACTIVATIONS[architecture.activation]()
        self.k = k  # волновое число для уравнения Гельмгольца
        
        # Создаем слои для каждого входа (x, y, z)
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
    
    def forward(self, x, y, z):
        # Преобразуем входы в 2D тензоры [batch_size, 1]
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        
        # Пропускаем через отдельные сети
        x_features = self.networks[0](x)
        y_features = self.networks[1](y)
        z_features = self.networks[2](z)
        
        # Объединяем признаки
        combined = torch.cat([x_features, y_features], dim=1)
        combined = self.activation(self.combine_layer1(combined))
        
        combined = torch.cat([combined, z_features], dim=1)
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
        self.k = model.k

    def residual_loss(self, x, y, z, source_term):
        # Убеждаемся, что входные тензоры требуют градиентов
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        if not z.requires_grad:
            z.requires_grad_(True)
        
        # Получаем выход модели и убеждаемся, что он требует градиентов
        u = self.model(x, y, z)
        if not u.requires_grad:
            u.requires_grad_(True)
        
        # Вычисляем производные
        uxx = compute_second_derivative(u, x)
        uyy = compute_second_derivative(u, y)
        uzz = compute_second_derivative(u, z)
        
        # Вычисляем невязку для уравнения Гельмгольца: ∇²u + k²u = f
        # или: uxx + uyy + uzz + k²u = f
        residual = uxx + uyy + uzz + self.k**2 * u - source_term
        return torch.mean(residual**2)

    def boundary_loss(self, x, y, z, u_true=None):
        """
        Вычисляет ошибку на границе.
        x, y, z - списки тензоров для разных сегментов границы
        u_true - список тензоров или None для нулевых граничных условий
        """
        loss = 0.0
        
        # Проходим по всем сегментам границы
        for i in range(len(x)):
            u_pred = self.model(x[i], y[i], z[i])
            
            if u_true is None or u_true[i] is None:
                # Нулевые граничные условия
                loss += torch.mean(u_pred**2)
            else:
                # Заданные граничные условия
                loss += torch.mean((u_pred - u_true[i])**2)
        
        # Возвращаем среднее значение ошибки по всем сегментам
        return loss / len(x)

    def __call__(self, *train_data):
        xc, yc, zc, fc, xb, yb, zb, ub = train_data
        
        loss_residual = self.residual_loss(xc, yc, zc, fc)
        loss_boundary = self.boundary_loss(xb, yb, zb, ub)
        
        return loss_residual + loss_boundary

# Функция шага оптимизации
def update_model(model, optimizer, train_data):
    optimizer.zero_grad()
    criterion = SPINN_Loss(model)
    loss = criterion(*train_data)
    loss.backward()
    optimizer.step()
    return loss.item()

# Точное решение для тестирования уравнения Гельмгольца
def helmholtz3d_exact_u(a1, a2, a3, x, y, z):
    return torch.sin(a1*torch.pi*x) * torch.sin(a2*torch.pi*y) * torch.sin(a3*torch.pi*z)

# Источниковый член уравнения Гельмгольца
def helmholtz3d_source_term(a1, a2, a3, x, y, z, k=1.0):
    u_gt = helmholtz3d_exact_u(a1, a2, a3, x, y, z)
    
    # Вычисление вторых производных
    uxx = -(a1*torch.pi)**2 * u_gt
    uyy = -(a2*torch.pi)**2 * u_gt
    uzz = -(a3*torch.pi)**2 * u_gt
    
    # Возвращаем источниковый член уравнения Гельмгольца
    return uxx + uyy + uzz + k**2 * u_gt

# Генератор тренировочных данных для уравнения Гельмгольца
def spinn_train_generator_helmholtz3d(NC, NB, seed=42, domain_size=1.0, k=1.0):
    torch.manual_seed(seed)
    
    # Точки внутри области (коллокационные точки)
    xc = torch.rand(NC) * 2 * domain_size - domain_size
    yc = torch.rand(NC) * 2 * domain_size - domain_size
    zc = torch.rand(NC) * 2 * domain_size - domain_size
    
    # Источниковая функция для правой части уравнения Гельмгольца
    # Пример: источниковый член для известного решения u = sin(πx)sin(πy)sin(πz)
    a1, a2, a3 = 1, 1, 1  # Коэффициенты для решения
    fc = helmholtz3d_source_term(a1, a2, a3, xc, yc, zc, k)
    
    # Генерация граничных точек
    xb = []
    yb = []
    zb = []
    ub = []
    
    # Точки на границах x = -domain_size и x = domain_size
    x_boundary = torch.full((NB,), -domain_size)
    xb.append(x_boundary)
    yb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    zb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    x_boundary = torch.full((NB,), domain_size)
    xb.append(x_boundary)
    yb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    zb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    # Точки на границах y = -domain_size и y = domain_size
    y_boundary = torch.full((NB,), -domain_size)
    xb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    yb.append(y_boundary)
    zb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    y_boundary = torch.full((NB,), domain_size)
    xb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    yb.append(y_boundary)
    zb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    # Точки на границах z = -domain_size и z = domain_size
    z_boundary = torch.full((NB,), -domain_size)
    xb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    yb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    zb.append(z_boundary)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    z_boundary = torch.full((NB,), domain_size)
    xb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    yb.append(torch.rand(NB) * 2 * domain_size - domain_size)
    zb.append(z_boundary)
    ub.append(torch.zeros(NB))  # Нулевые граничные условия
    
    return xc, yc, zc, fc, xb, yb, zb, ub

# Функция для генерации тестовых данных для уравнения Гельмгольца
def spinn_test_generator_helmholtz3d(NC_TEST, domain_size=1.0, k=1.0):
    # Создаем сетку точек
    x = torch.linspace(-domain_size, domain_size, NC_TEST)
    y = torch.linspace(-domain_size, domain_size, NC_TEST)
    z = torch.linspace(-domain_size, domain_size, NC_TEST)
    
    xm, ym, zm = torch.meshgrid(x, y, z, indexing='ij')
    
    # Аналитическое решение для тестирования
    a1, a2, a3 = 1, 1, 1  # Коэффициенты для решения
    u_gt = helmholtz3d_exact_u(a1, a2, a3, xm, ym, zm)
    
    return x, y, z, u_gt, xm, ym, zm

def relative_l2(u, u_gt):
    return torch.norm(u - u_gt) / torch.norm(u_gt)

def plot_helmholtz3d(x, y, z, u, logger: Optional[ResultLogger] = None, name: str = "solution"):
    # Преобразуем тензоры PyTorch в numpy массивы для визуализации
    x = x.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()
    z = z.detach().cpu().numpy().flatten()
    u = u.detach().cpu().numpy().flatten()
    
    # Нормализуем значения для цветовой карты
    u_norm = (u - u.min()) / (u.max() - u.min())
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Создаем scatter plot с нормализованными цветами
    scatter = ax.scatter(x, y, z, c=u_norm, s=1, cmap='seismic', vmin=0, vmax=1)
    
    # Настраиваем внешний вид
    ax.set_title('U(x, y, z)', fontsize=20)
    ax.set_xlabel('x', fontsize=18, labelpad=10)
    ax.set_ylabel('y', fontsize=18, labelpad=10)
    ax.set_zlabel('z', fontsize=18, labelpad=10)
    
    # Добавляем colorbar с реальными значениями
    cbar = plt.colorbar(scatter)
    cbar.set_label('u(x,y,z)', fontsize=16)
    
    if logger is not None:
        logger.save_plot(fig, name)
    else:
        plt.savefig('helmholtz3d.png', dpi=300, bbox_inches='tight')
        plt.close()

# Наследуем от базового класса BlackBoxOptimizer для решения задачи Гельмгольца
class HelmholtzOptimizer(BlackBoxOptimizer):
    def create_model(self, params: Dict) -> Tuple[SPINN, Dict]:
        """Creates a model with the given parameters."""
        # Extract architecture parameters
        n_layers = int(params['n_layers'])
        features = [int(params[f'layer_{i}_size']) for i in range(n_layers)]
        activation = params['activation']
        k = float(params.get('wave_number', 1.0))  # волновое число
        
        # Extract optimizer parameters
        lr_adamw = float(params['lr_adamw'])
        
        # Create architecture and model
        architecture = SPINNArchitecture(n_layers, features, activation)
        model = SPINN(architecture, k=k)
        
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
        """Generate training data for Helmholtz equation."""
        k = float(params.get('wave_number', 1.0))
        return spinn_train_generator_helmholtz3d(
            params['NC'], 
            params['NB'], 
            seed=params['SEED'],
            domain_size=params.get('domain_size', 1.0),
            k=k
        )
    
    def generate_test_data(self, params: Dict) -> Tuple:
        """Generate test data for Helmholtz equation."""
        k = float(params.get('wave_number', 1.0))
        return spinn_test_generator_helmholtz3d(
            params['NC_TEST'],
            domain_size=params.get('domain_size', 1.0),
            k=k
        )

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
        x, y, z, u_gt, xm, ym, zm = test_data
        
        # Очищаем историю обучения для нового trial
        self.training_history = []
        
        for epoch in range(1, n_epochs + 1):
            # AdamW step
            optimizer.zero_grad()
            xc, yc, zc, fc, xb, yb, zb, ub = train_data
            
            # Ensure tensors require gradients
            if not xc.requires_grad:
                xc.requires_grad_(True)
            if not yc.requires_grad:
                yc.requires_grad_(True)
            if not zc.requires_grad:
                zc.requires_grad_(True)
            
            loss_residual = criterion.residual_loss(xc, yc, zc, fc)
            loss_boundary = criterion.boundary_loss(xb, yb, zb, ub)
            loss = loss_residual + loss_boundary
            loss.backward()
            optimizer.step()
            
            # Compute validation error
            with torch.no_grad():
                model.eval()
                u = model(xm.reshape(-1), ym.reshape(-1), zm.reshape(-1))
                error = relative_l2(u, u_gt.reshape(-1))
                
                if error < best_error:
                    best_error = error
                
                model.train()
            
            # Сохраняем историю обучения
            self.training_history.append({
                'epoch': int(epoch),
                'total_loss': float(loss.item()),
                'residual_loss': float(loss_residual.item()),
                'boundary_loss': float(loss_boundary.item()),
                'initial_loss': 0.0,  # Added for compatibility with the logger
                'error': float(error.item()),
                'learning_rate': float(optimizer.param_groups[0]['lr'])
            })
            
            # Update scheduler if using ReduceLROnPlateau
            if scheduler is not None and optimizer_params['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(error)
            elif scheduler is not None and optimizer_params['scheduler_type'] == 'cosine_annealing':
                scheduler.step()
        
        # Final validation error
        with torch.no_grad():
            model.eval()
            u = model(xm.reshape(-1), ym.reshape(-1), zm.reshape(-1))
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
            'wave_number': self.best_trial['params'].get('wave_number', 1.0),
            'lr_adamw': self.best_trial['params']['lr_adamw'],
            'best_error': self.best_trial['value'],
            'optimization_history': getattr(self, 'optimization_history', []),
            'elapsed_time': getattr(self, 'elapsed_time', None)
        }

def main_with_algorithm(algorithm, n_trials, timeout, algorithm_params, **kwargs):
    print("\n" + "="*50)
    print("Starting SPINN optimization for Helmholtz equation")
    print("="*50)
    print("\nConfiguration:")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"NC (collocation points): {kwargs['NC']}")
    print(f"NB (boundary points): {kwargs['NB']}")
    print(f"NC_TEST (test points): {kwargs['NC_TEST']}")
    print(f"Random seed: {kwargs['SEED']}")
    print(f"Total epochs: {kwargs['EPOCHS']}")
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout if timeout else 'None'}")
    print(f"Wave number k: {kwargs.get('wave_number', 1.0)}")
    
    # Create results logger with algorithm name in directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/user/SPINN_PyTorch/results/helmholtz3d/spinn_clear/{algorithm}_{timestamp}"
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
    optimizer = HelmholtzOptimizer(
        n_trials=n_trials,
        timeout=timeout,
        study_name="spinn_optimization_helmholtz3d",
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
    print(f"Wave number k: {results['wave_number']}")
    
    print(f"\nOptimizer Configuration:")
    print(f"AdamW learning rate: {results['lr_adamw']:.2e}")
    
    print(f"\nBest validation error: {results['best_error']:.2e}")
    if results.get('elapsed_time'):
        print(f"Total optimization time: {results['elapsed_time']:.2f} seconds")
    
    print("\nTraining best model with full epochs...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SPINN(results['architecture'], k=float(results['wave_number'])).to(device)
    criterion = SPINN_Loss(model)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=results['lr_adamw'])
    
    train_data = spinn_train_generator_helmholtz3d(
        kwargs['NC'], 
        kwargs['NB'], 
        seed=kwargs['SEED'],
        domain_size=kwargs.get('domain_size', 1.0),
        k=float(results['wave_number'])
    )
    train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                 [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                 for t in train_data]
    
    x, y, z, u_gt, xm, ym, zm = spinn_test_generator_helmholtz3d(
        kwargs['NC_TEST'],
        domain_size=kwargs.get('domain_size', 1.0),
        k=float(results['wave_number'])
    )
    x, y, z = x.to(device), y.to(device), z.to(device)
    u_gt = u_gt.to(device)
    xm, ym, zm = xm.to(device), ym.to(device), zm.to(device)
    
    pbar = trange(1, kwargs['EPOCHS'] + 1)
    best_error = float('inf')
    
    for e in pbar:
        # AdamW step
        optimizer.zero_grad()
        xc, yc, zc, fc, xb, yb, zb, ub = train_data
        
        # Ensure tensors require gradients
        if not xc.requires_grad:
            xc.requires_grad_(True)
        if not yc.requires_grad:
            yc.requires_grad_(True)
        if not zc.requires_grad:
            zc.requires_grad_(True)
        
        loss_residual = criterion.residual_loss(xc, yc, zc, fc)
        loss_boundary = criterion.boundary_loss(xb, yb, zb, ub)
        loss = loss_residual + loss_boundary
        loss.backward()
        optimizer.step()
        
        # Log results at each step
        with torch.no_grad():
            model.eval()
            u = model(xm.reshape(-1), ym.reshape(-1), zm.reshape(-1))
            error = relative_l2(u, u_gt.reshape(-1))
            
            # Log results
            logger.log_training(
                e,
                {
                    'total': loss.item(),
                    'residual': loss_residual.item(),
                    'boundary': loss_boundary.item(),
                    'initial': 0.0  # Added for compatibility with the logger
                },
                error.item(),
                'adamw'
            )
            
            # Update progress bar
            pbar.set_description(
                f'Loss: {loss.item():.2e} '
                f'(R: {loss_residual.item():.2e}, '
                f'B: {loss_boundary.item():.2e}), '
                f'Error: {error.item():.2e}'
            )
            
            # Save plot and model only at LOG_ITER steps
            if e % kwargs['LOG_ITER'] == 0:
                if error < best_error:
                    best_error = error
                    u = u.reshape(xm.shape)
                    plot_helmholtz3d(xm, ym, zm, u, logger, f"solution_epoch_{e}")
                    # Save best model
                    logger.save_model(model, f"best_model_epoch_{e}")
        
        model.train()
    
    # Save final model
    logger.save_model(model, "final_model")
    print(f'\nFinal training completed! Best error: {best_error:.2e}')
    print(f'Results saved in: {logger.run_dir}')

def main(NC, NB, NC_TEST, SEED, EPOCHS, wave_number=1.0, domain_size=1.0):
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
        'NB': NB,
        'NC_TEST': NC_TEST,
        'SEED': SEED,
        'EPOCHS': EPOCHS,
        'wave_number': wave_number,
        'domain_size': domain_size
    }
    
    # Запускаем main с выбранным алгоритмом
    main_with_algorithm(args.algorithm, args.n_trials, args.timeout, algorithm_params=algorithm_params, **params)

if __name__ == '__main__':
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='SPINN optimization for Helmholtz equation')
    
    # Добавляем аргументы
    parser.add_argument('--algorithm', type=str, choices=['jade', 'lshade', 'nelder_mead', 'pso', 'grey_wolf', 'whales'],
                      default='nelder_mead', help='Optimization algorithm to use')
    parser.add_argument('--nc', type=int, default=1000, help='Number of collocation points')
    parser.add_argument('--nb', type=int, default=100, help='Number of boundary points')
    parser.add_argument('--nc-test', type=int, default=20, help='Number of test points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--n-trials', type=int, default=150, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Optimization timeout in seconds')
    parser.add_argument('--population-size', type=int, default=100, 
                      help='Population size for population-based algorithms (JADE, L-SHADE, PSO, Grey Wolf, Whales)')
    parser.add_argument('--log-iter', type=int, default=100,
                      help='Iteration interval for saving plots and models')
    parser.add_argument('--wave-number', type=float, default=1.0,
                      help='Wave number (k) for Helmholtz equation')
    parser.add_argument('--domain-size', type=float, default=1.0,
                      help='Size of the domain (from -size to +size)')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Добавляем параметр LOG_ITER
    params = {
        'NC': args.nc, 
        'NB': args.nb, 
        'NC_TEST': args.nc_test, 
        'SEED': args.seed, 
        'EPOCHS': args.epochs,
        'LOG_ITER': args.log_iter,
        'wave_number': args.wave_number,
        'domain_size': args.domain_size
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
