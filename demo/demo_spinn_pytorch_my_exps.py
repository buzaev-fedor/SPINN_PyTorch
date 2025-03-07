import os
import time
import json
import optuna
import numpy as np
from copy import deepcopy
from pathlib import Path
import csv
from datetime import datetime
import sys

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

# Import BBO algorithms
from bbo_utils.jade import JadeAlgorithm
from bbo_utils.lshade import LShadeAlgorithm
from bbo_utils.neldermead import NelderMead
from bbo_utils.pso import ParticleSwarmOptimization
from bbo_utils.grey_wolf_optimizer import GreyWolfOptimizer
from bbo_utils.whales import WhaleOptimization

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

class ResultLogger:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории
        self.models_dir = self.run_dir / "models"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Инициализируем файлы для логов
        self.training_log_file = self.logs_dir / "training_log.csv"
        self.metrics_log_file = self.logs_dir / "metrics_log.csv"
        self.optimizer_info_file = self.logs_dir / "optimizer_info.json"
        self.detailed_loss_file = self.logs_dir / "detailed_losses.csv"
        self.scheduler_log_file = self.logs_dir / "scheduler_log.csv"
        self.bbo_log_file = self.logs_dir / "bbo_log.txt"  # Новый файл для логирования BBO
        
        # Создаем заголовки CSV файлов
        with open(self.training_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'total_loss', 'residual_loss', 'initial_loss', 'boundary_loss', 'error'])
            
        with open(self.metrics_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_number', 'error', 'n_layers', 'features', 'activation', 
                           'lr_adamw', 'scheduler_type', 'scheduler_params',
                           'use_lbfgs', 'lr_lbfgs', 'lbfgs_start_ratio'])
        
        # Создаем заголовок для детального лога лоссов
        with open(self.detailed_loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'timestamp', 'total_loss', 'residual_loss', 'initial_loss', 
                           'boundary_loss', 'error', 'optimizer_type', 'current_lr'])
        
        # Создаем заголовок для лога планировщика
        with open(self.scheduler_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'scheduler_type', 'current_lr', 'metric_value'])
    
    def log_scheduler(self, epoch: int, scheduler_type: str, current_lr: float, metric_value: float):
        """Логирует информацию о работе планировщика."""
        with open(self.scheduler_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, scheduler_type, current_lr, metric_value])
    
    def log_optimizer_info(self, 
                          algorithm: str,
                          algorithm_params: Dict,
                          n_trials: int,
                          timeout: Optional[int] = None):
        """Сохраняет информацию о выбранном алгоритме оптимизации."""
        optimizer_info = {
            "algorithm": {
                "name": algorithm,
                "description": self._get_algorithm_description(algorithm),
                "parameters": algorithm_params
            },
            "optimization_settings": {
                "n_trials": n_trials,
                "timeout": timeout,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        }
        
        with open(self.optimizer_info_file, 'w') as f:
            json.dump(optimizer_info, f, indent=2)
    
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
    
    def log_training(self, epoch: int, losses: Dict[str, float], error: float, 
                    optimizer_type: str = 'adamw', current_lr: float = None):
        # Записываем в основной лог
        with open(self.training_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                losses['total'],
                losses['residual'],
                losses['initial'],
                losses['boundary'],
                error
            ])
        
        # Записываем в детальный лог с временной меткой
        with open(self.detailed_loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                losses['total'],
                losses['residual'],
                losses['initial'],
                losses['boundary'],
                error,
                optimizer_type,
                current_lr if current_lr is not None else ''
            ])
    
    def log_metrics(self, trial_number: int, error: float, architecture: Dict):
        with open(self.metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_number,
                error,
                architecture['n_layers'],
                str([architecture[f'layer_{i}_size'] for i in range(architecture['n_layers'])]),
                architecture['activation'],
                architecture['lr_adamw'],
                architecture['scheduler_type'],
                json.dumps(architecture.get('scheduler_params', {})),
                'false',  # use_lbfgs всегда false
                '',       # lr_lbfgs пустой
                ''        # lbfgs_start_ratio пустой
            ])
    
    def save_model(self, model: nn.Module, name: str):
        torch.save(model.state_dict(), self.models_dir / f"{name}.pth")
    
    def save_architecture(self, architecture: Dict, name: str):
        with open(self.models_dir / f"{name}_architecture.json", 'w') as f:
            json.dump(architecture, f, indent=2)
    
    def save_plot(self, fig: plt.Figure, name: str):
        fig.savefig(self.plots_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def log_bbo_info(self, algorithm: str, description: str, params: Dict):
        """Логирует информацию о используемом BBO алгоритме."""
        with open(self.bbo_log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BBO Algorithm: {algorithm.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Description:\n")
            f.write(f"{description}\n\n")
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        # Также добавляем информацию в optimizer_info.json
        if os.path.exists(self.optimizer_info_file):
            with open(self.optimizer_info_file, 'r') as f:
                optimizer_info = json.load(f)
            
            optimizer_info["bbo_algorithm"] = {
                "name": algorithm,
                "description": description,
                "parameters": params
            }
            
            with open(self.optimizer_info_file, 'w') as f:
                json.dump(optimizer_info, f, indent=2)

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
                 logger: Optional[ResultLogger] = None,
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

    def train_model(self, model: SPINN, optimizer_params: Dict, train_data: Tuple, test_data: Tuple,
                   device: torch.device, n_epochs: int, log_iter: int) -> float:
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
            if epoch % log_iter == 0:
                with torch.no_grad():
                    model.eval()
                    u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
                    error = relative_l2(u, u_gt.reshape(-1))
                    
                    if error < best_error:
                        best_error = error
                    
                    model.train()
                
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
        train_data = spinn_train_generator_klein_gordon3d(params['NC'], seed=params['SEED'])
        train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                     [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                     for t in train_data]
        
        test_data = spinn_test_generator_klein_gordon3d(params['NC_TEST'])
        test_data = [t.to(device) for t in test_data]
        
        # Train model with reduced epochs for optimization
        n_epochs = params['EPOCHS'] // 10
        
        if self.verbose:
            print(f"\nTraining for {n_epochs} epochs...")
        
        error = self.train_model(
            model=model,
            optimizer_params=optimizer_params,
            train_data=train_data,
            test_data=test_data,
            device=device,
            n_epochs=n_epochs,
            log_iter=params['LOG_ITER']
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
        
        return error

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
                algorithm_params=self.algorithm_params,
                n_trials=self.n_trials,
                timeout=self.timeout
            )
            
            # Log metrics
            self.logger.log_metrics(
                0,  # trial number
                self.best_trial['value'],
                self.best_trial['params']
            )
            
            # Логируем информацию о BBO алгоритме
            self.logger.log_bbo_info(
                algorithm=self.algorithm_name,
                description=self._get_algorithm_description(self.algorithm_name),
                params=self.algorithm_params
            )
            
            if self.verbose:
                print("Results saved successfully")
        
        return {
            'architecture': SPINNArchitecture(
                n_layers=self.best_trial['params']['n_layers'],
                features=[self.best_trial['params'][f'layer_{i}_size'] for i in range(self.best_trial['params']['n_layers'])],
                activation=self.best_trial['params']['activation']
            ),
            'lr_adamw': self.best_trial['params']['lr_adamw'],
            'best_error': self.best_trial['value'],
            'optimization_history': optimizer.objective_function_history,
            'elapsed_time': optimizer.elapsed_time if hasattr(optimizer, 'elapsed_time') else None
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
    print(f"Log interval: {kwargs['LOG_ITER']}")
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
    optimizer = BlackBoxOptimizer(
        n_trials=n_trials,
        timeout=timeout,
        study_name="spinn_optimization_klein_gordon3d",
        logger=logger,
        algorithm=algorithm,
        algorithm_params=algorithm_params[algorithm],
        verbose=True
    )
    
    # Log BBO algorithm information
    logger.log_bbo_info(
        algorithm=algorithm,
        description=optimizer._get_algorithm_description(algorithm),
        params=algorithm_params[algorithm]
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

def main(NC, NI, NB, NC_TEST, SEED, EPOCHS, LOG_ITER):
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
        'EPOCHS': EPOCHS,
        'LOG_ITER': LOG_ITER
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
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--log-iter', type=int, default=1000, help='Logging interval')
    parser.add_argument('--n-trials', type=int, default=150, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Optimization timeout in seconds')
    parser.add_argument('--population-size', type=int, default=100, 
                      help='Population size for population-based algorithms (JADE, L-SHADE, PSO, Grey Wolf, Whales)')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Запускаем main с введенными параметрами
    main(args.nc, args.ni, args.nb, args.nc_test, args.seed, args.epochs, args.log_iter)
