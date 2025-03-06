import os
import time
import json
import optuna
import numpy as np
from copy import deepcopy
from pathlib import Path
import csv
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import trange
from typing import Sequence, List, Dict, Any, Optional, Tuple
from functools import partial
import torch.nn as nn
import time

import torch
import torch.nn as nn
import torch.optim as optim

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
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.timestamp
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
                          pruner: str,
                          algorithm_params: Dict,
                          pruner_params: Dict,
                          n_trials: int,
                          timeout: Optional[int] = None):
        """Сохраняет информацию о выбранном алгоритме оптимизации."""
        optimizer_info = {
            "algorithm": {
                "name": algorithm,
                "description": self._get_algorithm_description(algorithm),
                "parameters": algorithm_params
            },
            "pruner": {
                "name": pruner,
                "description": self._get_pruner_description(pruner),
                "parameters": pruner_params
            },
            "optimization_settings": {
                "n_trials": n_trials,
                "timeout": timeout,
                "timestamp": self.timestamp
            }
        }
        
        with open(self.optimizer_info_file, 'w') as f:
            json.dump(optimizer_info, f, indent=2)
    
    def _get_algorithm_description(self, algorithm: str) -> str:
        """Возвращает описание алгоритма оптимизации."""
        descriptions = {
            'tpe': "Tree-structured Parzen Estimators (TPE) - Байесовская оптимизация, использующая древовидные оценщики Парзена. "
                  "Эффективен для условно-сложных пространств поиска и хорошо работает с категориальными параметрами.",
            
            'random': "Random Search - Простой случайный поиск в пространстве параметров. "
                     "Служит хорошим baseline и эффективен при большой размерности пространства поиска.",
            
            'cma': "Covariance Matrix Adaptation Evolution Strategy (CMA-ES) - Эволюционный алгоритм, "
                  "который адаптивно настраивает ковариационную матрицу. Особенно эффективен для непрерывных параметров.",
            
            'nsgaii': "Non-dominated Sorting Genetic Algorithm II (NSGA-II) - Генетический алгоритм для многоцелевой оптимизации. "
                     "Использует недоминируемую сортировку и поддерживает множество Парето.",
            
            'sobol': "Последовательности Соболя - Квази-случайный поиск, обеспечивающий лучшее покрытие пространства поиска "
                    "по сравнению с чисто случайным поиском. Эффективен для начального исследования пространства параметров."
        }
        return descriptions.get(algorithm, "Описание алгоритма отсутствует")
    
    def _get_pruner_description(self, pruner: str) -> str:
        """Возвращает описание алгоритма прунинга."""
        descriptions = {
            'median': "Median Pruner - Останавливает неперспективные попытки на основе медианного значения целевой метрики. "
                     "Простой и эффективный метод ранней остановки.",
            
            'percentile': "Percentile Pruner - Использует процентили для определения порога остановки. "
                        "Позволяет более гибко настраивать агрессивность прунинга.",
            
            'hyperband': "Hyperband Pruner - Адаптивный алгоритм на основе многорукого бандита. "
                        "Эффективно распределяет вычислительные ресурсы между разными конфигурациями.",
            
            'threshold': "Threshold Pruner - Простой метод остановки по заданному порогу. "
                       "Подходит, когда известно целевое значение метрики.",
            
            'none': "No Pruning - Отключение ранней остановки. Все попытки выполняются до конца."
        }
        return descriptions.get(pruner, "Описание прунера отсутствует")
    
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
                str(architecture['features']),
                architecture['activation'],
                architecture['lr_adamw'],
                architecture['scheduler_type'],
                json.dumps(architecture['scheduler_params']),
                architecture['use_lbfgs'],
                architecture.get('lr_lbfgs', ''),
                architecture.get('lbfgs_start_ratio', '')
            ])
    
    def save_model(self, model: nn.Module, name: str):
        torch.save(model.state_dict(), self.models_dir / f"{name}.pth")
    
    def save_architecture(self, architecture: Dict, name: str):
        with open(self.models_dir / f"{name}_architecture.json", 'w') as f:
            json.dump(architecture, f, indent=2)
    
    def save_plot(self, fig: plt.Figure, name: str):
        fig.savefig(self.plots_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

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
        'tpe': optuna.samplers.TPESampler,
        'random': optuna.samplers.RandomSampler,
        'cma': optuna.samplers.CmaEsSampler,
        'nsgaii': optuna.samplers.NSGAIISampler,
        'sobol': optuna.samplers.QMCSampler,
    }

    PRUNERS = {
        'median': optuna.pruners.MedianPruner,
        'percentile': optuna.pruners.PercentilePruner,
        'hyperband': optuna.pruners.HyperbandPruner,
        'threshold': optuna.pruners.ThresholdPruner,
        'none': optuna.pruners.NopPruner,
    }
    
    SCHEDULERS = {
        'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
        'cosine_annealing': optim.lr_scheduler.CosineAnnealingLR,
        'none': None
    }

    def __init__(self, 
                 n_trials: int = 150,
                 timeout: Optional[int] = None,
                 study_name: str = "spinn_optimization",
                 storage: Optional[str] = None,
                 logger: Optional[ResultLogger] = None,
                 algorithm: str = 'tpe',
                 pruner: str = 'median',
                 algorithm_params: Optional[Dict] = None,
                 pruner_params: Optional[Dict] = None):
        """
        Args:
            n_trials: Количество попыток оптимизации
            timeout: Таймаут в секундах (None для отсутствия таймаута)
            study_name: Имя исследования
            storage: URL для хранения результатов (None для хранения в памяти)
            logger: Логгер для сохранения результатов
            algorithm: Алгоритм оптимизации ('tpe', 'random', 'cma', 'nsgaii', 'sobol')
            pruner: Алгоритм прунинга ('median', 'percentile', 'hyperband', 'threshold', 'none')
            algorithm_params: Дополнительные параметры для алгоритма оптимизации
            pruner_params: Дополнительные параметры для алгоритма прунинга
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.best_trial = None
        self.study = None
        self.logger = logger
        
        # Проверяем корректность выбранного алгоритма
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHMS.keys())}")
        if pruner not in self.PRUNERS:
            raise ValueError(f"Unknown pruner: {pruner}. Available: {list(self.PRUNERS.keys())}")
        
        # Инициализируем параметры алгоритмов
        self.algorithm_params = algorithm_params or {}
        self.pruner_params = pruner_params or {}
        
        # Создаем сэмплер и прунер
        self.sampler = self.ALGORITHMS[algorithm](**self.algorithm_params)
        self.pruner = self.PRUNERS[pruner](**self.pruner_params)

    def create_model(self, trial: optuna.Trial) -> Tuple[SPINN, Dict]:
        """Создает модель с параметрами, предложенными Optuna."""
        # Гиперпараметры архитектуры
        n_layers = trial.suggest_int("n_layers", 2, 5)
        activation = trial.suggest_categorical("activation", ["tanh", "relu", "gelu", "silu"])
        
        # Размеры слоев
        features = []
        for i in range(n_layers):
            features.append(trial.suggest_int(f"layer_{i}_size", 16, 128, log=True))
        
        # Гиперпараметры AdamW
        lr_adamw = trial.suggest_float("lr_adamw", 1e-4, 1e-2, log=True)
        
        # Выбор планировщика скорости обучения
        scheduler_type = trial.suggest_categorical("scheduler_type", ["reduce_on_plateau", "cosine_annealing", "none"])
        scheduler_params = {}
        
        if scheduler_type == "reduce_on_plateau":
            scheduler_params.update({
                "factor": trial.suggest_float("scheduler_factor", 0.1, 0.5),
                "patience": trial.suggest_int("scheduler_patience", 5, 20),
                "min_lr": trial.suggest_float("scheduler_min_lr", 1e-6, 1e-4, log=True)
            })
        elif scheduler_type == "cosine_annealing":
            scheduler_params.update({
                "T_max": trial.suggest_int("scheduler_T_max", 50, 200),
                "eta_min": trial.suggest_float("scheduler_eta_min", 1e-6, 1e-4, log=True)
            })
        
        # Гиперпараметры LBFGS
        use_lbfgs = trial.suggest_categorical("use_lbfgs", [True, False])
        lbfgs_params = {}
        if use_lbfgs:
            lbfgs_params.update({
                "max_iter": trial.suggest_int("lbfgs_max_iter", 10, 50),
                "history_size": trial.suggest_int("lbfgs_history_size", 10, 100),
                "lr": trial.suggest_float("lr_lbfgs", 1e-3, 1.0, log=True),
                "start_epoch_ratio": trial.suggest_float("lbfgs_start_ratio", 0.5, 0.9)
            })
        
        architecture = SPINNArchitecture(n_layers, features, activation)
        model = SPINN(architecture)
        
        return model, {
            "lr_adamw": lr_adamw,
            "use_lbfgs": use_lbfgs,
            "lbfgs_params": lbfgs_params,
            "scheduler_type": scheduler_type,
            "scheduler_params": scheduler_params
        }

    def train_model(self, model: nn.Module, optimizer_params: Dict, train_data: Tuple, test_data: Tuple, 
                   device: torch.device, n_epochs: int, log_iter: int, trial: optuna.Trial) -> float:
        """Тренирует модель с заданными параметрами оптимизации."""
        criterion = SPINN_Loss(model)
        
        # Создаем AdamW оптимизатор с отдельным lr
        optimizer_adamw = optim.AdamW(model.parameters(), lr=optimizer_params["lr_adamw"])
        
        # Создаем планировщик скорости обучения
        scheduler_type = optimizer_params["scheduler_type"]
        scheduler_params = optimizer_params["scheduler_params"]
        
        if scheduler_type == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_adamw,
                mode='min',
                factor=scheduler_params['factor'],
                patience=scheduler_params['patience'],
                min_lr=scheduler_params['min_lr']
            )
        elif scheduler_type == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_adamw,
                T_max=scheduler_params['T_max'],
                eta_min=scheduler_params['eta_min']
            )
        else:
            scheduler = None
        
        # Если используем LBFGS, определяем когда переключаться
        use_lbfgs = optimizer_params["use_lbfgs"]
        if use_lbfgs:
            lbfgs_params = optimizer_params["lbfgs_params"]
            lbfgs_start_epoch = int(n_epochs * lbfgs_params["start_epoch_ratio"])
            optimizer_lbfgs = optim.LBFGS(
                model.parameters(),
                lr=lbfgs_params["lr"],
                max_iter=lbfgs_params["max_iter"],
                history_size=lbfgs_params["history_size"]
            )
        
        best_error = float('inf')
        tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
        t, x, y, u_gt, tm, xm, ym = test_data
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_residual = criterion.residual_loss(tc, xc, yc, uc)
            loss_initial = criterion.initial_loss(ti, xi, yi, ui)
            loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
            loss = loss_residual + loss_initial + loss_boundary
            loss.backward()
            return loss
        
        for e in range(1, n_epochs + 1):
            # Выбираем оптимизатор в зависимости от эпохи
            if use_lbfgs and e >= lbfgs_start_epoch:
                # LBFGS шаг
                loss = optimizer_lbfgs.step(closure)
                # Извлекаем компоненты loss для отображения
                with torch.no_grad():
                    tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
                    loss_residual = criterion.residual_loss(tc, xc, yc, uc)
                    loss_initial = criterion.initial_loss(ti, xi, yi, ui)
                    loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
            else:
                # AdamW шаг
                optimizer_adamw.zero_grad()
                loss_residual = criterion.residual_loss(tc, xc, yc, uc)
                loss_initial = criterion.initial_loss(ti, xi, yi, ui)
                loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
                loss = loss_residual + loss_initial + loss_boundary
                loss.backward()
                optimizer_adamw.step()
            
            # Вычисляем ошибку и обновляем планировщик
            with torch.no_grad():
                model.eval()
                u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
                error = relative_l2(u, u_gt.reshape(-1))
                
                # Получаем текущую скорость обучения
                current_lr = optimizer_adamw.param_groups[0]['lr']
                
                if scheduler is not None:
                    if scheduler_type == "reduce_on_plateau":
                        scheduler.step(error)
                    else:
                        scheduler.step()
                    
                    # Логируем информацию о планировщике
                    if self.logger is not None:
                        self.logger.log_scheduler(
                            epoch=e,
                            scheduler_type=scheduler_type,
                            current_lr=current_lr,
                            metric_value=error.item()
                        )
                
                # Логируем результаты обучения
                if self.logger is not None:
                    self.logger.log_training(
                        e,
                        {
                            'total': loss.item(),
                            'residual': loss_residual.item(),
                            'initial': loss_initial.item(),
                            'boundary': loss_boundary.item()
                        },
                        error.item(),
                        'lbfgs' if use_lbfgs and e >= lbfgs_start_epoch else 'adamw',
                        current_lr
                    )
                
                if e % log_iter == 0:
                    best_error = min(best_error, error.item())
                    trial.report(error.item(), e)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                model.train()
        
        return best_error

    def objective(self, trial: optuna.Trial, params: Dict) -> float:
        """Целевая функция для оптимизации."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Создаем модель и получаем параметры оптимизации
        model, optimizer_params = self.create_model(trial)
        model = model.to(device)
        
        # Генерируем данные
        train_data = spinn_train_generator_klein_gordon3d(params['NC'], seed=params['SEED'])
        train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                     [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                     for t in train_data]
        
        test_data = spinn_test_generator_klein_gordon3d(params['NC_TEST'])
        test_data = [t.to(device) for t in test_data]
        
        # Уменьшаем количество эпох для оптимизации
        n_epochs = params['EPOCHS'] // 10
        
        return self.train_model(
            model=model,
            optimizer_params=optimizer_params,
            train_data=train_data,
            test_data=test_data,
            device=device,
            n_epochs=n_epochs,
            log_iter=params['LOG_ITER'],
            trial=trial
        )

    def optimize(self, params: Dict) -> Dict[str, Any]:
        """Запускает процесс оптимизации."""
        # Логируем информацию об оптимизаторе, если есть логгер
        if self.logger is not None:
            self.logger.log_optimizer_info(
                algorithm=self.sampler.__class__.__name__.replace('Sampler', '').lower(),
                pruner=self.pruner.__class__.__name__.replace('Pruner', '').lower(),
                algorithm_params=self.algorithm_params,
                pruner_params=self.pruner_params,
                n_trials=self.n_trials,
                timeout=self.timeout
            )
        
        if isinstance(self.sampler, optuna.samplers.CmaEsSampler):
            # Для CMA-ES нужны специальные настройки
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                direction="minimize",
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                direction="minimize"
            )
        
        objective_with_params = lambda trial: self.objective(trial, params)
        
        try:
            self.study.optimize(
                objective_with_params,
                n_trials=self.n_trials,
                timeout=self.timeout,
                catch=(RuntimeError,)  # Ловим ошибки выполнения, но продолжаем оптимизацию
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        
        self.best_trial = self.study.best_trial
        best_params = self.best_trial.params
        
        n_layers = best_params["n_layers"]
        features = [best_params[f"layer_{i}_size"] for i in range(n_layers)]
        activation = best_params["activation"]
        
        best_architecture = SPINNArchitecture(n_layers, features, activation)
        
        # Собираем все параметры оптимизации
        optimizer_config = {
            "lr_adamw": best_params["lr_adamw"],
            "use_lbfgs": best_params.get("use_lbfgs", False)
        }
        
        if optimizer_config["use_lbfgs"]:
            optimizer_config["lbfgs_params"] = {
                "lr": best_params["lr_lbfgs"],
                "max_iter": best_params["lbfgs_max_iter"],
                "history_size": best_params["lbfgs_history_size"],
                "start_epoch_ratio": best_params["lbfgs_start_ratio"]
            }
        
        results = {
            "architecture": best_architecture,
            **optimizer_config,
            "best_error": self.best_trial.value,
            "optimization_history": [
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in self.study.trials
            ],
            "optimization_statistics": {
                "completed_trials": len(self.study.trials),
                "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }
        
        if self.logger is not None:
            # Сохраняем результаты оптимизации
            self.logger.save_architecture(best_architecture.to_dict(), "best_architecture")
            
            # Логируем метрики для каждой попытки
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    n_layers = trial.params["n_layers"]
                    features = [trial.params[f"layer_{i}_size"] for i in range(n_layers)]
                    trial_params = {
                        "n_layers": n_layers,
                        "features": features,
                        "activation": trial.params["activation"],
                        "lr_adamw": trial.params["lr_adamw"],
                        "use_lbfgs": trial.params.get("use_lbfgs", False)
                    }
                    
                    if trial_params["use_lbfgs"]:
                        trial_params["lbfgs_params"] = {
                            "lr": trial.params["lr_lbfgs"],
                            "start_epoch_ratio": trial.params["lbfgs_start_ratio"]
                        }
                    
                    self.logger.log_metrics(
                        trial.number,
                        trial.value,
                        trial_params
                    )
            
            # Сохраняем статистику оптимизации
            with open(self.logger.logs_dir / "optimization_statistics.json", 'w') as f:
                json.dump(results["optimization_statistics"], f, indent=2)
        
        return results

def main(NC, NI, NB, NC_TEST, SEED, EPOCHS, LOG_ITER):
    # Создаем логгер результатов
    logger = ResultLogger("/home/user/SPINN_PyTorch/results/klein_gordon3d/spinn_clear")
    
    params = {
        'NC': NC,
        'NI': NI,
        'NB': NB,
        'NC_TEST': NC_TEST,
        'SEED': SEED,
        'EPOCHS': EPOCHS,
        'LOG_ITER': LOG_ITER
    }
    
    # Сохраняем параметры запуска
    with open(logger.run_dir / "run_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    # Создаем оптимизатор с выбранным алгоритмом
    optimizer = BlackBoxOptimizer(
        n_trials=150,
        timeout=None,
        study_name="spinn_optimization_klein_gordon3d",
        logger=logger,
        algorithm='tpe',  # Можно выбрать: 'tpe', 'random', 'cma', 'nsgaii', 'sobol'
        pruner='median',  # Можно выбрать: 'median', 'percentile', 'hyperband', 'threshold', 'none'
        algorithm_params={'seed': SEED},
        pruner_params={'n_startup_trials': 5, 'n_warmup_steps': 20}
    )
    
    print("Starting Black Box Optimization...")
    results = optimizer.optimize(params)
    
    print("\nOptimization completed!")
    print("\nBest Architecture Found:")
    print(f"Number of layers: {results['architecture'].n_layers}")
    print(f"Features per layer: {results['architecture'].features}")
    print(f"Activation function: {results['architecture'].activation}")
    print(f"\nOptimizer Configuration:")
    print(f"AdamW learning rate: {results['lr_adamw']:.2e}")
    if results.get('use_lbfgs', False):
        print("\nLBFGS Configuration:")
        print(f"Start epoch ratio: {results['lbfgs_params']['start_epoch_ratio']:.2f}")
        print(f"Learning rate: {results['lbfgs_params']['lr']:.2e}")
        print(f"Max iterations: {results['lbfgs_params']['max_iter']}")
        print(f"History size: {results['lbfgs_params']['history_size']}")
    print(f"\nBest validation error: {results['best_error']:.2e}")
    
    # Тренируем лучшую модель на полном количестве эпох
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SPINN(results['architecture']).to(device)
    criterion = SPINN_Loss(model)
    
    # Создаем оптимизаторы согласно лучшим найденным параметрам
    optimizer_adamw = optim.AdamW(model.parameters(), lr=results['lr_adamw'])
    if results.get('use_lbfgs', False):
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=results['lbfgs_params']['lr'],
            max_iter=results['lbfgs_params']['max_iter'],
            history_size=results['lbfgs_params']['history_size']
        )
        lbfgs_start_epoch = int(EPOCHS * results['lbfgs_params']['start_epoch_ratio'])
    
    train_data = spinn_train_generator_klein_gordon3d(NC, seed=SEED)
    train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                 [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                 for t in train_data]
    
    t, x, y, u_gt, tm, xm, ym = spinn_test_generator_klein_gordon3d(NC_TEST)
    t, x, y = t.to(device), x.to(device), y.to(device)
    u_gt = u_gt.to(device)
    tm, xm, ym = tm.to(device), xm.to(device), ym.to(device)
    
    pbar = trange(1, EPOCHS + 1)
    best_error = float('inf')
    
    print("\nTraining best model with full epochs...")
    for e in pbar:
        # Выбираем оптимизатор в зависимости от эпохи
        if results.get('use_lbfgs', False) and e >= lbfgs_start_epoch:
            # LBFGS шаг
            def closure():
                optimizer_lbfgs.zero_grad()
                tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
                
                # Убеждаемся, что тензоры требуют градиентов
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
                return loss
            
            loss = optimizer_lbfgs.step(closure)
            # Извлекаем компоненты loss для отображения
            with torch.no_grad():
                tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
                loss_residual = criterion.residual_loss(tc, xc, yc, uc)
                loss_initial = criterion.initial_loss(ti, xi, yi, ui)
                loss_boundary = criterion.boundary_loss(tb, xb, yb, ub)
        else:
            # AdamW шаг
            optimizer_adamw.zero_grad()
            tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
            
            # Убеждаемся, что тензоры требуют градиентов
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
            optimizer_adamw.step()
        
        # Логируем результаты на каждом шаге
        with torch.no_grad():
            model.eval()
            u = model(tm.reshape(-1), xm.reshape(-1), ym.reshape(-1))
            error = relative_l2(u, u_gt.reshape(-1))
            
            # Логируем результаты
            logger.log_training(
                e,
                {
                    'total': loss.item(),
                    'residual': loss_residual.item(),
                    'initial': loss_initial.item(),
                    'boundary': loss_boundary.item()
                },
                error.item(),
                'lbfgs' if results.get('use_lbfgs', False) and e >= lbfgs_start_epoch else 'adamw'
            )
            
            # Обновляем progress bar на каждом шаге
            pbar.set_description(
                f'Loss: {loss.item():.2e} '
                f'(R: {loss_residual.item():.2e}, '
                f'I: {loss_initial.item():.2e}, '
                f'B: {loss_boundary.item():.2e}), '
                f'Error: {error.item():.2e}'
            )
            
            # Сохраняем график и модель только на шагах LOG_ITER
            if e % LOG_ITER == 0:
                if error < best_error:
                    best_error = error
                    u = u.reshape(tm.shape)
                    plot_klein_gordon3d(tm, xm, ym, u, logger, f"solution_epoch_{e}")
                    # Сохраняем лучшую модель
                    logger.save_model(model, f"best_model_epoch_{e}")
            
            model.train()
    
    # Сохраняем финальную модель
    logger.save_model(model, "final_model")
    print(f'\nFinal training completed! Best error: {best_error:.2e}')
    print(f'Results saved in: {logger.run_dir}')

if __name__ == '__main__':
    PARAMS = {
        'NC': 1000,        # количество точек коллокации
        'NI': 100,         # количество начальных точек
        'NB': 100,         # количество граничных точек
        'NC_TEST': 50,     # количество тестовых точек
        'SEED': 42,        # seed для воспроизводимости
        'EPOCHS': 10000,   # количество эпох
        'LOG_ITER': 1000   # частота логирования
    }

    main(**PARAMS)