import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class ResultLogger:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории
        self.models_dir = self.run_dir / "models"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        self.trials_dir = self.run_dir / "trials"
        self.csv_dir = self.run_dir / "csv_logs"
        
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.trials_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # Инициализируем DataFrame'ы для логов
        self.training_df = pd.DataFrame(columns=['epoch', 'total_loss', 'residual_loss', 'initial_loss', 'boundary_loss', 'error'])
        self.metrics_df = pd.DataFrame(columns=['trial_number', 'error', 'n_layers', 'features', 'activation', 
                                              'lr_adamw', 'scheduler_type', 'scheduler_params',
                                              'use_lbfgs', 'lr_lbfgs', 'lbfgs_start_ratio'])
        self.detailed_loss_df = pd.DataFrame(columns=['epoch', 'timestamp', 'total_loss', 'residual_loss', 'initial_loss', 
                                                    'boundary_loss', 'error', 'optimizer_type', 'current_lr'])
        self.scheduler_df = pd.DataFrame(columns=['epoch', 'scheduler_type', 'current_lr', 'metric_value'])
        self.trials_df = pd.DataFrame(columns=['trial_number', 'optimizer_name', 'error', 'n_layers', 'layer_sizes',
                                             'activation', 'lr_adamw', 'scheduler_type', 'scheduler_factor',
                                             'scheduler_patience', 'scheduler_min_lr', 'scheduler_T_max',
                                             'scheduler_eta_min', 'timestamp'])
        self.training_history_df = pd.DataFrame(columns=['trial_number', 'epoch', 'total_loss', 'residual_loss',
                                                        'initial_loss', 'boundary_loss', 'error', 'learning_rate'])
        self.optimizer_info_df = pd.DataFrame(columns=['algorithm', 'population_size', 'c', 'p', 'alpha', 'gamma',
                                                      'rho', 'sigma', 'w', 'c1', 'c2', 'num_wolves',
                                                      'n_trials', 'timeout', 'timestamp'])
        
        # Инициализируем файлы для логов
        self.training_log_file = self.logs_dir / "training_log.csv"
        self.metrics_log_file = self.logs_dir / "metrics_log.csv"
        self.optimizer_info_file = self.logs_dir / "optimizer_info.json"
        self.detailed_loss_file = self.logs_dir / "detailed_losses.csv"
        self.scheduler_log_file = self.logs_dir / "scheduler_log.csv"
        self.bbo_log_file = self.logs_dir / "bbo_log.txt"
        self.trials_summary_file = self.logs_dir / "trials_summary.json"
        
        # Инициализируем CSV файлы для логов
        self.trials_csv_file = self.csv_dir / "trials.csv"
        self.training_history_csv_file = self.csv_dir / "training_history.csv"
        self.optimizer_info_csv_file = self.csv_dir / "optimizer_info.csv"
        
        # Сохраняем пустые DataFrame'ы в CSV файлы
        self.training_df.to_csv(self.training_log_file, index=False)
        self.metrics_df.to_csv(self.metrics_log_file, index=False)
        self.detailed_loss_df.to_csv(self.detailed_loss_file, index=False)
        self.scheduler_df.to_csv(self.scheduler_log_file, index=False)
        self.trials_df.to_csv(self.trials_csv_file, index=False)
        self.training_history_df.to_csv(self.training_history_csv_file, index=False)
        self.optimizer_info_df.to_csv(self.optimizer_info_csv_file, index=False)
        
        # Инициализируем список для хранения результатов trials
        self.trials_results = []
    
    def _get_iteration_files(self, num_iter: int, optimizer_name: str) -> Dict[str, Path]:
        """Возвращает пути к файлам для конкретной итерации."""
        return {
            'training_log': self.logs_dir / f"{num_iter}_{optimizer_name}_training_log.csv",
            'metrics_log': self.logs_dir / f"{num_iter}_{optimizer_name}_metrics_log.csv",
            'detailed_loss': self.logs_dir / f"{num_iter}_{optimizer_name}_detailed_losses.csv",
            'scheduler_log': self.logs_dir / f"{num_iter}_{optimizer_name}_scheduler_log.csv",
            'trial_json': self.trials_dir / f"{num_iter}_{optimizer_name}.json",
            'trial_csv': self.csv_dir / f"{num_iter}_{optimizer_name}_trial.csv",
            'training_history': self.csv_dir / f"{num_iter}_{optimizer_name}_training_history.csv"
        }

    def log_trial(self, trial_number: int, optimizer_name: str, error: float, params: Dict, 
                 training_history: Optional[List[Dict]] = None):
        """Сохраняет результаты отдельного trial."""
        # Получаем пути к файлам для текущей итерации
        iteration_files = self._get_iteration_files(trial_number, optimizer_name)
        
        # Преобразуем тензоры в числа в истории обучения
        if training_history is not None:
            training_history = [
                {
                    'epoch': int(entry['epoch']),
                    'total_loss': float(entry['total_loss']),
                    'residual_loss': float(entry['residual_loss']),
                    'initial_loss': float(entry.get('initial_loss', 0.0)),
                    'boundary_loss': float(entry['boundary_loss']),
                    'error': float(entry['error']),
                    'learning_rate': float(entry['learning_rate'])
                }
                for entry in training_history
            ]
            
            # Добавляем историю обучения в DataFrame
            history_df = pd.DataFrame(training_history)
            history_df['trial_number'] = trial_number
            history_df.to_csv(iteration_files['training_history'], index=False)
        
        # Преобразуем параметры, содержащие тензоры или numpy типы
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (torch.Tensor, np.float32, np.float64, np.int32, np.int64)):
                processed_params[key] = float(value)
            elif isinstance(value, dict):
                processed_params[key] = {
                    k: float(v) if isinstance(v, (torch.Tensor, np.float32, np.float64, np.int32, np.int64)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                processed_params[key] = [
                    float(v) if isinstance(v, (torch.Tensor, np.float32, np.float64, np.int32, np.int64)) else v
                    for v in value
                ]
            else:
                processed_params[key] = value
        
        trial_data = {
            'trial_number': int(trial_number),
            'optimizer_name': str(optimizer_name),
            'error': float(error),
            'params': processed_params,
            'training_history': training_history,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Сохраняем в отдельный файл JSON
        with open(iteration_files['trial_json'], 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Добавляем в DataFrame trials
        trial_row = pd.DataFrame([{
            'trial_number': trial_number,
            'optimizer_name': optimizer_name,
            'error': error,
            'n_layers': processed_params['n_layers'],
            'layer_sizes': str([processed_params[f'layer_{j}_size'] for j in range(processed_params['n_layers'])]),
            'activation': processed_params['activation'],
            'lr_adamw': processed_params['lr_adamw'],
            'scheduler_type': processed_params['scheduler_type'],
            'scheduler_factor': processed_params.get('scheduler_factor', ''),
            'scheduler_patience': processed_params.get('scheduler_patience', ''),
            'scheduler_min_lr': processed_params.get('scheduler_min_lr', ''),
            'scheduler_T_max': processed_params.get('scheduler_T_max', ''),
            'scheduler_eta_min': processed_params.get('scheduler_eta_min', ''),
            'timestamp': trial_data['timestamp']
        }])
        
        trial_row.to_csv(iteration_files['trial_csv'], index=False)
        
        # Добавляем в общий список
        self.trials_results.append(trial_data)
        
        # Обновляем сводный файл
        self._update_trials_summary()
    
    def _update_trials_summary(self):
        """Обновляет сводный файл с результатами всех trials."""
        # Сортируем по ошибке
        sorted_trials = sorted(self.trials_results, key=lambda x: x['error'])
        
        # Сохраняем в JSON
        with open(self.trials_summary_file, 'w') as f:
            json.dump({
                'total_trials': len(sorted_trials),
                'trials': sorted_trials
            }, f, indent=2)
    
    def get_top_configurations(self, n: int = 10) -> List[Dict]:
        """Возвращает топ-N лучших конфигураций."""
        sorted_trials = sorted(self.trials_results, key=lambda x: x['error'])
        return sorted_trials[:n]
    
    def print_top_configurations(self, n: int = 10):
        """Выводит топ-N лучших конфигураций в консоль."""
        top_configs = self.get_top_configurations(n)
        print("\n" + "="*80)
        print(f"Top {n} Best Configurations:")
        print("="*80)
        
        for i, config in enumerate(top_configs, 1):
            print(f"\n{i}. Trial {config['trial_number']} ({config['optimizer_name']})")
            print(f"   Error: {config['error']:.2e}")
            print(f"   Architecture:")
            print(f"     Layers: {config['params']['n_layers']}")
            print(f"     Layer sizes: {[config['params'][f'layer_{j}_size'] for j in range(config['params']['n_layers'])]}")
            print(f"     Activation: {config['params']['activation']}")
            print(f"   Optimizer:")
            print(f"     AdamW lr: {config['params']['lr_adamw']:.2e}")
            print(f"     Scheduler: {config['params']['scheduler_type']}")
            if config['params']['scheduler_type'] != 'none':
                print("     Scheduler parameters:")
                if config['params']['scheduler_type'] == 'reduce_on_plateau':
                    print(f"       Factor: {config['params']['scheduler_factor']:.2f}")
                    print(f"       Patience: {config['params']['scheduler_patience']}")
                    print(f"       Min lr: {config['params']['scheduler_min_lr']:.2e}")
                elif config['params']['scheduler_type'] == 'cosine_annealing':
                    print(f"       T_max: {config['params']['scheduler_T_max']}")
                    print(f"       Eta min: {config['params']['scheduler_eta_min']:.2e}")
            print("-"*80)
    
    def log_scheduler(self, epoch: int, scheduler_type: str, current_lr: float, metric_value: float,
                     trial_number: int, optimizer_name: str):
        """Логирует информацию о работе планировщика."""
        iteration_files = self._get_iteration_files(trial_number, optimizer_name)
        
        scheduler_row = pd.DataFrame([{
            'epoch': epoch,
            'scheduler_type': scheduler_type,
            'current_lr': current_lr,
            'metric_value': metric_value
        }])
        
        scheduler_row.to_csv(iteration_files['scheduler_log'], index=False)
    
    def log_training(self, epoch: int, losses: Dict[str, float], error: float, 
                    optimizer_type: str = 'adamw', current_lr: float = None,
                    trial_number: int = None, optimizer_name: str = None):
        """Логирует информацию о процессе обучения."""
        if trial_number is not None and optimizer_name is not None:
            iteration_files = self._get_iteration_files(trial_number, optimizer_name)
            
            # Добавляем в training_df
            training_row = pd.DataFrame([{
                'epoch': epoch,
                'total_loss': losses['total'],
                'residual_loss': losses['residual'],
                'initial_loss': losses.get('initial', 0.0),
                'boundary_loss': losses['boundary'],
                'error': error
            }])
            
            # Сохраняем на каждом первом шаге итерации
            training_row.to_csv(iteration_files['training_log'], index=False)
            
            # Добавляем в detailed_loss_df
            detailed_row = pd.DataFrame([{
                'epoch': epoch,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                'total_loss': losses['total'],
                'residual_loss': losses['residual'],
                'initial_loss': losses.get('initial', 0.0),
                'boundary_loss': losses['boundary'],
                'error': error,
                'optimizer_type': optimizer_type,
                'current_lr': current_lr if current_lr is not None else ''
            }])
            
            # Сохраняем на каждом первом шаге итерации
            detailed_row.to_csv(iteration_files['detailed_loss'], index=False)
    
    def log_metrics(self, trial_number: int, error: float, architecture: Dict, optimizer_name: str):
        """Логирует метрики trial."""
        iteration_files = self._get_iteration_files(trial_number, optimizer_name)
        
        metrics_row = pd.DataFrame([{
            'trial_number': trial_number,
            'error': error,
            'n_layers': architecture['n_layers'],
            'features': str([architecture[f'layer_{i}_size'] for i in range(architecture['n_layers'])]),
            'activation': architecture['activation'],
            'lr_adamw': architecture['lr_adamw'],
            'scheduler_type': architecture['scheduler_type'],
            'scheduler_params': json.dumps(architecture.get('scheduler_params', {})),
            'use_lbfgs': 'false',
            'lr_lbfgs': '',
            'lbfgs_start_ratio': ''
        }])
        
        metrics_row.to_csv(iteration_files['metrics_log'], index=False)
    
    def save_model(self, model: torch.nn.Module, name: str):
        torch.save(model.state_dict(), self.models_dir / f"{name}.pth")
    
    def save_architecture(self, architecture: Dict, name: str):
        with open(self.models_dir / f"{name}_architecture.json", 'w') as f:
            json.dump(architecture, f, indent=2)
    
    def save_plot(self, fig: plt.Figure, name: str):
        fig.savefig(self.plots_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def log_optimizer_info(self, algorithm: str, description: str, params: Dict):
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
        else:
            # Создаем новый файл
            optimizer_info = {
                "bbo_algorithm": {
                    "name": algorithm,
                    "description": description,
                    "parameters": params
                }
            }
            with open(self.optimizer_info_file, 'w') as f:
                json.dump(optimizer_info, f, indent=2) 