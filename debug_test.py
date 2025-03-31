import torch
import sys
import os

# Добавляем директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import setup_networks, name_model
from utils.data_generators import generate_test_data_pytorch
from utils.eval_functions import eval_flow_mixing3d, eval_pinn, eval_spinn

# Создаем аргументы для теста
class Args:
    def __init__(self):
        self.model = 'spinn'
        self.equation = 'flow_mixing3d'
        self.nc = 32
        self.nc_test = 50
        self.seed = 111
        self.lr = 1e-3
        self.epochs = 5000
        self.mlp = 'modified_mlp'
        self.n_layers = 3
        self.features = 64
        self.r = 128
        self.out_dim = 1
        self.pos_enc = 0
        self.vmax = 0.385
        self.log_iter = 500
        self.plot_iter = 5000

args = Args()

# Устанавливаем случайное зерно
torch.manual_seed(args.seed)

# Создаем временную директорию для результатов
temp_dir = os.path.join(os.getcwd(), 'temp_test_results')
os.makedirs(temp_dir, exist_ok=True)
print(f"Временная директория: {temp_dir}")

# Создаем модель
model = setup_networks(args)
print(f"Создана модель с {sum(p.numel() for p in model.parameters() if p.requires_grad)} параметрами")

# Генерируем тестовые данные
print("Генерируем тестовые данные...")
test_data = generate_test_data_pytorch(args, temp_dir)
print(f"Тестовые данные содержат {len(test_data)} элементов")

# Выводим информацию о каждом тензоре
for i, tensor in enumerate(test_data):
    if isinstance(tensor, torch.Tensor):
        print(f"test_data[{i}]: тип={type(tensor).__name__}, форма={tensor.shape}, размер={tensor.numel()}")

# Пробуем выполнить оценку
try:
    # Проверяем функцию eval_flow_mixing3d
    print("\nТестируем функцию eval_flow_mixing3d:")
    error = eval_flow_mixing3d(model, *test_data)
    print(f"Ошибка: {error}")
    
    # Проверяем функцию eval_spinn
    print("\nТестируем функцию eval_spinn:")
    error = eval_spinn(model, *test_data)
    print(f"Ошибка: {error}")
    
except Exception as e:
    print(f"Возникла ошибка: {e}")
    import traceback
    traceback.print_exc()

# Очистка временной директории
import shutil
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"Временная директория {temp_dir} удалена") 