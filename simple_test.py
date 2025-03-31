import torch
import os
import numpy as np

# Создаем простой класс аргументов
class Args:
    def __init__(self):
        self.nc_test = 50

# Создаем простую функцию для генерации тестовых данных
def simple_generate_test_data(args, save_dir=None):
    """Упрощенная версия generate_test_data без зависимостей от других модулей."""
    # Параметры сетки
    nc_test = args.nc_test
    
    # Области определения переменных
    t_min, t_max = 0.0, 1.0
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    # Создаем равномерную сетку для тестовых точек
    t = torch.linspace(t_min, t_max, nc_test)
    x = torch.linspace(x_min, x_max, nc_test)
    y = torch.linspace(y_min, y_max, nc_test)
    
    print(f"simple_generate_test_data: создаем сетку размером {nc_test}x{nc_test}x{nc_test}")
    print(f"t: {t.shape}, x: {x.shape}, y: {y.shape}")
    
    # Создаем трехмерную сетку (пробуем разные версии meshgrid)
    try:
        # Версия 1: с параметром indexing='ij'
        T1, X1, Y1 = torch.meshgrid(t, x, y, indexing='ij')
        print("Версия 1 (с indexing='ij'):")
        print(f"T1: {T1.shape}, X1: {X1.shape}, Y1: {Y1.shape}")
    except Exception as e:
        print(f"Ошибка в версии 1: {e}")
    
    try:
        # Версия 2: без параметра indexing
        T2, X2, Y2 = torch.meshgrid(t, x, y)
        print("Версия 2 (без indexing):")
        print(f"T2: {T2.shape}, X2: {X2.shape}, Y2: {Y2.shape}")
    except Exception as e:
        print(f"Ошибка в версии 2: {e}")
    
    # Версия 3: создание вручную
    T3 = t.reshape(-1, 1, 1).expand(nc_test, nc_test, nc_test)
    X3 = x.reshape(1, -1, 1).expand(nc_test, nc_test, nc_test)
    Y3 = y.reshape(1, 1, -1).expand(nc_test, nc_test, nc_test)
    print("Версия 3 (вручную):")
    print(f"T3: {T3.shape}, X3: {X3.shape}, Y3: {Y3.shape}")
    
    # Сохраняем данные, если указана директория
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, 'test_data.npz'),
            t=t.numpy(),
            x=x.numpy(),
            y=y.numpy(),
            T=T3.numpy(),
            X=X3.numpy(),
            Y=Y3.numpy()
        )
    
    # Возвращаем разные версии для сравнения
    try:
        return {
            "version1": (t, x, y, T1, X1, Y1),
            "version2": (t, x, y, T2, X2, Y2),
            "version3": (t, x, y, T3, X3, Y3)
        }
    except:
        # Если версия 1 или 2 не сработала, вернем версию 3
        return (t, x, y, T3, X3, Y3)


# Точка входа
if __name__ == "__main__":
    args = Args()
    
    # Создаем временную директорию
    temp_dir = os.path.join(os.getcwd(), 'simple_test_results')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Временная директория: {temp_dir}")
    
    # Генерируем тестовые данные
    print("Генерируем тестовые данные...")
    try:
        results = simple_generate_test_data(args, temp_dir)
        
        if isinstance(results, dict):
            # Если вернулся словарь с разными версиями
            for version, data in results.items():
                print(f"\n{version}: {len(data)} элементов")
                for i, tensor in enumerate(data):
                    if isinstance(tensor, torch.Tensor):
                        print(f"  {i}: форма={tensor.shape}, тип={tensor.dtype}")
        else:
            # Если вернулся только один набор данных
            print(f"\nРезультат: {len(results)} элементов")
            for i, tensor in enumerate(results):
                if isinstance(tensor, torch.Tensor):
                    print(f"  {i}: форма={tensor.shape}, тип={tensor.dtype}")
    except Exception as e:
        print(f"Ошибка при генерации данных: {e}")
        import traceback
        traceback.print_exc()
    
    # Очищаем временную директорию
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Временная директория {temp_dir} удалена") 