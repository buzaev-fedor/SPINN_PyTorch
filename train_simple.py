import os
import sys
import torch
import argparse
from flow_mixing3d_pytorch import main

# Для отладки мы создаем класс аргументов вместо использования argparse
class Args:
    def __init__(self):
        self.model = 'spinn'
        self.equation = 'flow_mixing3d'
        self.nc = 32          # число точек коллокации
        self.nc_test = 50     # число тестовых точек
        self.seed = 111       # случайное зерно
        self.lr = 1e-3        # скорость обучения
        self.epochs = 50      # небольшое число эпох для теста
        self.mlp = 'modified_mlp'
        self.n_layers = 3
        self.features = 64
        self.r = 128
        self.out_dim = 1
        self.pos_enc = 0
        self.vmax = 0.385     # максимальная скорость для задачи flow_mixing3d
        self.log_iter = 10    # логирование каждые 10 итераций
        self.plot_iter = 50   # построение графиков каждые 50 итераций

if __name__ == "__main__":
    # Создаем аргументы
    args = Args()
    
    # Выводим информацию о запуске
    print(f"Запуск обучения модели {args.model} для задачи {args.equation}")
    print(f"Количество эпох: {args.epochs}")
    print(f"Скорость обучения: {args.lr}")
    
    # Запускаем основную функцию
    try:
        main(args)
        print("Обучение успешно завершено!")
    except Exception as e:
        print(f"Возникла ошибка: {e}")
        import traceback
        traceback.print_exc() 