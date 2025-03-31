import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from utils.data_generators import generate_test_data_pytorch, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *


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


def apply_model_spinn(model, params, *train_data):
    def residual_loss(t, x, y, a, b):
        # Убеждаемся, что входные тензоры требуют градиентов
        if not t.requires_grad:
            t.requires_grad_(True)
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Вычисляем производные
        u = model(t, x, y)
        
        # 1-я производная по t
        ut = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 1-я производная по x
        ux = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 1-я производная по y
        uy = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return torch.mean((ut + a*ux + b*uy)**2) 

    def initial_loss(t, x, y, u_true):
        u_pred = model(t, x, y)
        return torch.mean((u_pred - u_true)**2)

    def boundary_loss(t, x, y, u_true):
        loss = 0.
        for i in range(4):
            u_pred = model(t[i], x[i], y[i])
            loss += torch.mean((u_pred - u_true[i])**2)
        return loss

    # распаковываем данные
    tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

    # вычисляем функцию потерь
    loss_residual = 10 * residual_loss(tc, xc, yc, a, b)
    loss_initial = initial_loss(ti, xi, yi, ui)
    loss_boundary = boundary_loss(tb, xb, yb, ub)
    
    total_loss = loss_residual + loss_initial + loss_boundary

    return total_loss, (loss_residual, loss_initial, loss_boundary)


def apply_model_pinn(model, params, *train_data):
    def residual_loss(t, x, y, a, b):
        # Убеждаемся, что входные тензоры требуют градиентов
        if not t.requires_grad:
            t.requires_grad_(True)
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Вычисляем u
        u = model(t, x, y)
        
        # 1-я производная по t
        ut = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 1-я производная по x
        ux = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 1-я производная по y
        uy = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return torch.mean((ut + a*ux + b*uy)**2)

    def initial_boundary_loss(t, x, y, u_true):
        u_pred = model(t, x, y)
        return torch.mean((u_pred - u_true)**2)
    
    # распаковываем данные
    tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

    # вычисляем функцию потерь
    loss_residual = 10 * residual_loss(tc, xc, yc, a, b)
    loss_initial = initial_boundary_loss(ti, xi, yi, ui)
    loss_boundary = initial_boundary_loss(tb, xb, yb, ub)
    
    total_loss = loss_residual + loss_initial + loss_boundary

    return total_loss, (loss_residual, loss_initial, loss_boundary)


def update_model(optimizer, loss, model):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model


def main(args):
    """Основная функция для запуска обучения."""
    # установка случайного зерна
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # определяем устройство для вычислений
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # создаем и инициализируем модель
    model = setup_networks(args).to(device)

    # подсчет общего числа параметров
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Общее число параметров модели: {args.total_params}")

    # создаем имя модели
    name = name_model(args)

    # путь к директории результатов
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)

    # создаем директорию
    os.makedirs(result_dir, exist_ok=True)
    print(f"Результаты будут сохранены в: {result_dir}")

    # оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # датасеты для обучения
    train_data = generate_train_data(args)
    train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                 [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                 for t in train_data]
    
    # датасеты для тестирования
    test_data = generate_test_data_pytorch(args, result_dir)
    test_data = [t.to(device) if isinstance(t, torch.Tensor) else t for t in test_data]

    # функция оценки
    eval_fn = setup_eval_function(args.model, args.equation)

    # сохраняем конфигурацию обучения
    save_config(args, result_dir)

    # логи
    logs = []
    if os.path.exists(os.path.join(result_dir, 'log (loss, error).csv')):
        os.remove(os.path.join(result_dir, 'log (loss, error).csv'))
    if os.path.exists(os.path.join(result_dir, 'best_error.csv')):
        os.remove(os.path.join(result_dir, 'best_error.csv'))
    best = 100000.
    best_error = 100000.  # Инициализируем на случай, если ни одно значение не будет меньше best

    # начинаем обучение
    print(f"Начинаем обучение на {args.epochs} эпохах...")
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # исключаем время компиляции
            start = time.time()

        if e % 100 == 0:
            # берем новые входные данные
            train_data = generate_train_data(args)
            train_data = [t.to(device) if isinstance(t, torch.Tensor) else 
                         [tensor.to(device) for tensor in t] if isinstance(t, list) else t 
                         for t in train_data]

        # одна итерация обучения
        model.train()
        if args.model == 'spinn':
            loss, component_losses = apply_model_spinn(model, None, *train_data)
        elif args.model == 'pinn':
            loss, component_losses = apply_model_pinn(model, None, *train_data)
        
        update_model(optimizer, loss, model)

        if e % 10 == 0:
            model.eval()
            with torch.no_grad():
                if loss < best:
                    best = loss
                    best_error = eval_fn(model, *test_data)

        # логируем результаты
        if e % args.log_iter == 0:
            model.eval()
            with torch.no_grad():
                error = eval_fn(model, *test_data)
                print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, best error {best_error:.8f}')
                with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                    f.write(f'{loss.item()}, {error}, {best_error}\n')

    # обучение завершено
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    
    # сохраняем модель
    torch.save(model.state_dict(), os.path.join(result_dir, 'model.pt'))
    
    # сохраняем время выполнения
    runtime = np.array([runtime])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')

    # сохраняем лучшую ошибку
    with open(os.path.join(result_dir, 'best_error.csv'), 'a') as f:
        f.write(f'best error: {best_error}\n')


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--equation', type=str, default='flow_mixing3d', help='equation to solve')

    # input data settings
    parser.add_argument('--nc', type=int, default=64, help='the number of input points for each axis')
    parser.add_argument('--nc_test', type=int, default=100, help='the number of test points for each axis')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # model settings
    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=3, help='the number of layer')
    parser.add_argument('--features', type=int, default=64, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of a approximated tensor')
    parser.add_argument('--out_dim', type=int, default=1, help='size of model output')
    parser.add_argument('--pos_enc', type=int, default=0, help='size of the positional encoding (zero if no encoding)')

    # PDE settings
    parser.add_argument('--vmax', type=float, default=0.385, help='maximum tangential velocity')

    # log settings
    parser.add_argument('--log_iter', type=int, default=5000, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=50000, help='plot result every...')

    args = parser.parse_args()
    
    # Запускаем основную функцию
    main(args)

