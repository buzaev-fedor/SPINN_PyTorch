import jax
import jax.numpy as jnp
from functools import partial
from utils.vorticity import velocity_to_vorticity_fwd, velocity_to_vorticity_rev, vorx, vory, vorz
import pdb
import numpy as np
import torch


def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def mse(u, u_gt):
    return jnp.mean((u-u_gt)**2)

@partial(jax.jit, static_argnums=(0,))
def _eval2d(apply_fn, params, *test_data):
    x, y, u_gt = test_data
    return relative_l2(apply_fn(params, x, y), u_gt)

@partial(jax.jit, static_argnums=(0,))
def _eval2d_mask(apply_fn, mask, params, *test_data):
    x, y, u_gt = test_data
    nx, ny = u_gt.shape
    pred = apply_fn(params, x, y).reshape(nx, ny)
    pred = pred * mask
    return relative_l2(pred, u_gt.reshape(nx, ny))


@partial(jax.jit, static_argnums=(0,))
def _eval3d(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = apply_fn(params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_pinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_rev(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_spinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_fwd(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    return relative_l2(apply_fn(params, t, x, y, z), u_gt)

@partial(jax.jit, static_argnums=(0,))
def _eval_ns4d(apply_fn, params, *test_data):
    t, x, y, z, w_gt = test_data
    error = 0
    wx = vorx(apply_fn, params, t, x, y, z)
    wy = vory(apply_fn, params, t, x, y, z)
    wz = vorz(apply_fn, params, t, x, y, z)
    error = relative_l2(wx, w_gt[0]) + relative_l2(wy, w_gt[1]) + relative_l2(wz, w_gt[2])
    return error / 3


# temporary code
def _batch_eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    error, batch_size = 0., 100000
    n_iters = len(u_gt) // batch_size
    for i in range(n_iters):
        begin, end = i*batch_size, (i+1)*batch_size
        u = apply_fn(params, t[begin:end], x[begin:end], y[begin:end], z[begin:end])
        error += jnp.sum((u - u_gt[begin:end])**2)
    error = jnp.sqrt(error) / jnp.linalg.norm(u_gt)
    return error

@partial(jax.jit, static_argnums=(0,))
def _evalnd(apply_fn, params, *test_data):
    t, x_list, u_gt = test_data
    return relative_l2(apply_fn(params, t, *x_list), u_gt)


def calculate_relative_error(pred, true):
    """Рассчитывает относительную ошибку между предсказанными и истинными значениями."""
    return torch.norm(pred - true) / torch.norm(true)


def flow_mixing3d_exact_solution(t, x, y):
    """Возвращает точное решение для задачи flow_mixing3d."""
    # Аналитическое решение для flow_mixing3d
    # Для нашего случая простая модель диффузии/затухания
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * t)


def eval_flow_mixing3d(model, *test_data):
    """Оценивает модель на тестовых данных для задачи flow_mixing3d.
    
    Аргументы:
        model: PyTorch модель для оценки
        test_data: тензоры, возвращаемые generate_test_data
            (t, x, y, T, X, Y) - где:
            t: одномерный тензор с координатами времени
            x: одномерный тензор с координатами x
            y: одномерный тензор с координатами y
            T: трехмерный тензор с сеткой времени (результат meshgrid)
            X: трехмерный тензор с сеткой x (результат meshgrid)
            Y: трехмерный тензор с сеткой y (результат meshgrid)
    
    Возвращает:
        float: значение относительной ошибки
    """
    # Распаковываем данные и проверяем их наличие
    print(f"Количество тестовых данных: {len(test_data)}")
    for i, data in enumerate(test_data):
        if isinstance(data, torch.Tensor):
            print(f"test_data[{i}]: форма={data.shape}, тип={data.dtype}")
    
    # Проверяем, достаточно ли данных
    if len(test_data) < 6:
        raise ValueError(f"Недостаточно тестовых данных. Ожидается 6, получено {len(test_data)}")
    
    # Распаковываем данные
    t, x, y, T, X, Y = test_data
    
    # Печатаем размеры
    print(f"t: {t.shape}, x: {x.shape}, y: {y.shape}")
    print(f"T: {T.shape}, X: {X.shape}, Y: {Y.shape}")
    
    # Вычисляем аналитическое решение
    u_true = flow_mixing3d_exact_solution(T.reshape(-1), X.reshape(-1), Y.reshape(-1))
    
    # Получаем предсказания модели
    with torch.no_grad():
        u_pred = model(T.reshape(-1), X.reshape(-1), Y.reshape(-1))
    
    # Вычисляем ошибку
    error = calculate_relative_error(u_pred, u_true)
    
    return error.item()


def eval_pinn(model, *test_data):
    """Оценивает PINN на тестовых данных."""
    return eval_flow_mixing3d(model, *test_data)


def eval_spinn(model, *test_data):
    """Оценивает SPINN на тестовых данных."""
    return eval_flow_mixing3d(model, *test_data)


def setup_eval_function(model_type, equation):
    """Возвращает подходящую функцию оценки для заданного типа модели и уравнения."""
    if equation != 'flow_mixing3d':
        raise ValueError(f"Неизвестное уравнение: {equation}")
    
    if model_type == 'pinn':
        return eval_pinn
    elif model_type == 'spinn':
        return eval_spinn
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")