import os
import pdb
from functools import partial

import jax
import jax.numpy as jnp
import optax
import scipy.io
from networks.physics_informed_neural_networks import *
from utils.vorticity import (velocity_to_vorticity_fwd,
                             velocity_to_vorticity_rev)
import json
import torch
import torch.nn as nn


def setup_networks(args, key):
    # build network
    dim = args.equation[-2:]
    if args.model == 'pinn':
        # feature sizes
        feat_sizes = tuple([args.features for _ in range(args.n_layers - 1)] + [args.out_dim])
        if dim == '2d':
            model = PINN2d(feat_sizes)
        elif dim == '3d':
            model = PINN3d(feat_sizes, args.out_dim, args.pos_enc)
        elif dim == '4d':
            model = PINN4d(feat_sizes)
        else:
            raise NotImplementedError
    else: # SPINN
        # feature sizes
        feat_sizes = tuple([args.features for _ in range(args.n_layers)])
        if dim == '2d':
            model = SPINN2d(feat_sizes, args.r, args.mlp)
        elif dim == '3d':
            model = SPINN3d(feat_sizes, args.r, args.out_dim, args.pos_enc, args.mlp)
        elif dim == '4d':
            model = SPINN4d(feat_sizes, args.r, args.out_dim, args.mlp)
        else:
            raise NotImplementedError
    # initialize params
    # dummy inputs must be given
    if dim == '2d':
        params = model.init(
            key,
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1))
        )
    elif dim == '3d':
        if args.equation == 'navier_stokes3d':
            params = model.init(
                key,
                jnp.ones((args.nt, 1)),
                jnp.ones((args.nxy, 1)),
                jnp.ones((args.nxy, 1))
            )
        else:
            params = model.init(
                key,
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1))
            )
    elif dim == '4d':
        params = model.init(
            key,
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1))
        )
    else:
        raise NotImplementedError

    return jax.jit(model.apply), params


def name_model(args):
    name = [
        f'nl{args.n_layers}',
        f'fs{args.features}',
        f'lr{args.lr}',
        f's{args.seed}',
        f'r{args.r}'
    ]
    if args.model != 'spinn':
        del name[-1]
    if args.equation != 'navier_stokes3d':
        name.insert(0, f'nc{args.nc}')
    if args.equation == 'navier_stokes3d':
        name.insert(0, f'nxy{args.nxy}')
        name.insert(0, f'nt{args.nt}')
        name.append(f'on{args.offset_num}')
        name.append(f'oi{args.offset_iter}')
        name.append(f'lc{args.lbda_c}')
        name.append(f'lic{args.lbda_ic}')
    if args.equation == 'navier_stokes4d':
        name.append(f'lc{args.lbda_c}')
        name.append(f'li{args.lbda_ic}')
    if args.equation == 'helmholtz3d':
        name.append(f'a{args.a1}{args.a2}{args.a3}')
    if args.equation == 'klein_gordon3d':
        name.append(f'k{args.k}')
    
    name.append(f'{args.mlp}')
        
    return '_'.join(name)


def save_config(args, result_dir):
    with open(os.path.join(result_dir, 'configs.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


# single update function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


# save next initial condition for time-marching
def save_next_IC(root_dir, name, apply_fn, params, test_data, step_idx, e):
    os.makedirs(os.path.join(root_dir, name, 'IC_pred'), exist_ok=True)

    w_pred = velocity_to_vorticity_fwd(apply_fn, params, jnp.expand_dims(test_data[0][-1], axis=1), test_data[1], test_data[2])
    w_pred = w_pred.reshape(-1, test_data[1].shape[0], test_data[2].shape[0])[0]
    u0_pred, v0_pred = apply_fn(params, jnp.expand_dims(test_data[0][-1], axis=1), test_data[1], test_data[2])
    u0_pred, v0_pred = jnp.squeeze(u0_pred), jnp.squeeze(v0_pred)
    
    scipy.io.savemat(os.path.join(root_dir, name, f'IC_pred/w0_{step_idx+1}.mat'), mdict={'w0': w_pred, 'u0': u0_pred, 'v0': v0_pred, 't': jnp.expand_dims(test_data[0][-1], axis=1)})


class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_layers=3, out_dim=1, pos_enc=0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.pos_enc = pos_enc
        
        # Если используется позиционное кодирование, увеличиваем входную размерность
        if pos_enc > 0:
            input_dim = input_dim * (1 + 2 * pos_enc)
        
        # Создаем слои
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def positional_encoding(self, x):
        """Применяет позиционное кодирование к входному тензору."""
        if self.pos_enc == 0:
            return x
        
        # Предполагаем, что x имеет форму [batch_size, input_dim]
        encodings = [x]
        for i in range(1, self.pos_enc + 1):
            encodings.append(torch.sin(2**i * torch.pi * x))
            encodings.append(torch.cos(2**i * torch.pi * x))
            
        return torch.cat(encodings, dim=-1)
    
    def forward(self, t, x, y):
        # Объединяем входы
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        inputs = torch.cat([t, x, y], dim=1)
        
        # Применяем позиционное кодирование
        if self.pos_enc > 0:
            inputs = self.positional_encoding(inputs)
        
        # Пропускаем через сеть
        return self.net(inputs)

class ModifiedMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_layers=3, r=128, out_dim=1, pos_enc=0):
        super(ModifiedMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.r = r
        self.pos_enc = pos_enc
        
        # Если используется позиционное кодирование, увеличиваем входную размерность
        if pos_enc > 0:
            input_dim = input_dim * (1 + 2 * pos_enc)
        
        # Создаем первый слой
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        
        # Создаем промежуточные слои с пониженным рангом
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            # Слой пониженного ранга: hidden_dim -> r -> hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, r),
                nn.Tanh(),
                nn.Linear(r, hidden_dim),
                nn.Tanh()
            ))
        
        # Выходной слой
        self.out_layer = nn.Linear(hidden_dim, out_dim)
    
    def positional_encoding(self, x):
        """Применяет позиционное кодирование к входному тензору."""
        if self.pos_enc == 0:
            return x
        
        # Предполагаем, что x имеет форму [batch_size, input_dim]
        encodings = [x]
        for i in range(1, self.pos_enc + 1):
            encodings.append(torch.sin(2**i * torch.pi * x))
            encodings.append(torch.cos(2**i * torch.pi * x))
            
        return torch.cat(encodings, dim=-1)
    
    def forward(self, t, x, y):
        # Объединяем входы
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        inputs = torch.cat([t, x, y], dim=1)
        
        # Применяем позиционное кодирование
        if self.pos_enc > 0:
            inputs = self.positional_encoding(inputs)
        
        # Пропускаем через сеть
        x = self.fc1(inputs)
        x = self.activation(x)
        
        for layer in self.layers:
            residual = x
            x = layer(x) + residual
        
        return self.out_layer(x)

def setup_networks(args):
    """Создает и инициализирует нейронную сеть с заданными параметрами."""
    if args.mlp == 'mlp':
        model = MLP(
            input_dim=3,  # t, x, y
            hidden_dim=args.features, 
            n_layers=args.n_layers,
            out_dim=args.out_dim,
            pos_enc=args.pos_enc
        )
    elif args.mlp == 'modified_mlp':
        model = ModifiedMLP(
            input_dim=3,  # t, x, y
            hidden_dim=args.features,
            n_layers=args.n_layers,
            r=args.r,
            out_dim=args.out_dim,
            pos_enc=args.pos_enc
        )
    else:
        raise ValueError(f"Unknown MLP type: {args.mlp}")
    
    return model

def name_model(args):
    """Создает имя модели на основе параметров."""
    model_name = f"{args.mlp}_L{args.n_layers}_F{args.features}"
    if args.mlp == 'modified_mlp':
        model_name += f"_R{args.r}"
    
    if args.pos_enc > 0:
        model_name += f"_PE{args.pos_enc}"
    
    return model_name

def save_config(args, result_dir):
    """Сохраняет конфигурацию обучения в JSON файл."""
    config = vars(args)
    # Преобразуем непреобразуемые типы в строки
    for key, value in config.items():
        if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
            config[key] = str(value)
    
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)