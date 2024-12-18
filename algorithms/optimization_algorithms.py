from typing import Callable
import torch


def gradient_descent_const_step(x: torch.Tensor,
                                param: dict,
                                grad_func: Callable,
                                hyperparam: dict,
                                num_iter: int) -> torch.Tensor:
    for i in range(num_iter):
        x = x - hyperparam['alpha'] * grad_func(x, param)
    return x


def gradient_descent_const_step_with_iterates(x: torch.Tensor,
                                              param: dict,
                                              grad_func: Callable,
                                              hyperparam: dict,
                                              num_iter: int) -> torch.Tensor:
    iterates = [x]
    for i in range(num_iter):
        x = x - hyperparam['alpha'] * grad_func(x, param)
        iterates.append(x)
    return torch.stack(iterates)


def heavy_ball_const_hyperparam(x: torch.Tensor,
                                param: dict,
                                grad_func: Callable,
                                hyperparam: dict,
                                num_iter: int) -> torch.Tensor:
    x_old = x
    x_cur = x
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'] * grad_func(x_cur, param) + hyperparam['beta'] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
    return x_cur


def heavy_ball_const_hyperparam_with_iterates(x: torch.Tensor,
                                              param: dict,
                                              grad_func: Callable,
                                              hyperparam: dict,
                                              num_iter: int) -> torch.Tensor:
    x_old = x
    x_cur = x
    iterates = [x]
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'] * grad_func(x_cur, param) + hyperparam['beta'] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
        iterates.append(x_cur)
    return torch.stack(iterates)
