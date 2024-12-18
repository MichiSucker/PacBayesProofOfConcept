import torch
from typing import Callable, Tuple


def empirical_risk(algorithm: Callable,
                   loss_func: Callable,
                   grad_func: Callable,
                   x_0: torch.Tensor,
                   num_iterations: int) -> Callable:
    return lambda hyperparam, data: torch.mean(
        torch.stack([loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
                     for i in range(data.shape[0])]))


def empirical_sq_risk(algorithm: Callable,
                      loss_func: Callable,
                      grad_func: Callable,
                      x_0: torch.Tensor,
                      num_iterations: int) -> Callable:
    return lambda hyperparam, data: torch.mean(
        torch.stack([loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) ** 2
                     for i in range(data.shape[0])]))


def empirical_conv_estimates(algorithm: Callable,
                             loss_func: Callable,
                             grad_func: Callable,
                             x_0: torch.Tensor,
                             num_iterations: int) -> Callable:

    def f(hyperparam: dict, data: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        losses, sq_losses, init_loss, init_sq_loss, convergence = [], [], [], [], []

        for i in range(len(data)):
            cur_loss = loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
            cur_init_loss = loss_func(x_0, data[i])
            convergence.append(torch.tensor(1.0) if cur_loss <= cur_init_loss else torch.tensor(0.0))
            losses.append(cur_loss if convergence[-1] == 1 else torch.tensor(0.0))
            sq_losses.append(cur_loss**2 if convergence[-1] == 1 else torch.tensor(0.0))
            init_loss.append(cur_init_loss)
            init_sq_loss.append(cur_init_loss ** 2)

        return (torch.mean(torch.stack(losses)) / torch.mean(torch.stack(convergence)),
                torch.mean(torch.stack(sq_losses)) / torch.mean(torch.stack(convergence))**2,
                torch.mean(torch.stack(init_loss)),
                torch.mean(torch.stack(init_sq_loss)),
                torch.mean(torch.stack(convergence)))

    return f


def empirical_convergence_risk(algorithm: Callable,
                               loss_func: Callable,
                               grad_func: Callable,
                               x_0: torch.Tensor,
                               num_iterations: int) -> Callable:

    def f(hyperparam, data):
        losses, conv = [], []
        for i in range(len(data)):
            cur_loss = loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
            conv.append(torch.tensor(1.0) if cur_loss <= loss_func(x_0, data[i]) else torch.tensor(0.0))
            losses.append(cur_loss if conv[-1] == 1 else torch.tensor(0.0))
        return torch.mean(torch.stack(losses)) / torch.mean(torch.stack(conv))

    return f


def converged(algorithm: Callable,
              loss_func: Callable,
              grad_func: Callable,
              x_0: torch.Tensor,
              num_iterations: int) -> Callable:
    return lambda hyperparam, data: torch.stack(
        [torch.tensor(1.0)
         if loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) <= loss_func(x_0, data[i])
         else torch.tensor(0.0)
         for i in range(data.shape[0])])



