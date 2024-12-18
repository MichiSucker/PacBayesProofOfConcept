from typing import Tuple, Callable
import torch
import numpy as np


def setup_random_quadratic_problems_with_fixed_curvature(
        dim: int,
        n_prior: int, n_train: int, n_test: int) -> Tuple[dict, Callable, Callable, torch.Tensor, torch.Tensor]:

    # Shape of quadratic
    A = torch.distributions.uniform.Uniform(-10, 10).sample((dim, dim))
    eigenvalues = torch.linalg.eigvalsh(A.T @ A)
    lamb_min, lamb_max = eigenvalues[0], eigenvalues[-1]
    assert 0 < lamb_min < lamb_max

    # Sample rhs of quadratic problem
    mean = torch.distributions.uniform.Uniform(-5, 5).sample((dim, ))
    cov = torch.distributions.uniform.Uniform(-5, 5).sample((dim, dim))
    cov = torch.transpose(cov, 0, 1) @ cov

    B_train = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_train,))
    B_prior = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_prior,))
    B_test = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_test,))

    # Specify loss function
    def loss_function(x, parameter):
        return 0.5 * torch.linalg.norm(torch.matmul(parameter['A'], x) - parameter['b']) ** 2

    def grad_function(x, parameter):
        return torch.matmul(torch.t(parameter['A']), torch.matmul(parameter['A'], x) - parameter['b'])

    # Setup dict with all the parameters
    parameters = {
        'prior': np.array([{'A': A, 'b': B_prior[i, :]} for i in range(n_prior)]),
        'train': np.array([{'A': A, 'b': B_train[i, :]} for i in range(n_train)]),
        'test': np.array([{'A': A, 'b': B_test[i, :]} for i in range(n_test)])
    }

    return parameters, loss_function, grad_function, lamb_min, lamb_max


def setup_quadratic_with_variable_curvature_with_rand_perm(
        dim: int, n_prior: int, n_train: int, n_test: int,
        strong_convexity: torch.Tensor, smoothness: torch.Tensor) -> Tuple[dict, Callable, Callable]:

    # Create diagonal of quadratic matrix of the quadratic problem
    diagonal = [torch.linspace(torch.sqrt(strong_convexity[i]).item(),
                               torch.sqrt(smoothness[i]).item(), dim)[torch.randperm(dim)]
                for i in range(n_prior + n_train + n_test)]
    diagonal_prior = diagonal[:n_prior]
    diagonal_train = diagonal[n_prior:n_prior + n_train]
    diagonal_test = diagonal[n_prior + n_train:]

    # Sample rhs of quadratic problem
    mean = torch.distributions.uniform.Uniform(-5, 5).sample((dim, ))
    cov = torch.distributions.uniform.Uniform(-5, 5).sample((dim, dim))
    cov = torch.transpose(cov, 0, 1) @ cov

    B_train = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_train,))
    B_prior = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_prior,))
    B_test = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((n_test,))

    # Specify loss function
    def loss_function(x, parameter):
        return 0.5 * torch.linalg.norm(torch.matmul(parameter['A'], x) - parameter['b']) ** 2

    def grad_function(x, parameter):
        return torch.matmul(torch.t(parameter['A']), torch.matmul(parameter['A'], x) - parameter['b'])

    # Setup dict with all the parameters
    param_problem = {
        'prior': np.array([{'A': torch.diag(diagonal_prior[i]), 'b': B_prior[i, :]} for i in range(n_prior)]),
        'train': np.array([{'A': torch.diag(diagonal_train[i]), 'b': B_train[i, :]} for i in range(n_train)]),
        'test': np.array([{'A': torch.diag(diagonal_test[i]), 'b': B_test[i, :]} for i in range(n_test)])
    }

    return param_problem, loss_function, grad_function
