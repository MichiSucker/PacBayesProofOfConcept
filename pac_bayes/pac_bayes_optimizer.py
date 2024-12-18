import numpy as np
import torch
from tqdm import tqdm
from typing import Callable, Tuple


def get_upper_bound_in_lambda(empirical_sufficient_statistics: torch.Tensor,
                              natural_parameters: Callable,
                              eps: torch.Tensor,
                              num_test_points: torch.Tensor) -> Callable:
    return lambda lamb: -(torch.log(torch.mean(
        torch.exp(torch.matmul(empirical_sufficient_statistics, natural_parameters(lamb)))))
                          + torch.log(eps) - torch.log(num_test_points)) / natural_parameters(lamb)[0]


def get_log_posterior_density_on_samples(log_prior_density: torch.Tensor,
                                         empirical_sufficient_statistics: torch.Tensor,
                                         natural_parameters: torch.Tensor) -> torch.Tensor:
    return (log_prior_density
            + torch.matmul(empirical_sufficient_statistics, natural_parameters).reshape(log_prior_density.shape)
            - torch.log(torch.mean(torch.exp(torch.matmul(empirical_sufficient_statistics, natural_parameters)))))


def optimize_upper_bound_in_lambda(empirical_sufficient_statistics: torch.Tensor,
                                   natural_parameters: Callable,
                                   eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # Note that this is a one-dimensional function, so computationally this is not too much of a problem.
    num_test_points = torch.tensor(25000)
    F = get_upper_bound_in_lambda(empirical_sufficient_statistics, natural_parameters, eps, num_test_points)
    test_points = torch.linspace(start=1e-8, end=1., steps=num_test_points.item())
    F_test = torch.stack([F(x) for x in test_points])
    optimal_lambda = test_points[torch.argmin(F_test)]

    return optimal_lambda, F(optimal_lambda)


def get_samples_from_prior(priors, num_samples_prior):
    # Note that an independence assumption is made here: the joint density of the hyperparameter factorizes into
    # the marginal densities.
    samples_prior = {hyperparam: priors[hyperparam].sample((num_samples_prior,)) for hyperparam in priors.keys()}
    log_prior_marginals = {hyperparam: priors[hyperparam].log_prob(samples_prior[hyperparam])
                           for hyperparam in priors.keys()}
    return samples_prior, log_prior_marginals


def evaluate_sufficient_statistics(sufficient_statistics: Callable,
                                   data: list,
                                   samples_prior: dict,
                                   num_samples_prior: int,
                                   num_problems: int,
                                   batch_size_opt_lamb: int) -> torch.Tensor:
    # Compute sufficient statistic on "enough" sample from the prior
    # Note that computing the emp_suff_stat is (one of) the most expensive parts of the algorithm, especially if
    # samples_prior is large: the algorithm has to run n_samples * N_train times!
    # ==> Thus, could use (Batch-) SGD. But this does not retain the PAC-Bound!
    # Compute empirical sufficient statistics
    pbar = tqdm(range(num_samples_prior))
    pbar.set_description('Computing Sufficient Statistics')
    emp_suff_stat = torch.stack(
        [sufficient_statistics(
            {key: value[i] for key, value in samples_prior.items()},
            data[np.random.choice(np.arange(0, num_problems), replace=False, size=batch_size_opt_lamb)])
         for i in pbar])

    # Filter inf and -inf
    emp_suff_stat[torch.isinf(emp_suff_stat[:, 0]), 0] = -1e7  # First row is negative
    emp_suff_stat[torch.isnan(emp_suff_stat[:, 0]), 0] = -1e7
    emp_suff_stat[torch.isinf(emp_suff_stat[:, 1]), 1] = 1e7   # Second row is positive
    emp_suff_stat[torch.isnan(emp_suff_stat[:, 1]), 1] = 1e7

    return emp_suff_stat


def compute_posterior_density(log_prior_marginals: dict,
                              empirical_sufficient_statistics: torch.Tensor,
                              optimal_natural_parameters: torch.Tensor,
                              priors: dict,
                              samples_prior: dict) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    # Compute posterior density
    log_prior_density = torch.sum(torch.stack([p for p in log_prior_marginals.values()]), dim=0)
    log_f_post = get_log_posterior_density_on_samples(log_prior_density,
                                                      empirical_sufficient_statistics,
                                                      optimal_natural_parameters)

    # Normalize to get density over samples_prior
    log_f_post = log_f_post - torch.log(torch.sum(torch.exp(log_f_post)))
    log_prior_marginals = {hyperparam: priors[hyperparam].log_prob(samples_prior[hyperparam]) - torch.log(
        torch.sum(torch.exp(priors[hyperparam].log_prob(samples_prior[hyperparam])))) for hyperparam in priors.keys()
                           }

    return log_prior_density, log_prior_marginals, log_f_post


def select_best_hyperparameter(log_f_post: torch.Tensor, samples_prior: dict) -> dict:
    idx_learned = torch.argmax(log_f_post)
    hyperparams_learned = {hyperparam: samples_prior[hyperparam][idx_learned] for hyperparam in samples_prior.keys()}
    return hyperparams_learned


def pac_bayes_optimizer(sufficient_statistics: Callable,
                        natural_parameters: Callable,
                        priors: dict,
                        data: list,
                        num_samples_prior: int,
                        batch_size_opt_lamb: int,
                        eps: torch.Tensor) -> Tuple[dict, torch.Tensor, dict, torch.Tensor, dict, torch.Tensor]:

    num_problems = len(data)
    assert batch_size_opt_lamb <= len(data), "Batch size larger than training set."

    samples_prior, log_prior_marginals = get_samples_from_prior(priors, num_samples_prior)
    empirical_sufficient_statistics = evaluate_sufficient_statistics(sufficient_statistics=sufficient_statistics,
                                                                     data=data, samples_prior=samples_prior,
                                                                     num_samples_prior=num_samples_prior,
                                                                     num_problems=num_problems,
                                                                     batch_size_opt_lamb=batch_size_opt_lamb)
    lamb_opt, pac_bound = optimize_upper_bound_in_lambda(empirical_sufficient_statistics, natural_parameters, eps)
    log_prior_density, log_prior_marginals, log_f_post = compute_posterior_density(
        log_prior_marginals=log_prior_marginals, empirical_sufficient_statistics=empirical_sufficient_statistics,
        optimal_natural_parameters=natural_parameters(lamb_opt), priors=priors, samples_prior=samples_prior)
    hyperparams_learned = select_best_hyperparameter(log_f_post, samples_prior)

    return hyperparams_learned, pac_bound, samples_prior, log_prior_density, log_prior_marginals, log_f_post
