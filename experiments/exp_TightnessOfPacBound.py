import torch
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable

from algorithms.optimization_algorithms import (heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates)
from problems.parametric_problems import setup_quadratic_with_variable_curvature_with_rand_perm
from pac_bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer
from pac_bayes.Prior import iterative_prior
from helper.for_estimation import empirical_convergence_risk, converged, empirical_conv_estimates
from helper.for_plotting import set_size


def specify_plot_layout() -> float:

    # Setup style for paper
    width = 234.8775    # AISTATS
    # width = 469.75499   # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    plt.rcParams.update(tex_fonts)

    return width


def specify_colors() -> Tuple[str, str, str, str]:
    color_mean = '#405f9a'
    color_pac = '#f2c05d'
    color_learned = '#0bb4ff'
    color_std = '#e60049'
    return color_std, color_learned, color_pac, color_mean


def specify_setup() -> Tuple[int, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    dimension = 50
    x_0 = torch.zeros(dimension)
    conv_prob = torch.tensor(0.9)
    eps = torch.tensor(0.01)
    num_iterations = 50

    return dimension, x_0, num_iterations, eps, conv_prob


def specify_size_of_datasets() -> Tuple[int, int, int]:
    # num_problems_prior = 200
    # num_problems_train = 1000
    # num_problems_test = 200
    num_problems_prior = 50
    num_problems_train = 100
    num_problems_test = 50
    return num_problems_prior, num_problems_train, num_problems_test


def specify_number_of_samples_from_prior() -> int:
    # num_samples_prior = 200
    num_samples_prior = 50
    return num_samples_prior


def get_samples_of_strong_convexity(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.tensor(0.05)
    mu_min, mu_max = mu.clone(), mu.clone()
    return torch.ones((n_samples,)) * mu, mu_min, mu_max


def get_samples_of_smoothness(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L_min, L_max = torch.tensor(1.0), torch.tensor(5000.0)
    samples = torch.distributions.uniform.Uniform(low=L_min, high=L_max).sample((n_samples,))
    return samples, L_min, L_max


def create_parametric_problems_with_differing_smoothness(
        dimension: int,
        num_problems_prior: int,
        num_problems_train: int,
        num_problems_test: int) -> Tuple[dict, Callable, Callable, torch.Tensor, torch.Tensor]:

    samples_smoothness, L_min, L_max = get_samples_of_smoothness(
        n_samples=num_problems_prior + num_problems_train + num_problems_test)
    samples_strong_convexity, mu_min, mu_max = get_samples_of_strong_convexity(
        n_samples=num_problems_prior + num_problems_train + num_problems_test)
    param_problem, loss_func, grad_func = setup_quadratic_with_variable_curvature_with_rand_perm(
        dim=dimension, n_prior=num_problems_prior, n_train=num_problems_train, n_test=num_problems_test,
        strong_convexity=samples_strong_convexity, smoothness=samples_smoothness)

    return param_problem, loss_func, grad_func, mu_min, L_max


def split_data_set(data: list) -> Tuple[list, list]:
    return data[0:int(len(data) / 2)], data[int(len(data)/2):]


def specify_algorithm_to_be_used(smoothness_constant: torch.Tensor,
                                 strong_convexity_constant: torch.Tensor) -> Tuple[Callable, Callable, dict]:
    q = ((torch.sqrt(smoothness_constant) - torch.sqrt(strong_convexity_constant)) /
         (torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant)))
    std_hyperparams = {
        'alpha': 4 / (torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant)) ** 2 * torch.ones(1),
        'beta': q ** 2 * torch.ones(1)}     # torch.ones(1) is just for the shape of the hyperparameters.
    algorithm, algorithm_with_iterates = heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
    return algorithm, algorithm_with_iterates, std_hyperparams


def get_estimators(algorithm: Callable,
                   loss_func: Callable,
                   grad_func: Callable,
                   x_0: torch.Tensor,
                   num_iterations: int) -> Tuple[Callable, Callable, Callable]:
    emp_conv_risk = empirical_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations)
    conv_check = converged(algorithm, loss_func, grad_func, x_0, num_iterations)
    emp_conv_est = empirical_conv_estimates(algorithm, loss_func, grad_func, x_0, num_iterations)
    return emp_conv_risk, conv_check, emp_conv_est


def get_sufficient_statistics(emp_conv_est: Callable, num_problems_train: int) -> Callable:

    def f(hyperparam, data):
        e_conv_risk, e_sq_conv_risk, e_init_loss, e_init_sq_loss, e_conv_prob = emp_conv_est(hyperparam, data)
        return torch.tensor([-e_conv_risk, e_init_sq_loss / num_problems_train / e_conv_prob ** 2])

    return f


def get_natural_parameters() -> Callable:
    return lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])


def define_prior_distribution_for_current_convergence_probability(
        smoothness_constant: torch.Tensor, conv_prob: torch.Tensor, std_hyperparams: dict) -> Tuple[dict, dict]:
    priors = {'alpha': torch.distributions.uniform.Uniform, 'beta': torch.distributions.uniform.Uniform}
    prior = {
        'alpha': torch.distributions.uniform.Uniform(
            low=0.5 * (2 / (smoothness_constant * conv_prob)) * torch.ones(1),
            high=3.0 * (2 / (smoothness_constant * conv_prob)) * torch.ones(1)),
        'beta': torch.distributions.uniform.Uniform(
            low=0.5 * std_hyperparams['beta'].clone(),
            high=2.0 * std_hyperparams['beta'].clone())}
    prior_params = {
        'alpha': {'low': 0.5 * (2 / (smoothness_constant * conv_prob)) * torch.ones(1),
                  'high': 2.0 * (2 / (smoothness_constant * conv_prob)) * torch.ones(1)},
        'beta': {'low': 0.5 * std_hyperparams['beta'].clone(),
                 'high': 2.0 * std_hyperparams['beta'].clone()}}
    estimators = {
        'alpha': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)},
        'beta': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)}}

    prior_dict = {'priors': priors, 'prior_params': prior_params, 'estimator': estimators}
    return prior, prior_dict


def create_data_dependent_prior(
        smoothness_constant: torch.Tensor,
        conv_prob: torch.Tensor,
        std_hyperparams: dict,
        suff_stat: Callable,
        nat_param: Callable,
        num_samples_prior: int,
        data: list,
        test_data: list,
        eps: torch.Tensor,
        conv_check: Callable) -> dict:

    prior, prior_dict = define_prior_distribution_for_current_convergence_probability(
        smoothness_constant=smoothness_constant, conv_prob=conv_prob, std_hyperparams=std_hyperparams)

    # Create data-dependent prior on prior-dataset
    prior, list_of_priors = iterative_prior(prior=prior, prior_dict=prior_dict,
                                            suff_stat=suff_stat, nat_param=nat_param,
                                            num_samples_prior=num_samples_prior,
                                            data=data,
                                            batch_size_opt_lamb=len(data),
                                            eps=eps, num_it=2,
                                            conv_check=conv_check, conv_prob=conv_prob,
                                            test_data=test_data)
    return prior


def compute_losses_over_iterations(dataset: list,
                                   loss_func: Callable,
                                   grad_func: Callable,
                                   algorithm_with_iterates: Callable,
                                   x_0: torch.Tensor,
                                   num_iterations: int,
                                   learned_hyperparameters: dict,
                                   std_hyperparameters: dict) -> Tuple[torch.Tensor, torch.Tensor]:

    losses_learned = torch.zeros((len(dataset), num_iterations + 1))
    losses_std = torch.zeros((len(dataset), num_iterations + 1))
    convergence = 0

    # Compare on test data: First, check for divergence, and, if it did not diverge, compare to standard algorithm.
    for i, p in enumerate(dataset):

        it_learned = torch.stack(
            [loss_func(a, p) for a in
             algorithm_with_iterates(x_0, p, grad_func, learned_hyperparameters, num_iterations)])
        if it_learned[-1] > it_learned[0] or torch.isnan(it_learned[-1]):
            continue
        else:
            losses_std[i, :] = torch.stack(
                [loss_func(a, p) for a in
                 algorithm_with_iterates(x_0, p, grad_func, std_hyperparameters, num_iterations)])
            losses_learned[i, :] = it_learned
            convergence += 1

    return losses_learned, losses_std


def finalize_plot(ax, losses_learned: NDArray, losses_std: NDArray, pac_bound: torch.Tensor):

    color_std, color_learned, color_pac, color_mean = specify_colors()
    bins = np.logspace(np.log10(np.minimum(np.min(losses_learned[:, -1]),
                                           np.min(losses_std[:, -1]))),
                       np.log10(np.maximum(np.max(losses_learned[:, -1]),
                                           np.max(losses_std[:, -1]))), 50)

    ax.hist(losses_std[:, -1], bins=bins, label='$\\alpha_{\\rm{std}}$', color=color_std, edgecolor='black')
    ax.hist(losses_learned[:, -1], bins=bins, label='$\\alpha_{\\rm{pac}}$', color=color_learned, edgecolor='black')
    ax.axvline(np.mean(losses_learned[:, -1]), 0, 1, color=color_mean, linestyle='dashed',
               label='$\\hat{\\mathcal{R}}_{\\rm{test}}(\\alpha_{\\rm{pac}})$')
    ax.axvline(pac_bound, 0, 1, color=color_pac, linestyle='dotted', label='PAC-Bound')
    ax.set(xlabel='$l(\\alpha, \\theta_i)$')
    ax.grid('on')
    ax.set_xscale('log')
    ax.legend()


def exp_loss_histogram_with_pac_bound():

    dimension, x_0, num_iterations, eps, conv_prob = specify_setup()
    num_problems_prior, num_problems_train, num_problems_test = specify_size_of_datasets()
    num_samples_prior = specify_number_of_samples_from_prior()

    param_problem, loss_func, grad_func, mu_min, L_max = create_parametric_problems_with_differing_smoothness(
        dimension=dimension, num_problems_prior=num_problems_prior, num_problems_train=num_problems_train,
        num_problems_test=num_problems_test)
    data_prior_creation, data_prior_test = split_data_set(param_problem['prior'])
    algorithm, algorithm_with_iterates, std_hyperparams = specify_algorithm_to_be_used(smoothness_constant=L_max,
                                                                                       strong_convexity_constant=mu_min)

    emp_conv_risk, conv_check, emp_conv_est = get_estimators(
        algorithm=algorithm, loss_func=loss_func, grad_func=grad_func, x_0=x_0, num_iterations=num_iterations)
    sufficient_statistics = get_sufficient_statistics(emp_conv_est=emp_conv_est, num_problems_train=num_problems_train)
    natural_parameters = get_natural_parameters()

    print(std_hyperparams)
    print("Risk of std. Hyperparameters = {}".format(emp_conv_risk(std_hyperparams, param_problem['prior'])))

    prior = create_data_dependent_prior(
        smoothness_constant=L_max, conv_prob=conv_prob, std_hyperparams=std_hyperparams,
        suff_stat=sufficient_statistics, nat_param=natural_parameters, num_samples_prior=num_samples_prior,
        data=data_prior_creation, eps=eps, conv_check=conv_check, test_data=data_prior_test)

    (learned_hyperparameters,
     pac_bound,
     samples_prior,
     log_prior_density,
     log_prior_marginals,
     log_posterior_density) = pac_bayes_optimizer(suff_stat=sufficient_statistics, nat_param=natural_parameters,
                                                  priors=prior, data=param_problem['train'],
                                                  num_samples_prior=num_samples_prior,
                                                  batch_size_opt_lamb=num_problems_train, eps=eps)

    losses_learned, losses_std = compute_losses_over_iterations(
        dataset=param_problem['test'], loss_func=loss_func, grad_func=grad_func,
        algorithm_with_iterates=algorithm_with_iterates, x_0=x_0, num_iterations=num_iterations,
        learned_hyperparameters=learned_hyperparameters, std_hyperparameters=std_hyperparams)

    width = specify_plot_layout()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(width))
    finalize_plot(ax=ax, losses_std=losses_std.numpy(), losses_learned=losses_learned.numpy(), pac_bound=pac_bound)
    PATH = '/home/michael/Desktop/AISTATS_2023/experiments/tightness_pac_bound/'
    plt.savefig(PATH + 'loss_histogram_with_pac_bound.pdf', dpi=300, bbox_inches='tight')
    plt.show()
