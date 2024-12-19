import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable

from algorithms.optimization_algorithms import heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
from problems.parametric_problems import setup_quadratic_with_variable_curvature_with_rand_perm
from pac_bayes.pac_bayes_optimizer import pac_bayes_optimizer
from pac_bayes.data_dependent_prior import iterative_prior
from helper.for_estimation import empirical_convergence_risk,  converged, empirical_conv_estimates
from helper.for_plotting import set_size


def specify_plot_layout() -> float:

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
        "ytick.labelsize": 7
    }

    plt.rcParams.update(tex_fonts)
    return width


def specify_colors() -> Tuple[str, str]:
    linecolor = "#00203f"
    dot_color = "#adefd1"
    return linecolor, dot_color


def specify_convergence_probabilities() -> torch.Tensor:
    convergence_probabilities = torch.arange(0.05, 0.96, 0.01)
    return convergence_probabilities


def specify_setup() -> Tuple[int, torch.Tensor, torch.Tensor, int]:
    dimension = 50
    x_0 = torch.zeros(dimension)
    eps = torch.tensor(0.01)
    num_iterations = 50
    return dimension, x_0, eps, num_iterations


def specify_number_of_samples_from_prior() -> int:
    num_samples_prior = 150
    return num_samples_prior


def specify_size_of_datasets() -> Tuple[int, int, int]:
    num_problems_prior = 100
    num_problems_train = 100
    num_problems_test = 5000
    return num_problems_prior, num_problems_train, num_problems_test


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


def get_natural_parameters() -> Callable:
    return lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])


def get_sufficient_statistics(emp_conv_est: Callable, num_problems_train: int) -> Callable:

    def f(hyperparam, data):
        e_conv_risk, e_sq_conv_risk, e_init_loss, e_init_sq_loss, e_conv_prob = emp_conv_est(hyperparam, data)
        return torch.tensor([-e_conv_risk, e_init_sq_loss / num_problems_train / e_conv_prob ** 2])

    return f


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
    prior, list_of_priors = iterative_prior(prior=prior, prior_dict=prior_dict, sufficient_statistics=suff_stat,
                                            natural_parameters=nat_param, num_samples_prior=num_samples_prior,
                                            data=data, batch_size_opt_lamb=len(data), eps=eps, num_it=2,
                                            conv_check=conv_check, convergence_probability=conv_prob,
                                            test_data=test_data)
    return prior


def create_subsample_from_dataset(dataset, size):
    subsample_idx = np.random.randint(0, len(dataset), size=size)
    return dataset[subsample_idx]


def compute_convergence_probability(dataset: list,
                                    loss_func: Callable,
                                    grad_func: Callable,
                                    algorithm_with_iterates: Callable,
                                    x_0: torch.Tensor,
                                    learned_hyperparameters: dict,
                                    num_iterations: int) -> float:
    convergence = 0
    for p in dataset:
        it_learned = torch.stack(
            [loss_func(a, p) for a in
             algorithm_with_iterates(x_0, p, grad_func, learned_hyperparameters, num_iterations)])
        if it_learned[-1] > it_learned[0] or torch.isnan(it_learned[-1]):
            continue
        else:
            convergence += 1
    conv_prob_test = convergence / len(dataset)
    return conv_prob_test


def add_result_to_plot(ax,
                       conv_prob: torch.Tensor,
                       number_of_trials: int,
                       empirical_conv_probabilities: list,
                       dot_color: str) -> None:
    ax.scatter([conv_prob] * number_of_trials, empirical_conv_probabilities, marker='x', s=5, color=dot_color)


def finalize_plot(ax, linecolor: str) -> None:
    ax.plot(np.arange(0.0, 1.1, 0.1), np.arange(0.0, 1.1, 0.1), linestyle='dashed', color=linecolor)
    ax.set(xlabel='$\\epsilon_{\\rm{conv}}$', ylabel='$\\hat{p}(\\alpha)$')
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.grid('on')


def exp_heavy_ball_convergence_probability():

    convergence_probabilities = specify_convergence_probabilities()
    dimension, x_0, eps, num_iterations = specify_setup()
    num_samples_prior = specify_number_of_samples_from_prior()
    num_problems_prior, num_problems_train, num_problems_test = specify_size_of_datasets()

    param_problem, loss_func, grad_func, mu_min, L_max = create_parametric_problems_with_differing_smoothness(
        dimension=dimension, num_problems_prior=num_problems_prior, num_problems_train=num_problems_train,
        num_problems_test=num_problems_test)
    data_prior_creation, data_prior_test = split_data_set(param_problem['prior'])
    algorithm, algorithm_with_iterates, std_hyperparams = specify_algorithm_to_be_used(smoothness_constant=L_max,
                                                                                       strong_convexity_constant=mu_min)
    emp_conv_risk, conv_check, emp_conv_est = get_estimators(algorithm, loss_func, grad_func, x_0, num_iterations)
    sufficient_statistics = get_sufficient_statistics(emp_conv_est=emp_conv_est, num_problems_train=num_problems_train)
    natural_parameters = get_natural_parameters()

    width = specify_plot_layout()
    linecolor, dot_color = specify_colors()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(width))

    for k in range(len(convergence_probabilities)):

        conv_prob = convergence_probabilities[k]
        print("\nConvergence Probability = {:.2f}$".format(conv_prob.item()))

        prior = create_data_dependent_prior(
            smoothness_constant=L_max, conv_prob=conv_prob, std_hyperparams=std_hyperparams,
            suff_stat=sufficient_statistics, nat_param=natural_parameters, num_samples_prior=num_samples_prior,
            data=data_prior_creation, eps=eps, conv_check=conv_check, test_data=data_prior_test)

        (learned_hyperparameters,
         pac_bound,
         samples_prior,
         log_prior_density,
         log_prior_marginals,
         log_posterior_density) = pac_bayes_optimizer(sufficient_statistics=sufficient_statistics,
                                                      natural_parameters=natural_parameters, priors=prior,
                                                      data=param_problem['train'], num_samples_prior=num_samples_prior,
                                                      batch_size_opt_lamb=num_problems_train, eps=eps)

        # Compare on test data
        number_of_trials = 25
        empirical_conv_probabilities = []
        for _ in range(number_of_trials):
            size_of_subsample = 250
            subsample = create_subsample_from_dataset(dataset=param_problem['test'], size=size_of_subsample)
            conv_prob_test = compute_convergence_probability(dataset=subsample,
                                                             loss_func=loss_func,
                                                             grad_func=grad_func,
                                                             algorithm_with_iterates=algorithm_with_iterates,
                                                             x_0=x_0,
                                                             learned_hyperparameters=learned_hyperparameters,
                                                             num_iterations=num_iterations)
            empirical_conv_probabilities.append(conv_prob_test)

        add_result_to_plot(ax=ax, conv_prob=conv_prob, number_of_trials=number_of_trials,
                           empirical_conv_probabilities=empirical_conv_probabilities, dot_color=dot_color)

    finalize_plot(ax=ax, linecolor=linecolor)
    savings_path = '/home/michael/Desktop/AISTATS_2023/experiments/convergence_probability/'
    plt.savefig(savings_path + 'empirical_convergence_probability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(savings_path + 'empirical_convergence_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
