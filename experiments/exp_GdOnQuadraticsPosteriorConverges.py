import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

from pac_bayes.pac_bayes_optimizer import pac_bayes_optimizer
from algorithms.optimization_algorithms import gradient_descent_const_step, gradient_descent_const_step_with_iterates
from helper.for_estimation import empirical_risk, empirical_sq_risk
from helper.for_plotting import set_size
from problems.parametric_problems import setup_random_quadratic_problems_with_fixed_curvature


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
        "ytick.labelsize": 7,
    }

    plt.rcParams.update(tex_fonts)
    return width


def specify_colors() -> Tuple[str, str, str, list]:
    prior_color = 'black'
    std_strongly_convex_color = '#ff0097'
    std_convex_color = '#fa8775'
    posterior_colors = ['#0000c5', '#0086ff', '#1acd7f', '#c5e345']
    return prior_color, std_strongly_convex_color, std_convex_color, posterior_colors


def specify_linestyles() -> List[str]:
    return ['solid', 'dashed', 'dotted', 'dashdot']


def get_pac_parameters() -> Tuple[torch.Tensor, int]:
    eps = torch.tensor(0.01)
    num_samples_prior = 500
    return eps, num_samples_prior


def specify_dimension_and_get_initial_point() -> torch.Tensor:
    dim = 50
    x_0 = torch.zeros(dim)
    return x_0


def get_loss_functions(dimension: int,
                       n_prior: int,
                       n_train: int,
                       n_test: int) -> Tuple[dict, Callable, Callable, torch.Tensor, torch.Tensor]:
    parameters, loss_func, grad_func, lamb_min, lamb_max = setup_random_quadratic_problems_with_fixed_curvature(
        dim=dimension, n_prior=n_prior, n_train=n_train, n_test=n_test)
    return parameters, loss_func, grad_func, lamb_min, lamb_max


def get_algorithms() -> Tuple[Callable, Callable]:
    return gradient_descent_const_step, gradient_descent_const_step_with_iterates


def get_standard_hyperparameters(lamb_min: torch.Tensor, lamb_max: torch.Tensor) -> torch.Tensor:
    return 2 / (lamb_min + lamb_max)


def define_prior_distribution(lamb_max: torch.Tensor, alpha_std: torch.Tensor) -> Tuple[dict, dict]:
    prior = {'alpha': torch.distributions.normal.Normal(loc=torch.mean(torch.stack([1/lamb_max, alpha_std])),
                                                        scale=0.2 * alpha_std)}
    prior_dict = {'priors': prior}
    return prior, prior_dict


def get_sufficient_statistics(algorithm: Callable,
                              loss_func: Callable,
                              grad_func: Callable,
                              x_0: torch.Tensor,
                              num_it: int,
                              lamb_min: torch.Tensor,
                              lamb_max: torch.Tensor,
                              n_train: int) -> Callable:

    # Set up function rho and corresponding constant. Note that this rho only holds for gradient descent
    def rho(hyperparam):
        return torch.max(torch.stack([abs(1 - hyperparam['alpha'] * lamb_min),
                                      abs(1 - hyperparam['alpha'] * lamb_max)])) ** (2 * num_it)

    c = 1

    # Get empirical risk functional
    emp_risk = empirical_risk(algorithm, loss_func, grad_func, x_0, num_it)
    emp_sq_risk = empirical_sq_risk(algorithm, loss_func, grad_func, x_0, num_it)

    return lambda hyperparam, data: torch.tensor(
        [-emp_risk(hyperparam, data), c ** 2 * rho(hyperparam) ** 2 * emp_sq_risk(hyperparam, data) / n_train])


def get_natural_parameters() -> Callable:
    return lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])


def get_samples_and_density_for_plotting(samples_prior: torch.Tensor,
                                         log_density: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    idx_sorted = torch.argsort(samples_prior)
    return samples_prior[idx_sorted], torch.exp(log_density[idx_sorted])


# Experiment to show evolving posterior distribution.
# This is done to show that for an increasing number of steps, the constant step-size distribution should converge (
# peak around) the analytical optimal step-size.
def exp_posterior_distribution_of_const_step_gradient_descent():

    # Specify style for paper
    width = specify_plot_layout()
    prior_color, std_strongly_convex_color, std_convex_color, posterior_colors = specify_colors()
    linestyles = specify_linestyles()

    # Specify setup of experiment
    iterations_to_test = [5, 15, 45, 135]
    x_0 = specify_dimension_and_get_initial_point()
    eps, num_samples_prior = get_pac_parameters()
    parameters, loss_func, grad_func, lamb_min, lamb_max = get_loss_functions(
        dimension=x_0.shape[0], n_prior=50, n_train=200, n_test=50)
    algorithm, algorithm_with_iterates = get_algorithms()
    alpha_std = get_standard_hyperparameters(lamb_min=lamb_min, lamb_max=lamb_max)
    n_train = len(parameters['train'])

    # Setup dictionary for prior (in the general way of implementation)
    priors, prior_dict = define_prior_distribution(lamb_max=lamb_max, alpha_std=alpha_std)
    hyperparameter_name = list(prior_dict['priors'].keys())[0]
    samples_prior, log_prior_marginals = {}, {}  # This is just to supress a warning. Will be overwritten in the loop.

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(width))
    for i, num_it in enumerate(iterations_to_test):

        print("\nNumber of Iterations = {}".format(num_it))

        sufficient_statistics = get_sufficient_statistics(
            algorithm=algorithm, loss_func=loss_func, grad_func=grad_func, x_0=x_0, num_it=num_it, lamb_min=lamb_min,
            lamb_max=lamb_max, n_train=n_train)
        natural_parameters = get_natural_parameters()

        # Get samples_prior from prior, as well as prior and posterior density on those samples_prior
        _, _, samples_prior, log_prior_density, log_prior_marginals, log_posterior_density = pac_bayes_optimizer(
            sufficient_statistics=sufficient_statistics, natural_parameters=natural_parameters, priors=priors,
            data=parameters['train'], num_samples_prior=num_samples_prior, batch_size_opt_lamb=n_train, eps=eps)

        samples_plotting, density_posterior_plotting = get_samples_and_density_for_plotting(
            samples_prior=samples_prior[hyperparameter_name], log_density=log_posterior_density)
        ax.plot(samples_plotting, density_posterior_plotting,
                color=posterior_colors[i], linestyle=linestyles[i], label='$n_{\\rm{it}}$ = ' + str(num_it))

    samples_plotting, density_prior_plotting = get_samples_and_density_for_plotting(
            samples_prior=samples_prior[hyperparameter_name], log_density=log_prior_marginals[hyperparameter_name])
    ax.plot(samples_plotting, density_prior_plotting, color=prior_color, label='prior')
    ax.axvline(alpha_std.item(), 0, 1, color=std_strongly_convex_color, linestyle='dashed',
               label='$\\alpha_{\\rm{std}}$')
    ax.axvline(1/lamb_max, 0, 1, color=std_convex_color, linestyle='dashed', label='$\\frac{1}{L}$')
    ax.set(xlabel='$\\alpha$')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid('on')
    ax.legend()

    PATH = '/home/michael/Desktop/AISTATS_2023/experiments/convergence_of_posterior/'
    plt.savefig(PATH + 'GD_posterior_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()
