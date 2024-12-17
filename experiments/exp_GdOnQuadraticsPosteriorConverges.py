import torch
import matplotlib.pyplot as plt

from pac_bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer
from algorithms.optimization_algorithms import gradient_descent_const_step, gradient_descent_const_step_with_iterates
from helper.helper_functions import empirical_risk, empirical_sq_risk, set_size
from problems.parametric_problems import setup_random_quadratic_problems_with_fixed_curvature


# Experiment to show evolving posterior distibution.
# This is done to show that for an increasing number of steps, the constant step-size distribution should converge (
# peak around) the analytical optimal step-size.
def exp_const_step_gd_distribution():

    # # Setup style for paper
    prior_color = 'black'
    std_scls_color = '#ff0097'
    std_cls_color = '#fa8775'
    iterations_to_test = [5, 15, 45, 135]
    # iterations_to_test = [5, 5, 5, 5]
    posterior_colors = ['#0000c5', '#0086ff', '#1acd7f', '#c5e345']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    #mark_every = [35, 45, 55, 65]
    #markers = ['s', 'p', 'P', '*']
    width = 234.8775    # AISTATS
    #width = 469.75499   # Arxiv
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

    # Get quadratic problem
    dim = 50
    N_prior, N_train, N_test = 50, 200, 50
    num_samples_prior = 500
    param_problem, loss_func, grad_func, lamb_min, lamb_max = setup_random_quadratic_problems_with_fixed_curvature(
        dim=dim, n_prior=N_prior, n_train=N_train, n_test=N_test, n_validation=0)
    x_0 = torch.zeros(dim)
    eps = torch.tensor(0.01)

    # Setup algorithm
    algorithm, algorithm_with_iterates = gradient_descent_const_step, gradient_descent_const_step_with_iterates

    # Get standard hyperparameter
    alpha_std = 2 / (lamb_min + lamb_max)

    # Setup dictionary for prior (in the general way of implementation)
    priors = {'alpha': torch.distributions.normal.Normal(loc=torch.mean(torch.stack([1/lamb_max, alpha_std])),
                                                         scale=0.2 * alpha_std)}
    prior_dict = {'priors': priors}
    hyperparam_names = prior_dict['priors'].keys()
    n_train = len(param_problem['train'])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(width))
    for i, num_it in enumerate(iterations_to_test):

        print("Number of Iterations = {}".format(num_it))

        # Set up function rho and corresponding constant. Note that this rho only holds for gradient descent
        # CHECK THIS FUNCTION!
        rho = lambda hyperparam: torch.max(torch.stack([abs(1 - hyperparam['alpha'] * lamb_min),
                                                        abs(1 - hyperparam['alpha'] * lamb_max)])) ** (2 * num_it)
        c = 1

        # Get empirical risk functional
        emp_risk = empirical_risk(algorithm, loss_func, grad_func, x_0, num_it)
        emp_sq_risk = empirical_sq_risk(algorithm, loss_func, grad_func, x_0, num_it)

        # Construct natural parameters and sufficient statistics
        suff_stat = lambda hyperparam, data: torch.tensor(
            [-emp_risk(hyperparam, data), c**2 * rho(hyperparam)**2 * emp_sq_risk(hyperparam, data) / n_train])
        nat_param = lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])

        # Get samples_prior from prior, as well as prior and posterior density on those samples_prior
        _, _, samples_prior, log_prior_density, log_prior_marginals, log_posterior_density = pac_bayes_optimizer(
            suff_stat=suff_stat, nat_param=nat_param, priors=priors,
            data=param_problem['train'],
            num_samples_prior=torch.tensor(num_samples_prior),
            batch_size_opt_lamb=n_train,
            eps=eps)
        hp = list(hyperparam_names)[0]
        idx_sorted = torch.argsort(samples_prior[hp])
        ax.plot(samples_prior[hp][idx_sorted].cpu(), torch.exp(log_posterior_density[idx_sorted]).cpu(),
                color=posterior_colors[i], linestyle=linestyles[i], label='$n_{it}$ = ' + str(num_it))

        print("\n")

    ax.plot(samples_prior[hp][idx_sorted].cpu(), torch.exp(log_prior_marginals[hp][idx_sorted]).cpu(),
            color=prior_color, label='prior')
    ax.axvline(2 / (lamb_min + lamb_max), 0, 1,
               color=std_scls_color, linestyle='dashed', label='$\\alpha_{std}$')
    ax.axvline(1 / lamb_max, 0, 1,
               color=std_cls_color, linestyle='dashed', label='$\\frac{1}{L}$')
    ax.set(xlabel='$\\alpha$')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend()

    PATH = '/home/michael/Desktop/AISTATS_2023/experiments/convergence_of_posterior/'
    plt.savefig(PATH + 'GD_posterior_distribution.pdf', dpi=300, bbox_inches='tight')

    plt.show()
