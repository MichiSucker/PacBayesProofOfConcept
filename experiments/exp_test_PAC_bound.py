import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PAC_Bayes.optimization_algorithms import heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
from Problems.parametric_problems import setup_quadratic_with_variable_curvature_with_rand_perm
from PAC_Bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer
from PAC_Bayes.Prior import iterative_prior
from PAC_Bayes.Helper.helper_functions import empirical_convergence_risk, converged, empirical_conv_estimates, set_size


def get_sufficient_statistics(emp_conv_est, num_problems_train):
    def f(hyperparam, data):
        e_conv_risk, e_sq_conv_risk, e_init_loss, e_init_sq_loss, e_conv_prob = emp_conv_est(hyperparam, data)
        return torch.tensor([-e_conv_risk, e_init_sq_loss / num_problems_train / e_conv_prob ** 2])

    return f


def exp_PAC_bound():

    # Setup
    dim = 50
    conv_prob = 0.9
    eps = torch.tensor(0.01)
    num_iterations = 50
    num_samples_prior = 200
    x_0 = torch.zeros(dim)
    num_problems_prior, num_problems_train, num_problems_test = 200, 1000, 200

    # Setup style for paper
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
    color_mean = '#405f9a'
    color_pac = '#f2c05d'
    color_learned = '#0bb4ff'
    color_std = '#e60049'

    plt.rcParams.update(tex_fonts)

    # Create problems with differing L-smoothness (as this is mainly important for convergence speed)
    mu = torch.tensor(0.05)
    L_min, L_max = torch.tensor(1.0), torch.tensor(5000.0)

    # Sample L-smoothness constants
    L = torch.distributions.uniform.Uniform(low=L_min, high=L_max).sample((num_problems_prior + num_problems_train + num_problems_test,))
    # L = (L_max - L_min) * torch.distributions.beta.Beta(0.5, 0.5).sample(
    #     (num_problems_prior + num_problems_train + num_problems_test,)) + L_min
    # Also repeat mu to reuse function below (setup_quadratic_with_variable_curvature)
    mu = torch.ones((num_problems_prior + num_problems_train + num_problems_test,)) * mu

    # Create quadratic problems with variing right-hand side and operator
    param_problem, loss_func, grad_func = setup_quadratic_with_variable_curvature_with_rand_perm(dim=dim,
                                                                                                 N_prior=num_problems_prior,
                                                                                                 N_train=num_problems_train,
                                                                                                 N_test=num_problems_test,
                                                                                                 mu=mu, L=L)

    # Get algorithm, standard hyperparameter and dictionary for prior
    algorithm, algorithm_with_iterates = heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
    q = (torch.sqrt(L_max) - torch.sqrt(mu[0])) / (torch.sqrt(L_max) + torch.sqrt(mu[0]))
    std_hyperparams = {'alpha': 4/(torch.sqrt(L_max) + torch.sqrt(mu[0]))**2 * torch.ones(1),
                       'beta': q**2 * torch.ones(1)}

    # Get empirical risk functional and estimator for convergence probability
    emp_conv_risk = empirical_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations)
    conv_check = converged(algorithm, loss_func, grad_func, x_0, num_iterations)
    emp_conv_est = empirical_conv_estimates(algorithm, loss_func, grad_func, x_0, num_iterations)

    print("Risk of std. Hyperparameters = {}".format(emp_conv_risk(std_hyperparams, param_problem['prior'])))

    # Construct sufficient statistics and natural parameters
    suff_stat = get_sufficient_statistics(emp_conv_est=emp_conv_est, num_problems_train=num_problems_train)
    nat_param = lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(width))


    priors = {'alpha': torch.distributions.uniform.Uniform, 'beta': torch.distributions.uniform.Uniform}
    prior = {'alpha': torch.distributions.uniform.Uniform(low=0.5 * (2 /L_max) * torch.ones(1),
                                                          high=3.0 * (2 /L_max) * torch.ones(1)),
             'beta': torch.distributions.uniform.Uniform(low=0.5 * std_hyperparams['beta'].clone(),
                                                         high=2.0 * std_hyperparams['beta'].clone())}
    prior_params = {'alpha': {'low': 0.5 * (2 / L_max) * torch.ones(1),
                              'high': 2.0 * (2 / L_max) * torch.ones(1)},
                    'beta': {'low': 0.5 * std_hyperparams['beta'].clone(), 'high': 2.0 * std_hyperparams['beta'].clone()}}
    estimators = {'alpha': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)},
                  'beta': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)}}

    prior_dict = {
        'priors': priors,
        'prior_params': prior_params,
        'estimator': estimators
    }

    # Create quantile_distance data-dependent prior on quantile_distance prior-dataset
    prior, list_of_priors = iterative_prior(prior, prior_dict,
                                            suff_stat, nat_param,
                                            num_samples_prior,
                                            param_problem['prior'][0:int(len(param_problem)/2)],
                                            batch_size_opt_lamb=len(param_problem['prior'][0:int(len(param_problem)/2)]),
                                            eps=eps, num_it=2,
                                            conv_check=conv_check, conv_prob=conv_prob,
                                            test_data=param_problem['prior'][int(len(param_problem)/2):])

    # Run the PAC-Bayes optimization
    learned_hyperparameters, pac_bound, samples_prior, log_prior_density, log_prior_marginals, log_posterior_density = pac_bayes_optimizer(
        suff_stat, nat_param,
        prior, param_problem['train'], num_samples_prior, batch_size_opt_lamb=num_problems_train, eps=eps)

    # Validation
    # Compute error over time
    iterates_learned = torch.zeros((num_problems_test, num_iterations + 1))
    iterates_std = torch.zeros((num_problems_test, num_iterations + 1))
    convergence = 0

    # Compare on test data
    for i, p in enumerate(param_problem['test']):

        # First, check for divergence
        it_learned = torch.stack(
            [loss_func(a, p) for a in
             algorithm_with_iterates(x_0, p, grad_func, learned_hyperparameters, num_iterations)])
        if it_learned[-1] > it_learned[0] or torch.isnan(it_learned[-1]):
            continue
        # If it didn't diverge, compare to standard version
        else:
            iterates_std[i, :] = torch.stack(
                [loss_func(a, p) for a in algorithm_with_iterates(x_0, p, grad_func, std_hyperparams, num_iterations)])
            iterates_learned[i, :] = it_learned
            convergence += 1

    df = pd.DataFrame({
        'std': iterates_std[:, -1],
        'learned': iterates_learned[:, -1]
    })

    sns.histplot(x='std', data=df, label='$\\alpha_{std}$', color=color_std)
    sns.histplot(x='learned', data=df, label='$\\alpha_{pac}$', color=color_learned)
    ax.axvline(torch.mean(iterates_learned[:, -1]), 0, 1, color=color_mean, linestyle='dashed', label='$\hat{\mathcal{R}}_{test}(\\alpha_{pac})$')
    ax.axvline(pac_bound, 0, 1, color=color_pac, linestyle='dotted', label='PAC-Bound')
    ax.set(xlabel='$l(\\alpha, \\theta_i)$')
    ax.legend()

    PATH = '/home/michael/Desktop/Figures/AISTATS2023/'
    plt.savefig(PATH + 'HB_PAC_bound.pdf', dpi=300, bbox_inches='tight')

    plt.show()
