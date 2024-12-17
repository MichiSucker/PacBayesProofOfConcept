
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from PAC_Bayes.optimization_algorithms import heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
from Problems.parametric_problems import setup_quadratic_with_variable_curvature_with_rand_perm
from PAC_Bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer
from PAC_Bayes.Prior import iterative_prior
from PAC_Bayes.Helper.helper_functions import empirical_convergence_risk, empirical_sq_convergence_risk, empirical_conv_prob, \
    converged


def exp_conditioning_HB_on_convergence():

    # Setup
    dim = 50
    eps, conv_prob = torch.tensor(0.01), torch.tensor(0.95)
    num_iterations = 50
    num_samples_prior = 100
    x_0 = torch.zeros(dim)
    num_problems_prior, num_problems_train, num_problems_test = 100, 100, 200

    # Create problems with differing strong convexity and L-smoothness
    mu_min, mu_max = torch.tensor(0.005), torch.tensor(0.5)
    L_min, L_max = torch.tensor(50.0), torch.tensor(5000.0)

    # Sample strong-convexity constant and L-smoothness constant
    mu = (mu_max - mu_min) * torch.distributions.beta.Beta(20, 1).sample(
        (num_problems_prior + num_problems_train + num_problems_test,)) + mu_min
    L = (L_max - L_min) * torch.distributions.beta.Beta(1, 20).sample(
        (num_problems_prior + num_problems_train + num_problems_test,)) + L_min

    # Create quadratic problems with variing right-hand side and operator
    param_problem, loss_func, grad_func = setup_quadratic_with_variable_curvature_with_rand_perm(dim=dim,
                                                                                                 N_prior=num_problems_prior,
                                                                                                 N_train=num_problems_train,
                                                                                                 N_test=num_problems_test,
                                                                                                 mu=mu, L=L)

    # Show distribution over mu and L
    df_mu = {'prior': mu[:num_problems_prior], 'train': mu[num_problems_prior:num_problems_prior + num_problems_train],
             'test': mu[num_problems_prior + num_problems_train:]}
    df_L = {'prior': L[:num_problems_prior], 'train': L[num_problems_prior:num_problems_prior + num_problems_train],
            'test': L[num_problems_prior + num_problems_train:]}
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.histplot(df_mu, ax=ax[0])
    sns.histplot(df_L, ax=ax[1])
    ax[0].set_title("$\mu$")
    ax[1].set_title("L")

    # Get algorithm, standard hyperparameter and dictionary for prior
    algorithm, algorithm_with_iterates = heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
    q = (torch.sqrt(L_max) - torch.sqrt(mu_min)) / (torch.sqrt(L_max) + torch.sqrt(mu_min))
    std_hyperparams = {'alpha': 4/(torch.sqrt(L_max) + torch.sqrt(mu_min))**2 * torch.ones(1),
                       'beta': q**2 * torch.ones(1)}
    priors = {'alpha': torch.distributions.uniform.Uniform, 'beta': torch.distributions.uniform.Uniform}
    prior = {'alpha': torch.distributions.uniform.Uniform(low=0.0, high=10 * std_hyperparams['alpha']),
             'beta': torch.distributions.uniform.Uniform(low=0.0, high=1 * std_hyperparams['beta'])}
    prior_params = {'alpha': {'low': torch.tensor(0.0), 'high': 100 * std_hyperparams['alpha']},
                    'beta': {'low': torch.tensor(0.0), 'high': 10 * std_hyperparams['beta']}}
    estimators = {'alpha': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)},
                  'beta': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)}}
    prior_dict = {
        'priors': priors,
        'prior_params': prior_params,
        'estimator': estimators
    }

    # Get empirical risk functional and estimator for convergence probability
    emp_conv_risk = empirical_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations)
    emp_conv_prob = empirical_conv_prob(algorithm, loss_func, grad_func, x_0, num_iterations)
    conv_check = converged(algorithm, loss_func, grad_func, x_0, num_iterations)

    # Get empirical mean of squared risk at starting point and after num_iterations
    init_sq_risk = empirical_sq_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations=0)

    # Construct sufficient statistics and natural parameters
    suff_stat = lambda hyperparam, data: torch.tensor(
        [-emp_conv_risk(hyperparam, data) / emp_conv_prob(hyperparam, data),
         init_sq_risk(hyperparam, data) / num_problems_train / emp_conv_prob(hyperparam, data) ** 2])
    nat_param = lambda lamb: torch.stack([lamb, -0.5 * lamb ** 2])

    # Create quantile_distance data-dependent prior on quantile_distance prior-dataset
    prior, list_of_priors = iterative_prior(prior, prior_dict,
                                            suff_stat, nat_param,
                                            num_samples_prior,
                                            param_problem['prior'],
                                            batch_size_opt_lamb=len(param_problem['prior']),
                                            eps=eps, num_it=3,
                                            conv_check=conv_check, conv_prob=conv_prob, test_data=param_problem['train'])

    # Run the PAC-Bayes optimization
    learned_hyperparameters, pac_bound, samples_prior, log_prior_density, log_prior_marginals, log_posterior_density = pac_bayes_optimizer(
        suff_stat, nat_param,
        prior, param_problem['train'], num_samples_prior, batch_size_opt_lamb=num_problems_train, eps=eps)
    print("Standard Step-Size = {}, Learned Step-Size = {}".format(std_hyperparams, learned_hyperparameters))

    # Get convergence probability by an empirical estimate and estimate its std deviation
    convergence_test = converged(algorithm, loss_func, grad_func, x_0, num_iterations)
    conv_data = convergence_test(learned_hyperparameters, np.concatenate((param_problem['prior'], param_problem['train'])))
    sub_samples = torch.stack([torch.mean(conv_data[np.random.randint(0, len(conv_data), size=int(0.25*len(conv_data)))])
                               for _ in range(100)])
    est_conv_prob = 100 * torch.mean(conv_data)
    std_conv_prob = 100 * torch.std(sub_samples)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    sns.histplot(sub_samples, ax=ax, stat='density')
    ax.axvline(0.01 * est_conv_prob, 0, 1, color='red', linestyle='dashed')
    ax.set(title='Estimated Convergence Probability', xlabel='p')

    print("Algorithm should converge in at least {:.2f} % of the cases.".format(est_conv_prob - 2 * std_conv_prob))

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
    conv_prob_test = 100 * convergence/len(param_problem['test'])
    print("Algorithm converged in {}% of the cases.".format(conv_prob_test))

    init_error = torch.mean(
        torch.stack([loss_func(x_0, param_problem['test'][i]) for i in range(num_problems_test)]))

    # Extract end results
    val_hyperparam_std = iterates_std[:, -1]
    val_hyperparam_learned = iterates_learned[:, -1]

    df = pd.DataFrame({
        'opt': val_hyperparam_std.cpu(),
        'learned': val_hyperparam_learned.cpu()
    })

    print("Initial Error = {:.2f}".format(init_error))
    print("Alpha Standard = {:.2f} +- {:.2f}, Relative Error = {:.2f}%".format(
        torch.mean(val_hyperparam_std), torch.std(val_hyperparam_std) / num_problems_test,
                                        100 * torch.mean(val_hyperparam_std) / init_error))
    print("Alpha Learned = {:.2f} +- {:.2f}, Relative Error = {:.2f}%".format(
        torch.mean(val_hyperparam_learned), torch.std(val_hyperparam_learned) / num_problems_test,
                                            100 * torch.mean(val_hyperparam_learned) / init_error))
    print("PAC-Bayes Bound = {:.2f}".format(pac_bound))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    sns.histplot(df, kde=True, ax=ax, stat='density')
    ax.axvline(torch.mean(val_hyperparam_learned).cpu(), 0, 1, color='green', linestyle='dashed', label='emp.mean')
    ax.axvline(pac_bound, 0, 1, color='red', linestyle='dashed', label='PAC-Bound')
    ax.set_title("Validation Error and PAC-Bound")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(np.arange(num_iterations + 1), torch.mean(iterates_std, dim=0).cpu(),
            color='black', linestyle='dashdot', label='std. mean')
    ax.plot(np.arange(num_iterations + 1), torch.median(iterates_std, dim=0).values.cpu(),
            color='black', linestyle='dotted', label='std. median')
    ax.plot(np.arange(num_iterations + 1), torch.mean(iterates_learned, dim=0).cpu(),
            color='dodgerblue', linestyle='dashdot', label='learned mean')
    ax.plot(np.arange(num_iterations + 1), torch.median(iterates_learned, dim=0).values.cpu(),
            color='dodgerblue', linestyle='dotted', label='learned median')
    ax.fill_between(np.arange(num_iterations + 1), torch.quantile(iterates_std, q=0.05, dim=0).cpu(),
                    torch.quantile(iterates_std, q=0.95, dim=0).cpu(), color='black', alpha=0.5)
    ax.fill_between(np.arange(num_iterations + 1), torch.quantile(iterates_learned, q=0.05, dim=0).cpu(),
                    torch.quantile(iterates_learned, q=0.95, dim=0).cpu(), color='dodgerblue', alpha=0.5)
    ax.set_yscale('log')

    plt.show()