

import numpy as np
import torch
from tqdm import tqdm


def setup_f(emp_suff_stat, natural_param, eps, num_test_points):
    return lambda lamb: -(torch.log(torch.mean(torch.exp(torch.matmul(emp_suff_stat, natural_param(lamb)))))
                          + torch.log(eps) - torch.log(num_test_points)) / natural_param(lamb)[0]


def get_log_posterior_density_on_samples(log_prior_density, emp_suff_stat, nat_param):
    return log_prior_density + torch.matmul(emp_suff_stat, nat_param).reshape(log_prior_density.shape) -\
           torch.log(torch.mean(torch.exp(torch.matmul(emp_suff_stat, nat_param))))


def optimize_F(emp_suff_stat, nat_param, eps):

    # Setup function F(lambda)
    num_test_points = torch.tensor(25000)
    F = setup_f(emp_suff_stat, nat_param, eps, num_test_points)

    # Optimize F:
    # Note that F is quantile_distance one-dimensional function, so computationally this is not too much of quantile_distance problem.
    test_points = torch.linspace(1e-8, 1, num_test_points)
    F_test = torch.stack([F(x) for x in test_points])

    # Take minimizer of gridsearch as starting point
    l_0 = test_points[torch.argmin(F_test)]

    return l_0, F


def pac_bayes_optimizer(
        suff_stat, nat_param,
        priors, data, num_samples_prior, batch_size_opt_lamb, eps,
):

    num_problems = len(data)
    assert batch_size_opt_lamb <= len(data), "Batch size larger than training set."

    # Create samples_prior from prior and store the corresponding log-density.
    # Note that an independence assumption is made here: the joint density of the hyperparameter factorizes into the marginal densities.
    samples_prior = {hyperparam: priors[hyperparam].sample((num_samples_prior,)) for hyperparam in priors.keys()}
    log_prior_marginals = {hyperparam: priors[hyperparam].log_prob(samples_prior[hyperparam]) for hyperparam in priors.keys()}

    # Compute sufficient statistic on "enough" sample from the prior
    # Note that computing the emp_suff_stat is (one of) the most expensive parts of the algorithm, especially if
    # smaples_prior is large: the algorithm has to run n_samples * N_train times!
    # ==> Thus, could use (Batch-) SGD. But this does not retain the PAC-Bound!
    # Compute empirical sufficient statistics
    pbar = tqdm(range(num_samples_prior))
    pbar.set_description('Computing Sufficient Statistics')
    emp_suff_stat = torch.stack(
        [suff_stat({key: value[i] for key, value in samples_prior.items()},
                   data[np.random.choice(np.arange(0, num_problems), replace=False, size=batch_size_opt_lamb)])
         for i in pbar])

    # Filter inf and -inf
    emp_suff_stat[torch.isinf(emp_suff_stat[:, 0]), 0] = -1e7  # First row is negative
    emp_suff_stat[torch.isnan(emp_suff_stat[:, 0]), 0] = -1e7
    emp_suff_stat[torch.isinf(emp_suff_stat[:, 1]), 1] = 1e7   # Second row is positive
    emp_suff_stat[torch.isnan(emp_suff_stat[:, 1]), 1] = 1e7

    lamb_opt, F = optimize_F(emp_suff_stat, nat_param, eps)

    # Compute posterior density
    log_prior_density = torch.sum(torch.stack([p for p in log_prior_marginals.values()]), axis=0)
    log_f_post = get_log_posterior_density_on_samples(log_prior_density, emp_suff_stat, nat_param(lamb_opt))

    # Normalize to get density over samples_prior
    log_f_post = log_f_post - torch.log(torch.sum(torch.exp(log_f_post)))
    log_prior_marginals = {hyperparam: priors[hyperparam].log_prob(samples_prior[hyperparam])
                                       - torch.log(
        torch.sum(torch.exp(priors[hyperparam].log_prob(samples_prior[hyperparam])))) for hyperparam in priors.keys()}

    # Select best hyperparameter
    idx_learned = torch.argmax(log_f_post)
    hyperparams_learned = {hyperparam: samples_prior[hyperparam][idx_learned] for hyperparam in samples_prior.keys()}

    return hyperparams_learned, F(lamb_opt), samples_prior, log_prior_density, log_prior_marginals, log_f_post





    # # Take exp(log())-transform to ensure positivity
    # log_lamb = torch.nn.Parameter(torch.log(l_0.requires_grad_(True)))
    # tradeoff_param = torch.nn.ParameterList([log_lamb])
    #
    # # Setup optimizer
    # optimizer = torch.optim.SGD(tradeoff_param, lr=1e-2 * l_0)
    #
    # # Progress bar and optimization
    # pbar = tqdm(range(n_max_iter))
    # pbar.set_description('Optimizing Lambda')
    # for i in pbar:
    #
    #     # Store old lambda for convergence check
    #     lamb_old = tradeoff_param[0].clone().detach()
    #
    #     # Compute loss
    #     loss = F(torch.exp(tradeoff_param[0]))
    #     optimizer.zero_grad()
    #     loss.backward()
    #
    #     # print(torch.exp(tradeoff_param[0]), loss, tradeoff_param[0].grad, "\n")
    #
    #     if torch.isnan(tradeoff_param[0].grad.data.norm(2)):
    #         print("Optimization of Lambda Failed!")
    #         break
    #
    #     optimizer.step()
    #
    #     if tradeoff_param[0].grad.data.norm(2) < 1e-3 or torch.abs(
    #             torch.exp(lamb_old) - torch.exp(tradeoff_param[0].detach())) < 1e-5 * l_0:
    #         break
    #
    # # Store optimal lambda
    # lamb_opt = torch.exp(tradeoff_param[0].detach())
    #
    # # Sanity Check by visualizing (not necessary)
    # if plot:
    #     xes = torch.linspace(0.1 * lamb_opt, 10, 10000).cpu()
    #     #xes = torch.linspace(0.1 * lamb_opt, 10 * lamb_opt, 1000).cpu()
    #     F_xes = [F(x) for x in xes]
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    #     ax.plot(xes, F_xes)
    #     ax.scatter(torch.exp(tradeoff_param[0].detach()).cpu(), F(torch.exp(tradeoff_param[0].detach())).cpu(), color='red')
    #     ax.set(title='F', xlabel='$\lambda$')
    #
    # return lamb_opt, F