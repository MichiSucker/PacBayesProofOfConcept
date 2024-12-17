
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from pac_bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer
from helper.helper_functions import converged

def prior_by_ERM(hyperparameters, n_max_iter, emp_conv_risk, data, conv_check, conv_prob):

    train, test = data[:int(len(data)/2)], data[int(len(data)/2):]
    optim_parameter = [torch.nn.Parameter(hp, requires_grad=True) for hp in hyperparameters.values()]
    lr = 5e-5
    optimizer = torch.optim.Adam(optim_parameter, lr=lr)
    convergence_check = conv_check(
        {hp_name: optim_parameter[j] for j, hp_name in enumerate(hyperparameters.keys())},
        test)

    assert torch.mean(convergence_check).item() >= conv_prob, "Can't guarantee convergence probability."

    pbar = tqdm(range(n_max_iter))
    pbar.set_description("Optimizing Prior")
    for i in pbar:

        old_params = [hp.clone().detach() for hyperparam in optimizer.param_groups for hp in hyperparam['params']]

        loss = emp_conv_risk({hp_name: optim_parameter[j] for j, hp_name in enumerate(hyperparameters.keys())},
                             train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        convergence_check = conv_check(
            {hp_name: optim_parameter[j] for j, hp_name in enumerate(hyperparameters.keys())},
            test)

        grad_norm = torch.sum(torch.stack([optimizer.param_groups[i]['params'][j].grad.norm(2) for i in range(len(optimizer.param_groups)) for j in range(len(optimizer.param_groups[i]['params']))]))

        if i % 25 == 0:
            print("Loss = {:.2f}, Grad. Norm = {:.2f}, Conv. Prob. = {:.2f}, Hyperparam = {}".format(
                loss, grad_norm, torch.mean(convergence_check), optimizer.param_groups))

        if loss <= 1e-8 or grad_norm <= 1e-8 or 0.95 * conv_prob <= torch.mean(convergence_check) <= 1.1 * conv_prob:
            print("Loss = {:.2f}, Emp. Conv. Prob. = {:.2f}, Conv. Prob. = {:.2f}".format(loss, torch.mean(convergence_check), conv_prob))
            break

        elif torch.mean(convergence_check) < 0.95 * conv_prob:
            print("Reinitialize: Loss = {:.2f}, Grad. Norm = {:.2f}, Emp. Conv. Prob. = {:.2f}, Conv. Prob. = {:.2f}".format(
                loss, grad_norm, torch.mean(convergence_check), conv_prob))
            print("Old Params. = {}\n".format(old_params))
            optim_parameter = [torch.nn.Parameter(hp, requires_grad=True) for hp in old_params]
            lr = 0.9 * lr
            optimizer = torch.optim.Adam(optim_parameter, lr=lr)

    return {hp_name: optim_parameter[j].detach() for j, hp_name in enumerate(hyperparameters.keys())}


def iterative_prior(prior, prior_dict, suff_stat, nat_param, num_samples_prior, data, batch_size_opt_lamb,
                    eps, num_it, conv_check, conv_prob, test_data):

    list_of_priors = [prior]

    for i in range(num_it):

        # Run the PAC-Bayes optimization with prior
        learned_hyperparameters, _, samples_prior, _, _, log_posterior_density = pac_bayes_optimizer(
            suff_stat, nat_param,
            prior, data, num_samples_prior, batch_size_opt_lamb, eps)

        # Filter for convergence guarantee
        pbar = tqdm(range(num_samples_prior))
        pbar.set_description("Convergence Check")
        emp_conv_prob = torch.stack([torch.mean(conv_check({key: value[i] for key, value in samples_prior.items()},
                                                           test_data)) for i in pbar])
        # print(emp_conv_prob)
        idx_conv_guarantee = (emp_conv_prob >= conv_prob)
        assert torch.sum(idx_conv_guarantee) > 1, "Couldn't satisfy convergence probability."
        log_posterior_density = log_posterior_density[idx_conv_guarantee]
        samples_prior = {hyperparam: samples_prior[hyperparam][idx_conv_guarantee] for hyperparam in samples_prior.keys()}

        # Filter samples_prior according to their posterior density
        k = max(int(0.75 * len(log_posterior_density)), 2)
        idx_good_samples = torch.topk(torch.t(log_posterior_density), k=k).indices
        good_samples = {hyperparam: samples_prior[hyperparam][idx_good_samples] for hyperparam in samples_prior.keys()}

        prior = {hyperparam: prior_dict['priors'][hyperparam](
            *[prior_dict['estimator'][hyperparam][param](good_samples[hyperparam])
              for param in prior_dict['prior_params'][hyperparam]]) for hyperparam in prior.keys()}

        list_of_priors.append(prior)

    return prior, list_of_priors


def optimize_prior(prior_dict, emp_risk, param_problem, batch_size, n_max_iter, n_samples):

    # Fix prior
    priors = prior_dict['priors']
    prior_params = prior_dict['prior_params']
    transformations = prior_dict['transformations']
    req_grad = prior_dict['require_grad']
    param_lr = prior_dict['lr']

    # Specify optimization parameters and optimizer
    # Note that each optimization parameter can potentially get quantile_distance different learning rate.
    optim_parameter = {
        hyperparam: {param: torch.nn.Parameter(prior_params[hyperparam][param], requires_grad=req_grad[hyperparam][param]) for param in prior_params[hyperparam].keys()}
        for hyperparam in priors.keys()}
    optimizer = torch.optim.SGD(
        [{"params": optim_parameter[hyperparam][param], "lr": param_lr[hyperparam][param]}
          for hyperparam in priors.keys() for param in prior_params[hyperparam]], lr=1e-12
    )

    avg_loss, avg_grad_norm, avg_change = [], [], []
    old_avg_loss, old_avg_grad_norm = 1e8, 1e8
    pbar = tqdm(range(n_max_iter))
    pbar.set_description("Optimizing Prior")
    for i in pbar:

        # Store old parameters for convergence check
        param_old = torch.stack([p['params'][0].clone().detach() for p in optimizer.param_groups])

        # Sample hyperparameter for the algorithm from prior distribution with current parameters
        # for pathwise gradient estimator
        hyperparams = {hyperparam: priors[hyperparam](
            *[transformations[hyperparam][param](optim_parameter[hyperparam][param])
              for param in prior_params[hyperparam].keys()]).rsample((n_samples,)) for hyperparam in priors.keys()}

        # Compute loss as empirical mean over hyperparameter
        # Note that also here we use subsampling of the dataset to increase the speed
        loss = torch.mean(torch.stack(
            [emp_risk({key: value[i] for key, value in hyperparams.items()}, param_problem['prior'][np.random.choice(np.arange(0, len(param_problem['prior'])),
                                                                  replace=False, size=batch_size)])
             for i in range(n_samples)]))

        optimizer.zero_grad()
        loss.backward()

        for g in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(g['params'], 1e4)

        #print(loss)
        #print([g['params'][0] for g in optimizer.param_groups], "\n")

        avg_loss.append(loss)
        avg_grad_norm.append(torch.sum(torch.stack([p['params'][0].grad.norm(2) if p['params'][0].grad is not None
                                                    else torch.tensor(0) for p in optimizer.param_groups])))

        if i % 1000 == 0 and i > 0:
            new_avg_loss, new_avg_grad_norm = torch.mean(torch.stack(avg_loss)), torch.mean(torch.stack(avg_grad_norm))
            print(new_avg_loss)
            avg_rel_change = torch.mean(torch.stack(avg_change))
            # If the relative change of the parameters is less than 0.1 %, stop.
            if avg_rel_change < 1e-1:
                print("Prior converged after {} iterations.".format(i))
                break
            avg_loss, avg_grad_norm, avg_change = [], [], []
            if new_avg_loss >= old_avg_loss and new_avg_grad_norm >= old_avg_grad_norm:
                for g in optimizer.param_groups:
                    g['lr'] = 0.5 * g['lr']
            old_avg_loss, old_avg_grad_norm = new_avg_loss, new_avg_grad_norm

        optimizer.step()
        avg_change.append(100 * torch.norm(optimizer.param_groups[0]['params'][0] - param_old[0]) / torch.norm(param_old[0]))

    # Save data-dependent prior
    prior_opt = {hyperparam: priors[hyperparam](
            *[transformations[hyperparam][param](optim_parameter[hyperparam][param].detach())
              for param in prior_params[hyperparam].keys()]) for hyperparam in priors.keys()}

    # Store corresponding values of the learned parameters
    prior_dict['prior_params'] = {hyperparam: {param: transformations[hyperparam][param](optim_parameter[hyperparam][param].detach()) for param in prior_params[hyperparam].keys()} for hyperparam in priors.keys()}

    return prior_opt, prior_dict


def plot_prior(algorithm_name, num_iterations, std_hyperparams, learned_hyperparams, mean_bt, mean_at, std_bt, std_at,
               samples_prior, log_prior_density, log_posterior_density):

    num_hyperparam = len(mean_bt.keys())
    fig = plt.figure(figsize=(12, 6))

    if 'const' in algorithm_name:
        for i, hyperparam in enumerate(mean_bt.keys()):
            idx_sorted = torch.argsort(samples_prior[hyperparam])
            ax = fig.add_subplot(1, num_hyperparam, i+1)
            ax.plot(samples_prior[hyperparam][idx_sorted], torch.exp(log_prior_density[idx_sorted]), color='blue', label='prior')
            ax.plot(samples_prior[hyperparam][idx_sorted], torch.exp(log_posterior_density[idx_sorted]), color='orange', label='posterior')
            ax.axvline(std_hyperparams[hyperparam], 0, 1, color='red', linestyle='dashed', label='std.')
            ax.legend()

    else:

        for i, hyperparam in enumerate(mean_bt.keys()):
            ax = fig.add_subplot(1, num_hyperparam, i+1)
            ax.plot(np.arange(num_iterations), mean_bt[hyperparam], color='gray', label='prior w/o data', linestyle='dashed')
            ax.fill_between(np.arange(num_iterations),
                            mean_bt[hyperparam] + 3 * std_bt[hyperparam],
                            mean_bt[hyperparam] - 3 * std_bt[hyperparam],
                            alpha=0.2, color='gray')
            ax.plot(np.arange(num_iterations), std_hyperparams[hyperparam], color='black', label='std.')
            ax.plot(np.arange(num_iterations), learned_hyperparams[hyperparam], color='dodgerblue', label='learned')
            ax.plot(np.arange(num_iterations), mean_at[hyperparam], color='orange', label='prior')
            ax.fill_between(np.arange(num_iterations),
                            mean_at[hyperparam] + 3 * std_at[hyperparam],
                            mean_at[hyperparam] - 3 * std_at[hyperparam], alpha=0.5,
                            color='orange')
            ax.set(title=hyperparam, xlabel='it.')
            ax.set_xticks(np.arange(1, num_iterations, int(num_iterations / 10)))
            ax.grid('on')
            ax.legend()

    plt.title(algorithm_name)

