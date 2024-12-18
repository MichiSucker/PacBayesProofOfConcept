import torch
from tqdm import tqdm
from pac_bayes.PAC_Bayes_Optimizer import pac_bayes_optimizer


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
