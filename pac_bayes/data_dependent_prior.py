from typing import Callable, Tuple
import torch
from tqdm import tqdm
from pac_bayes.pac_bayes_optimizer import pac_bayes_optimizer


def filter_samples_based_on_guarantee(samples_prior: dict,
                                      num_samples_prior: int,
                                      conv_check: Callable,
                                      test_data: list,
                                      convergence_probability: torch.Tensor,
                                      log_posterior_density: torch.Tensor) -> Tuple[dict, torch.Tensor]:

    pbar = tqdm(range(num_samples_prior))
    pbar.set_description("Convergence Check")
    emp_conv_prob = torch.stack([torch.mean(conv_check({key: value[i] for key, value in samples_prior.items()},
                                                       test_data)) for i in pbar])
    idx_convergence_guarantee_satisfied = (emp_conv_prob >= convergence_probability)
    assert torch.sum(idx_convergence_guarantee_satisfied) > 1, "Couldn't satisfy convergence probability."
    log_posterior_density = log_posterior_density[idx_convergence_guarantee_satisfied]
    samples_prior = {hyperparam: samples_prior[hyperparam][idx_convergence_guarantee_satisfied]
                     for hyperparam in samples_prior.keys()}
    return samples_prior, log_posterior_density


def select_good_samples(samples_prior: dict, log_posterior_density: torch.Tensor) -> dict:
    k = max(int(0.75 * len(log_posterior_density)), 2)
    idx_good_samples = torch.topk(torch.t(log_posterior_density), k=k).indices
    good_samples = {hyperparam: samples_prior[hyperparam][idx_good_samples] for hyperparam in samples_prior.keys()}
    return good_samples


def build_new_prior(old_prior: dict, prior_dict: dict, good_samples: dict) -> dict:
    prior = {hyperparam: prior_dict['priors'][hyperparam](
        *[prior_dict['estimator'][hyperparam][param](good_samples[hyperparam])
          for param in prior_dict['prior_params'][hyperparam]])
        for hyperparam in old_prior.keys()}
    return prior


def iterative_prior(prior: dict,
                    prior_dict: dict,
                    sufficient_statistics: Callable,
                    natural_parameters: Callable,
                    num_samples_prior: int,
                    data: list,
                    batch_size_opt_lamb: int,
                    eps: torch.Tensor,
                    num_it: int,
                    conv_check: Callable,
                    convergence_probability: torch.Tensor,
                    test_data: list) -> Tuple[dict, list]:

    list_of_priors = [prior]
    for i in range(num_it):

        learned_hyperparameters, _, samples_prior, _, _, log_posterior_density = pac_bayes_optimizer(
            sufficient_statistics, natural_parameters,
            prior, data, num_samples_prior, batch_size_opt_lamb, eps)
        samples_prior, log_posterior_density = filter_samples_based_on_guarantee(
            samples_prior=samples_prior,
            num_samples_prior=num_samples_prior,
            conv_check=conv_check,
            test_data=test_data,
            convergence_probability=convergence_probability,
            log_posterior_density=log_posterior_density)
        good_samples = select_good_samples(samples_prior=samples_prior, log_posterior_density=log_posterior_density)
        prior = build_new_prior(old_prior=prior, prior_dict=prior_dict, good_samples=good_samples)
        list_of_priors.append(prior)

    return prior, list_of_priors
