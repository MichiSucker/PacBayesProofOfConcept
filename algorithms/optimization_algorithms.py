
import torch


def neural_grad_descent(x, param, grad_func, hyperparam, num_iter):
    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.
    for i in range(num_iter):
        grad = grad_func(x, param)
        x = x - hyperparam(x, grad, torch.linalg.norm(grad))
    return x


def neural_grad_descent_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]

    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.
    for i in range(num_iter):
        grad = grad_func(x, param)
        x = x - hyperparam(x, grad, torch.linalg.norm(grad))
        iterates.append(x)
    return torch.stack(iterates)


def neural_polyak_descent(x, param, grad_func, hyperparam, num_iter):

    # Init x_prev
    x_prev = x

    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.
    for i in range(num_iter):

        # Store current point
        x_cur = x.clone()

        # Update current point
        grad = grad_func(x, param)
        x = x - hyperparam(x, x_prev, grad, torch.linalg.norm(grad))

        # Update old point to last current point
        x_prev = x_cur.clone()

    return x


def neural_polyak_descent_with_iterates(x, param, grad_func, hyperparam, num_iter):

    # Init
    x_prev = x
    iterates = [x]

    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.
    for i in range(num_iter):

        # Store current point
        x_cur = x.clone()

        # Update current point
        grad = grad_func(x, param)
        x = x - hyperparam(x, x_prev, grad, torch.linalg.norm(grad))
        iterates.append(x)

        # Update old point to last current point
        x_prev = x_cur.clone()

    return torch.stack(iterates)


def neural_grad_descent_rnn(x, param, grad_func, hyperparam, num_iter):

    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.

    # Initialize hidden state (to zero)
    if hasattr(hyperparam, 'init_hidden'):
        hidden = hyperparam.init_hidden()
    else:
        hidden = torch.zeros(2 * x.shape[0])

    # Iterate for given number of iterates
    # In each iteration: Create new direction and hidden state as output
    for i in range(num_iter):
        grad = grad_func(x, param)
        dir, hidden = hyperparam(x, grad, torch.linalg.norm(grad), hidden)
        x = x - dir
    return x


def neural_grad_descent_with_iterates_rnn(x, param, grad_func, hyperparam, num_iter):

    # Initialize hidden state (to zero)
    if hasattr(hyperparam, 'init_hidden'):
        hidden = hyperparam.init_hidden()
    else:
        hidden = torch.zeros(2 * x.shape[0])

    iterates = [x]
    # The hyperparameter is quantile_distance neural network
    # that takes x and the current gradient as its inputs.
    for i in range(num_iter):
        grad = grad_func(x, param)
        dir, hidden = hyperparam(x, grad, torch.linalg.norm(grad), hidden)
        x = x - dir
        iterates.append(x)
    return torch.stack(iterates)


def hybrid_grad_descent_rnn(x, param, grad_func, hyperparam, num_iter, block_len, one_step_std_algo):

    # The hyperparameter is quantile_distance neural network
    x_old = x.clone()
    x = one_step_std_algo(x_old, param)
    hidden = torch.cat((x_old, x))

    for i in range(1, num_iter):

        if (i % (block_len+2) == 0) or (i % (block_len+2) == 1):
            x_old = x.clone()
            x = one_step_std_algo(x_old, param)
            hidden = torch.cat((x_old, x))

            #####################
            # Test
            #####################
            hidden = x - x_old
        else:
            grad = grad_func(x, param)
            dir, hidden = hyperparam(x, grad, torch.linalg.norm(grad), hidden)
            x = x - dir

    return x


def hybrid_grad_descent_with_iterates_rnn(x, param, grad_func, hyperparam, num_iter, block_len, one_step_std_algo):

    # The hyperparameter is quantile_distance neural network
    x_old = x.clone()
    x = one_step_std_algo(x_old, param)
    hidden = torch.cat((x_old, x))
    prev_dir = x - x_old
    iterates = [x_old, x]

    for i in range(1, num_iter):

        if (i % (block_len+2) == 0) or (i % (block_len+2) == 1):
            x_old = x.clone()
            x = one_step_std_algo(x_old, param)
            hidden = torch.cat((x_old, x))

            ###############################################################################
            # Tests for different hidden states
            # g, g_old = grad_func(x, param), grad_func(x_old, param)
            # hidden = torch.cat((x-x_old, (g/torch.linalg.norm(g) - (g_old / torch.linalg.norm(g_old)))))
            hidden = x-x_old
            prev_dir = x - x_old
            ################################################################################
        else:
            grad = grad_func(x, param)
            dir, hidden = hyperparam(prev_dir, grad, torch.linalg.norm(grad), hidden)
            prev_dir = dir.clone()
            x = x - dir
        iterates.append(x)
    return torch.stack(iterates)


def block_rnn(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    hidden = torch.zeros(x.shape[0])
    while len(iterates) < num_iter:
        iter, hidden = hyperparam(iterates[-1], hidden, param, grad_func)
        iterates.extend(iter)
    return iterates[-1]


def block_rnn_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    hidden = torch.zeros(x.shape[0])
    while len(iterates) < num_iter:
        iter, hidden = hyperparam(iterates[-1], hidden, param, grad_func)
        iterates.extend(iter)
    return torch.stack(iterates)


# Standard Gradient descent with sequence of step-sizes
def gradient_descent(x, param, grad_func, hyperparam, num_iter):

    for i in range(num_iter):
        x = x - hyperparam['alpha'][i] * grad_func(x, param)
    return x


def grad_descent_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    for i in range(num_iter):
        x = x - hyperparam['alpha'][i] * grad_func(x, param)
        iterates.append(x)
    return torch.stack(iterates)


# Standard Gradient descent with constant step-size
# Hyperparameter is just quantile_distance single step-size
def gradient_descent_const_step(x, param, grad_func, hyperparam, num_iter):
    for i in range(num_iter):
        x = x - hyperparam['alpha'] * grad_func(x, param)
    return x


def gradient_descent_const_step_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    for i in range(num_iter):
        x = x - hyperparam['alpha'] * grad_func(x, param)
        iterates.append(x)
    return torch.stack(iterates)


# Preconditioned gradient descent with constant diagonal preconditioner
def precond_gradient_descent_const_diag(x, param, grad_func, hyperparam, num_iter):

    for i in range(num_iter):
        x = x - torch.matmul(torch.diag(hyperparam['D']) ** 2, grad_func(x, param))
    return x


def precond_gradient_descent_const_diag_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    for i in range(num_iter):
        x = x - torch.matmul(torch.diag(hyperparam['D']) ** 2, grad_func(x, param))
        iterates.append(x)
    return torch.stack(iterates)


# Preconditioned gradient descent with constant preconditioner
def precond_gradient_descent_const(x, param, grad_func, hyperparam, num_iter):

    for i in range(num_iter):
        x = x - torch.matmul(torch.matmul(torch.transpose(hyperparam['D'], 0, 1), hyperparam['D']), grad_func(x, param))
    return x


def precond_gradient_descent_const_with_iterates(x, param, grad_func, hyperparam, num_iter):
    iterates = [x]
    for i in range(num_iter):
        x = x - torch.matmul(torch.matmul(torch.transpose(hyperparam['D'], 0, 1), hyperparam['D']), grad_func(x, param))
        iterates.append(x)
    return torch.stack(iterates)


# Heavy-Ball method with sequence of hyperparameter
def heavy_ball(x, param, grad_func, hyperparam, num_iter):
    x_old = x
    x_cur = x
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'][i] * grad_func(x_cur, param) + hyperparam['beta'][i] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
    return x_cur


def heavy_ball_with_iterates(x, param, grad_func, hyperparam, num_iter):
    x_old = x
    x_cur = x
    iterates = [x]
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'][i] * grad_func(x_cur, param) + hyperparam['beta'][i] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
        iterates.append(x_cur)
    return torch.stack(iterates)


# Heavy-ball with const. hyperparameter
def heavy_ball_const_hyperparam(x, param, grad_func, hyperparam, num_iter):
    x_old = x
    x_cur = x
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'] * grad_func(x_cur, param) + hyperparam['beta'] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
    return x_cur


def heavy_ball_const_hyperparam_with_iterates(x, param, grad_func, hyperparam, num_iter):
    x_old = x
    x_cur = x
    iterates = [x]
    for i in range(num_iter):
        x_new = x_cur - hyperparam['alpha'] * grad_func(x_cur, param) + hyperparam['beta'] * (x_cur - x_old)
        x_old = x_cur
        x_cur = x_new
        iterates.append(x_cur)
    return torch.stack(iterates)


def setup_algorithm(algorithm_name, lamb_min, lamb_max, num_iterations, dim):

    # Gradient Descent with sequence of step-sizes
    if algorithm_name == 'gradient_descent':
        alpha_std = 2 / (lamb_min + lamb_max)
        algorithm, algorithm_with_iterates = gradient_descent, grad_descent_with_iterates
        std_hyperparams = {'alpha': torch.ones(num_iterations) * alpha_std}
        priors = {'alpha': torch.distributions.multivariate_normal.MultivariateNormal}
        prior_params = {'alpha': {'loc': 0.5 * alpha_std * torch.ones(num_iterations), 'covariance_matrix': (alpha_std ** 2 / 9) * torch.ones(num_iterations)}}
        require_grad = {'alpha': {'loc': True, 'covariance_matrix': True}}
        transformations = {'alpha': {'loc': lambda p: p, 'covariance_matrix': lambda p: torch.diag(p**2)}}
        lr = {'alpha': {'loc': 1e-8, 'covariance_matrix': 1e-8}}
        estimators = {'alpha': {'loc': lambda p: torch.mean(p, axis=0), 'covariance_matrix': lambda p: torch.diag(torch.diag(torch.cov(torch.t(p))))}}
        prior_dict = {
            'priors': priors,
            'prior_params': prior_params,
            'require_grad': require_grad,
            'transformations': transformations,
            'lr': lr,
            'estimator': estimators
        }
        rho = lambda hyperparam: 1
        c = 1
        # Convergence rate bound for function values and assuming that f_star = 0
        conv_rate_bound = (lamb_max / lamb_min) * ((lamb_max - lamb_min) / (lamb_max + lamb_min)) ** (2 * num_iterations)

    # Gradient Descent with constant step-size
    elif algorithm_name == 'gradient_descent_const_hyperparam':
        alpha_std = 2 / (lamb_min + lamb_max)
        algorithm, algorithm_with_iterates = gradient_descent_const_step, gradient_descent_const_step_with_iterates
        std_hyperparams = {'alpha': alpha_std * torch.ones(1)}

        priors = {'alpha': torch.distributions.uniform.Uniform}
        prior_params = {'alpha': {'low': torch.tensor(0.0), 'high': 50 * 2 / lamb_max}}
        require_grad = {'alpha': {'low': True, 'high': True}}
        transformations = {'alpha': {'low': lambda p: p, 'high': lambda p: p}}
        lr = {'alpha': {'low': 1e-9, 'high': 1e-9}}
        estimators = {'alpha': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)}}

        # priors = {'alpha': torch.distributions.normal.Normal}
        # prior_params = {'alpha': {'mean': 0.99 * alpha_std,
        #                           'std': 5 * alpha_std}}
        # require_grad = {'alpha': {'mean': True, 'std': True}}
        # transformations = {'alpha': {'mean': lambda p: p, 'std': lambda p: p ** 2}}
        # lr = {'alpha': {'mean': 1e-9, 'std': 1e-9}}
        prior_dict = {
            'priors': priors,
            'prior_params': prior_params,
            'require_grad': require_grad,
            'transformations': transformations,
            'lr': lr,
            'estimator': estimators
        }
        # Convergence rate bound for function values and assuming that f_star = 0
        conv_rate_bound = (lamb_max / lamb_min) * ((lamb_max - lamb_min) / (lamb_max + lamb_min)) ** (
                    2 * num_iterations)

        # Set up function rho and corresponding constant. Note that this rho only holds for gradient descent (!)
        # TO-DO: Extend rho for more algorithms
        rho = lambda hyperparam: (lamb_max / lamb_min) * \
                                 torch.max(torch.stack([abs(1 - hyperparam['alpha'] * lamb_min),
                                                        abs(1 - hyperparam['alpha'] * lamb_max)])) ** (2 * num_iterations)
        rho = lambda hyperparam: 1
        c = 1

        # Gradient Descent with constant preconditioner
        # Note that the baseline here is still std. gradient descent (in lack of quantile_distance std. preconditioned version)

    elif algorithm_name == 'precond_gradient_descent_const_hyperparam':
        alpha_std = 2 / (lamb_min + lamb_max)
        algorithm, algorithm_with_iterates = precond_gradient_descent_const, precond_gradient_descent_const_with_iterates
        std_hyperparams = {'D': alpha_std * torch.eye(dim)}
        priors = {'D': torch.distributions.multivariate_normal.MultivariateNormal}
        prior_params = {'D': {'loc': 0.99 * alpha_std * torch.ones(dim),
                                  'covariance_matrix': 2 * alpha_std * torch.ones(dim)}}
        require_grad = {'D': {'loc': True, 'covariance_matrix': True}}
        transformations = {'D': {'loc': lambda p: p, 'covariance_matrix': lambda p: torch.diag(p ** 2)}}
        lr = {'D': {'loc': 1e-7, 'covariance_matrix': 1e-12}}
        prior_dict = {
            'priors': priors,
            'prior_params': prior_params,
            'require_grad': require_grad,
            'transformations': transformations,
            'lr': lr
        }
        # Convergence rate bound for function values and assuming that f_star = 0
        conv_rate_bound = (lamb_max / lamb_min) * ((lamb_max - lamb_min) / (lamb_max + lamb_min)) ** (
                2 * num_iterations)
        rho = lambda hyperparam: 1
        c = 1

    # Heavy-Ball with sequence of 2-parameters
    elif algorithm_name == 'heavy_ball':
        q = (torch.sqrt(lamb_max) - torch.sqrt(lamb_min)) / (torch.sqrt(lamb_max) + torch.sqrt(lamb_min))
        alpha_std = 4 / (torch.sqrt(lamb_max) + torch.sqrt(lamb_min)) ** 2
        beta_std = q ** 2
        algorithm, algorithm_with_iterates = heavy_ball, heavy_ball_with_iterates
        std_hyperparams = {'alpha': alpha_std * torch.ones(num_iterations),
                           'beta': beta_std * torch.ones(num_iterations)}
        priors = {'alpha': torch.distributions.multivariate_normal.MultivariateNormal,
                  'beta': torch.distributions.multivariate_normal.MultivariateNormal}
        prior_params = {'alpha': {'loc': 0.5 * alpha_std * torch.ones(num_iterations), 'covariance_matrix': 0.5 * alpha_std * torch.ones(num_iterations)},
                        'beta': {'loc': 0.5 * beta_std * torch.ones(num_iterations), 'covariance_matrix': 0.5 * beta_std * torch.ones(num_iterations)}}
        require_grad = {'alpha': {'loc': True, 'covariance_matrix': True},
                        'beta': {'loc': True, 'covariance_matrix': True}}
        transformations = {'alpha': {'loc': lambda p: p, 'covariance_matrix': lambda p: torch.diag(p ** 2)},
                           'beta': {'loc': lambda p: p, 'covariance_matrix': lambda p: torch.diag(p ** 2)}}
        lr = {'alpha': {'loc': 1e-8, 'covariance_matrix': 1e-9},
              'beta': {'loc': 1e-8, 'covariance_matrix': 1e-6}}
        estimators = {'alpha': {'loc': lambda p: torch.mean(p), 'covariance_matrix': lambda p: torch.cov(p)},
                      'beta': {'loc': lambda p: torch.mean(p), 'covariance_matrix': lambda p: torch.cov(p)}}
        prior_dict = {
            'priors': priors,
            'prior_params': prior_params,
            'require_grad': require_grad,
            'transformations': transformations,
            'lr': lr,
            'estimator': estimators
        }
        conv_rate_bound = (lamb_max / lamb_min) * q ** (2 * num_iterations)
        rho = lambda hyperparam: 1
        c = 1

    elif algorithm_name == 'heavy_ball_const_hyperparam':
        q = (torch.sqrt(lamb_max) - torch.sqrt(lamb_min)) / (torch.sqrt(lamb_max) + torch.sqrt(lamb_min))
        alpha_std = 4 / (torch.sqrt(lamb_max) + torch.sqrt(lamb_min)) ** 2
        beta_std = q ** 2
        algorithm, algorithm_with_iterates = heavy_ball_const_hyperparam, heavy_ball_const_hyperparam_with_iterates
        std_hyperparams = {'alpha': alpha_std * torch.ones(1),
                           'beta': beta_std * torch.ones(1)}
        priors = {'alpha': torch.distributions.uniform.Uniform,
                  'beta': torch.distributions.uniform.Uniform}
        prior_params = {'alpha': {'low': torch.tensor(0.0), 'high': 50 * 2 * (1 + beta_std)/lamb_max},
                        'beta': {'low': torch.tensor(0.0), 'high': 5 * torch.tensor(1.0)}}
        require_grad = {'alpha': {'low': True, 'high': True},
                        'beta': {'low': True, 'high': True}}
        transformations = {'alpha': {'low': lambda p: p, 'high': lambda p: p},
                           'beta': {'low': lambda p: p, 'high': lambda p: p}}
        lr = {'alpha': {'low': 1e-9, 'high': 1e-9},
              'beta': {'low': 1e-9, 'high': 1e-9}}
        estimators = {'alpha': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)},
                      'beta': {'low': lambda p: torch.min(p), 'high': lambda p: torch.max(p)}}

        # priors = {'alpha': torch.distributions.normal.Normal,
        #           'beta': torch.distributions.normal.Normal}
        # prior_params = {'alpha': {'mean': 0.8 * alpha_std, 'std': 5 * alpha_std},
        #                 'beta': {'mean': 0.8 * beta_std, 'std': 0.25 * beta_std}}
        # require_grad = {'alpha': {'mean': True, 'std': True},
        #                 'beta': {'mean': True, 'std': True}}
        # transformations = {'alpha': {'mean': lambda p: p, 'std': lambda p: p ** 2},
        #                    'beta': {'mean': lambda p: p, 'std': lambda p: p ** 2}}
        # lr = {'alpha': {'mean': 1e-9, 'std': 1e-9},
        #       'beta': {'mean': 1e-9, 'std': 1e-9}}
        prior_dict = {
            'priors': priors,
            'prior_params': prior_params,
            'require_grad': require_grad,
            'transformations': transformations,
            'lr': lr,
            'estimator': estimators
        }
        conv_rate_bound = (lamb_max / lamb_min) * q ** (2 * num_iterations)
        rho = lambda hyperparam: 1
        c = 1

    else:
        print("This algorithm is not yet specified.")

    return algorithm, algorithm_with_iterates, std_hyperparams, prior_dict, conv_rate_bound, rho, c

