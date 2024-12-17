
import torch


def run_algo_on_data(algorithm, loss_func, grad_func, x_0, num_iterations, data):
    return lambda hyperparam: torch.stack(
        [loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) for i in range(data.shape[0])])


def empirical_risk(algorithm, loss_func, grad_func, x_0, num_iterations):
    return lambda hyperparam, data: torch.mean(
        torch.stack([loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
            for i in range(data.shape[0])]))


def empirical_sq_risk(algorithm, loss_func, grad_func, x_0, num_iterations):
    return lambda hyperparam, data: torch.mean(
        torch.stack([loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) ** 2
                     for i in range(data.shape[0])]))


def empirical_conv_estimates(algorithm, loss_func, grad_func, x_0, num_iterations):

    def f(hyperparam, data):
        losses, sq_losses, init_loss, init_sq_loss, convergence = [], [], [], [], []

        for i in range(len(data)):
            cur_loss = loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
            cur_init_loss = loss_func(x_0, data[i])
            convergence.append(torch.tensor(1.0) if cur_loss <= cur_init_loss else torch.tensor(0.0))
            losses.append(cur_loss if convergence[-1] == 1 else torch.tensor(0.0))
            sq_losses.append(cur_loss**2 if convergence[-1] == 1 else torch.tensor(0.0))
            init_loss.append(cur_init_loss)
            init_sq_loss.append(cur_init_loss ** 2)

        return torch.mean(torch.stack(losses)) / torch.mean(torch.stack(convergence)), \
               torch.mean(torch.stack(sq_losses)) / torch.mean(torch.stack(convergence))**2, \
               torch.mean(torch.stack(init_loss)), \
               torch.mean(torch.stack(init_sq_loss)),\
               torch.mean(torch.stack(convergence))

    return f


# Replace two computations of loss with walrus operator ":=" in python 3.8
def empirical_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations):

    def f(hyperparam, data):
        losses, conv = [], []
        for i in range(len(data)):
            cur_loss = loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])
            conv.append(torch.tensor(1.0) if cur_loss <= loss_func(x_0, data[i]) else torch.tensor(0.0))
            losses.append(cur_loss if conv[-1] == 1 else torch.tensor(0.0))
        return torch.mean(torch.stack(losses)) / torch.mean(torch.stack(conv))#, torch.mean(torch.stack(conv))

    return f


def empirical_sq_convergence_risk(algorithm, loss_func, grad_func, x_0, num_iterations):

    def f(hyperparam, data):
        losses, conv = [], []
        for i in range(len(data)):
            cur_loss = loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i])**2
            conv.append(1 if cur_loss <= loss_func(x_0, data[i]) else torch.tensor(0.0))
            losses.append(cur_loss if conv[-1] == 1 else torch.tensor(0.0))
        return torch.mean(torch.stack(losses)) / torch.mean(torch.stack(conv))**2

    return f


def converged(algorithm, loss_func, grad_func, x_0, num_iterations):
    return lambda hyperparam, data: torch.stack([torch.tensor(1.0) if loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) <= loss_func(x_0, data[i]) else torch.tensor(0.0)
                     for i in range(data.shape[0])])


def empirical_conv_prob(algorithm, loss_func, grad_func, x_0, num_iterations):
    return lambda hyperparam, data: torch.mean(
        torch.stack([torch.tensor(1.0) if loss_func(algorithm(x_0, data[i], grad_func, hyperparam, num_iterations), data[i]) <= loss_func(x_0, data[i]) else torch.tensor(0.0)
                     for i in range(data.shape[0])]))


def get_loss_over_iterates(algo_with_it, loss_func, data, num_iterations, hyperparam):
    # Setup empty iterates array
    iterates = torch.zeros((len(data), num_iterations + 1))

    # Fill each line with the losses over the iterations
    for i, p in enumerate(data):
        iterates[i, :] = torch.stack([loss_func(out, p) for out in algo_with_it(hyperparam, p)])
    return iterates


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
